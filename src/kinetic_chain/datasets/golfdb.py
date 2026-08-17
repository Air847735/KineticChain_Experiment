"""GolfDB 轉接層（**真人標註**）。

GolfDB（McNally et al., CVPR-W 2019）有 1400 段裁切好的高爾夫揮桿影片，
每段都有 8 個揮桿事件的人工標註。這是本專案唯一有真人事件標註的資料來源，
也是唯一能與已發表基準（SwingNet, PCE 76.1%）對照的評估集。

資料取得
--------
- 標註：``https://github.com/wmcnally/golfdb`` 的 ``data/golfDB.pkl``
- 影片：作者提供的 ``videos_160.zip``（160×160 已裁切，約 699 MB）

原始標註的 ``events`` 有 10 個數字：第 0 個與第 9 個是片段在原影片中的起訖影格，
中間 8 個才是事件。``videos_160`` 已經裁到這個區間，所以事件影格要減掉 ``events[0]``。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from ..data import Clip
from ..errors import DatasetError, KineticChainError
from ..skeleton import to_canonical

logger = logging.getLogger(__name__)

DATASET_NAME = "golfdb"
SPORT = "golf_swing"

#: RTMPose 輸出的關鍵點布局。
LAYOUT = "coco17"

#: GolfDB 的 8 個事件，依標註順序排列，對應到本專案的事件 id。
#: 依 McNally et al. 的定義：Address, Toe-up, Mid-backswing (arm parallel), Top,
#: Mid-downswing (arm parallel), Impact, Mid-follow-through (shaft parallel), Finish.
GOLFDB_EVENTS: tuple[str, ...] = (
    "address",
    "golf_toe_up",
    "golf_mid_backswing",
    "loading_peak",
    "golf_mid_downswing",
    "release_impact",
    "follow_through_mid",
    "finish",
)


def _cache_path(cache_dir: Path, clip_id: int) -> Path:
    return cache_dir / f"{clip_id}.npz"


def extract_poses(
    annotations_path: Path | str,
    videos_dir: Path | str,
    cache_dir: Path | str,
    *,
    device: str = "cuda",
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    """對每支影片跑一次姿態抽取並存成 ``.npz`` 快取。

    姿態抽取比訓練慢得多（1400 段各約 200 影格），一定要先跑一次存下來。
    ``videos_160`` 已經裁切到單一球員，因此用整張畫面當偵測框、完全跳過人體偵測器。

    Returns
    -------
    ``{"done": n, "skipped": n, "failed": n}``
    """
    import pandas as pd

    from ..pose import PoseExtractor, save_sequence

    videos_dir = Path(videos_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_pickle(annotations_path)
    if limit is not None:
        frame = frame.iloc[:limit]

    extractor: PoseExtractor | None = None
    stats = {"done": 0, "skipped": 0, "failed": 0}

    try:
        from tqdm import tqdm

        rows = tqdm(list(frame.itertuples()), desc="golfdb pose")
    except ImportError:
        rows = list(frame.itertuples())

    for row in rows:
        target = _cache_path(cache_dir, int(row.id))
        if target.is_file() and not overwrite:
            stats["skipped"] += 1
            continue
        video = videos_dir / f"{int(row.id)}.mp4"
        if not video.is_file():
            logger.warning("找不到影片 %s", video)
            stats["failed"] += 1
            continue
        if extractor is None:
            extractor = PoseExtractor(bbox_strategy="whole_frame", device=device)
        try:
            save_sequence(extractor.extract_video(video), target)
            stats["done"] += 1
        except KineticChainError as exc:
            logger.warning("%s 姿態抽取失敗：%s", video.name, exc)
            stats["failed"] += 1

    return stats


def load(
    annotations_path: Path | str,
    cache_dir: Path | str,
    *,
    splits: Sequence[int] | None = None,
    views: Sequence[str] | None = None,
    min_coverage: float = 0.8,
    limit: int | None = None,
) -> list[Clip]:
    """讀取 GolfDB 標註與姿態快取，產生 :class:`Clip`。

    Parameters
    ----------
    annotations_path:
        ``golfDB.pkl`` 的路徑。
    cache_dir:
        :func:`extract_poses` 產生的 ``.npz`` 目錄。缺少快取的片段會被略過。
    splits:
        只保留這些 split（GolfDB 提供 1–4 的四折）。``None`` 表示全部。
    views:
        只保留這些視角（``down-the-line`` / ``face-on`` / ``other``）。
    min_coverage:
        關鍵點信心足夠的影格比例低於此值的片段捨棄。
    """
    import pandas as pd

    from ..pose import load_sequence

    cache_dir = Path(cache_dir)
    frame = pd.read_pickle(annotations_path)
    if splits is not None:
        frame = frame[frame["split"].isin(list(splits))]
    if views is not None:
        frame = frame[frame["view"].isin(list(views))]
    if limit is not None:
        frame = frame.iloc[:limit]

    clips: list[Clip] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for row in frame.itertuples():
        path = _cache_path(cache_dir, int(row.id))
        if not path.is_file():
            skip("no_pose_cache")
            continue

        sequence = load_sequence(path)
        raw_events = np.asarray(row.events, dtype=np.int64)
        if raw_events.size != len(GOLFDB_EVENTS) + 2:
            raise DatasetError(
                f"GolfDB 片段 {row.id} 的 events 應有 {len(GOLFDB_EVENTS) + 2} 個數字，"
                f"收到 {raw_events.size}"
            )
        # events[0] 是片段在原影片中的起始影格；videos_160 已裁到這裡
        frames = raw_events[1:-1] - raw_events[0]
        n = sequence.num_frames
        if frames.min() < 0 or frames.max() >= n:
            skip("event_out_of_range")
            continue

        events = dict(zip(GOLFDB_EVENTS, (int(f) for f in frames)))
        pose = to_canonical(sequence.keypoints, LAYOUT)

        try:
            clip = Clip(
                clip_id=f"{DATASET_NAME}/{int(row.id)}",
                sport=SPORT,
                pose=pose,
                fps=sequence.fps,
                events=events,
                label_source="human",
                dataset=DATASET_NAME,
                fold=int(row.split),
            )
            coverage = clip.signals().coverage
        except KineticChainError as exc:
            logger.debug("golfdb/%s 建立片段失敗：%s", row.id, exc)
            skip("clip_failed")
            continue

        if coverage < min_coverage:
            skip("low_coverage")
            continue
        clips.append(clip)

    logger.info("GolfDB: 讀入 %d 段；捨棄 %s", len(clips), dict(sorted(skipped.items())))
    return clips


def split_ids(annotations_path: Path | str, split: int) -> set[int]:
    """GolfDB 官方第 ``split`` 折的片段 id。用於重現作者的訓練／驗證切分。"""
    import pandas as pd

    frame = pd.read_pickle(annotations_path)
    return set(int(i) for i in frame[frame["split"] == split]["id"])
