"""自備影片轉接層。

跟 GolfDB／Penn Action 的差別是**沒有任何標註**：只有影片。因此關鍵點由 RTMPose
估計（不是人工真值），事件由 :mod:`kinetic_chain.weak_labels` 的規則推導（弱標註）。
兩層估計疊在一起，品質下限比兩個公開資料集都低，報表必須據此解讀。

目錄結構沒有規定，只要是同一個運動的影片放在同一個資料夾即可：

    /path/to/videos/*.MOV   →   load(root, sport="clean_and_jerk")

姿態抽取很慢（真實解析度的影片每支數十秒），因此一律先快取成 ``.npz``；
:func:`extract_poses` 負責這件事，:func:`load` 只讀快取。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

from ..data import Clip
from ..errors import KineticChainError
from ..events import get_sport
from ..features import compute
from ..skeleton import to_canonical
from ..weak_labels import derive

logger = logging.getLogger(__name__)

DATASET_NAME = "local_video"

#: RTMPose 輸出的關鍵點布局。
LAYOUT = "coco17"

#: 常見的影片副檔名。iPhone 錄的是 .MOV，Android 多為 .mp4。
VIDEO_SUFFIXES = (".mov", ".mp4", ".m4v", ".avi", ".mkv")

#: 自動裁切時，判定「槓仍在地面」的高度門檻，佔手腕高度全距的比例。
FLOOR_FRACTION = 0.15

#: 自動裁切在動作兩端各留的邊界，單位為秒。
TRIM_MARGIN_SECONDS = 0.4


def auto_trim(signals, fps: float) -> tuple[int, int]:
    """從未裁切的影片中框出單次舉重的區間。

    自備影片不像 GolfDB／Penn Action 已經裁好——實測 12 支影片中，一支 987 格
    （16 秒）的片段裡真正的舉只佔 f430–620，其餘是走位、架槓與收尾。
    弱標註規則假設一段片段恰好包含一次完整動作（`docs/architecture.md` 的 A1），
    未裁切時規則會把事件全部塞進架槓階段。

    作法：以手腕高度的最大值（槓最高）為錨點，往前找最後一次手腕仍貼近地面高度的
    影格當起點。用高度而不是速度：架槓時身體也在動，速度分不出來，但槓在不在地上
    是明確的。
    """
    import numpy as np

    height = np.asarray(signals.signals["wrist_height"], dtype=float)
    if height.size < 4:
        return 0, height.size
    peak = int(height.argmax())
    span = float(height.max() - height.min())
    if span < 1e-9:
        return 0, height.size

    floor = float(height.min()) + FLOOR_FRACTION * span
    grounded = np.flatnonzero(height[: peak + 1] <= floor)
    start = int(grounded[-1]) if grounded.size else 0

    margin = int(round(TRIM_MARGIN_SECONDS * fps))
    return max(0, start - margin), min(height.size, peak + margin + 1)


def iter_videos(root: Path | str) -> list[Path]:
    """列出資料夾下的影片，排序後回傳（順序要穩定，切分才可重現）。"""
    root = Path(root)
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )


def extract_poses(
    root: Path | str,
    cache_dir: Path | str,
    *,
    device: str = "cuda",
    bbox_strategy: str = "detect",
    overwrite: bool = False,
) -> dict[str, int]:
    """對每支影片跑一次姿態抽取並快取。

    ``bbox_strategy`` 預設 ``"detect"``：自備影片不會事先裁切到單一運動員。
    畫面中有其他人時由 :meth:`PoseExtractor._select` 以「最大 + 時間連續」挑出主體。
    """
    from ..pose import PoseExtractor, save_sequence

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    extractor: "PoseExtractor | None" = None
    stats = {"done": 0, "skipped": 0, "failed": 0}

    for video in iter_videos(root):
        target = cache_dir / f"{video.stem}.npz"
        if target.is_file() and not overwrite:
            stats["skipped"] += 1
            continue
        if extractor is None:
            extractor = PoseExtractor(bbox_strategy=bbox_strategy, device=device)
        try:
            save_sequence(extractor.extract_video(video, progress=True), target)
            stats["done"] += 1
        except KineticChainError as exc:
            logger.warning("%s 姿態抽取失敗：%s", video.name, exc)
            stats["failed"] += 1
    return stats


def load(
    root: Path | str,
    cache_dir: Path | str,
    sport: str,
    *,
    min_coverage: float = 0.8,
    only: Sequence[str] | None = None,
    trim: Iterable[tuple[str, int, int]] | None = None,
    auto: bool = False,
) -> list[Clip]:
    """讀取快取的姿態並產生帶弱標註的 :class:`Clip`。

    Parameters
    ----------
    sport:
        這批影片的運動項目 id。自備影片沒有標註，呼叫端必須明說。
    only:
        只保留這些檔名（不含副檔名）。
    trim:
        ``(檔名, 起, 迄)`` 的清單，把片段裁到單一次動作。弱標註規則假設一段片段
        恰好包含一次完整動作（`docs/architecture.md` 的 A1），未裁切的長影片會失敗。
    auto:
        以 :func:`auto_trim` 自動框出單次動作。``trim`` 明確指定者優先。
    """
    from ..pose import load_sequence

    spec = get_sport(sport)
    cache_dir = Path(cache_dir)
    windows = {name: (lo, hi) for name, lo, hi in (trim or ())}
    wanted = set(only) if only else None

    clips: list[Clip] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for video in iter_videos(root):
        if wanted is not None and video.stem not in wanted:
            continue
        path = cache_dir / f"{video.stem}.npz"
        if not path.is_file():
            skip("no_pose_cache")
            continue

        sequence = load_sequence(path)
        pose = to_canonical(sequence.keypoints, LAYOUT)
        if video.stem in windows:
            lo, hi = windows[video.stem]
            pose = pose[lo:hi]

        try:
            signals = compute(
                pose, sequence.fps, handedness_sensitive=spec.handedness_sensitive
            )
            if auto and video.stem not in windows:
                lo, hi = auto_trim(signals, sequence.fps)
                pose = pose[lo:hi]
                signals = compute(
                    pose, sequence.fps, handedness_sensitive=spec.handedness_sensitive
                )
        except KineticChainError as exc:
            logger.debug("%s: 特徵計算失敗 %s", video.stem, exc)
            skip("features_failed")
            continue

        if signals.coverage < min_coverage:
            skip("low_coverage")
            continue

        try:
            events = derive(signals, spec)
        except KineticChainError as exc:
            logger.debug("%s: 弱標註失敗 %s", video.stem, exc)
            skip("weak_label_invalid")
            continue

        clip = Clip(
            clip_id=f"{DATASET_NAME}/{video.stem}",
            sport=sport,
            pose=pose,
            fps=sequence.fps,
            events=events,
            label_source="weak",
            dataset=DATASET_NAME,
            coverage=signals.coverage,
        )
        clip._signals = signals
        clips.append(clip)

    logger.info("自備影片：讀入 %d 段；捨棄 %s", len(clips), dict(sorted(skipped.items())))
    return clips
