"""人工事件標註（CSV）。

這是目前唯一能產生 ``label_source="human"`` 的自備資料路徑。GolfDB 的標註是別人做的、
Penn Action 根本沒有事件標註（只能靠 `weak_labels` 的規則推導），所以在自己的影片上
要有真值，就得走這裡。

格式刻意用**寬表 CSV**：一支影片一列、一個事件一欄，可以直接用試算表編輯。
欄位由標題列決定，不寫死——想少標幾個階段就刪掉那幾欄，載入時自動只產生剩下的事件。

    video,attempt,fps,address,clean_liftoff,clean_catch,clean_overhead,finish,note
    IMG_4604,1,59.94,12,45,88,203,268,
    IMG_4787,1,59.93,429,448,546,928,952,過膝被槓片遮住

規則：

- **影格編號以原始影片為準**（0 起算），不是裁切後的。裁切是衍生的，標註不該綁在上面。
- **空白 ≠ 0**。空白代表「這一格標不出來」（被遮擋、這次動作沒有這個階段），
  載入時該事件會被略過，訓練與評估都會遮罩掉它，不會被當成第 0 格。
- ``attempt`` 給一支影片裡有多次試舉用；目前一支一次就填 1。
- 事件欄名必須是 :mod:`kinetic_chain.events` 註冊過的 id，否則載入時直接報錯，
  不做模糊比對——欄名打錯而被靜默忽略，會變成無聲的資料損失。
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Mapping, Sequence

from ..data import Clip
from ..errors import DatasetError, KineticChainError
from ..events import event_index, get_sport
from ..features import compute
from ..skeleton import to_canonical

logger = logging.getLogger(__name__)

DATASET_NAME = "human_annotation"
LAYOUT = "coco17"

#: 不是事件的欄位。其餘欄位一律當成事件 id。
META_COLUMNS = ("video", "attempt", "fps", "note")


def event_columns(header: Sequence[str]) -> list[str]:
    """從標題列取出事件欄位，並驗證每個都是註冊過的事件 id。"""
    events = [c for c in header if c and c not in META_COLUMNS]
    if not events:
        raise DatasetError(
            f"標題列沒有任何事件欄位；非事件欄位為 {META_COLUMNS}，標題列是 {list(header)}"
        )
    for name in events:
        event_index(name)      # 未註冊的 id 直接拋 UnknownEventError
    return events


def write_template(
    path: Path | str,
    rows: Sequence[Mapping[str, object]],
    events: Sequence[str],
) -> Path:
    """寫出（可預先填好的）標註範本。

    預先填入模型的預測值，讓標註者**修正**而不是從零建立——實測模型在頭尾事件上
    已有 0.75–0.875 的準確度，那些格多半不用動。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["video", "attempt", "fps", *events, "note"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})
    return path


def read(path: Path | str) -> tuple[list[str], list[dict[str, str]]]:
    """讀回 CSV，回傳 ``(事件欄位, 資料列)``。"""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"找不到標註檔：{path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetError(f"標註檔沒有標題列：{path}")
        return event_columns(reader.fieldnames), list(reader)


def load(
    annotations: Path | str,
    cache_dir: Path | str,
    sport: str,
    *,
    min_coverage: float = 0.8,
) -> list[Clip]:
    """讀取人工標註 + 姿態快取，產生 ``label_source="human"`` 的 :class:`Clip`。

    姿態仍由 RTMPose 估計（`local_video.extract_poses` 產生的 ``.npz``）；
    人工提供的只有事件時間點。
    """
    from ..pose import load_sequence

    spec = get_sport(sport)
    cache_dir = Path(cache_dir)
    events, rows = read(annotations)

    unknown = [e for e in events if e not in spec.events]
    if unknown:
        raise DatasetError(
            f"標註檔有 {sport!r} 沒有宣告的事件欄位 {unknown}；"
            f"該運動宣告的是 {list(spec.events)}"
        )

    clips: list[Clip] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for row in rows:
        name = (row.get("video") or "").strip()
        if not name:
            continue
        path = cache_dir / f"{name}.npz"
        if not path.is_file():
            skip("no_pose_cache")
            continue

        marks: dict[str, int] = {}
        for e in events:
            raw = (row.get(e) or "").strip()
            if not raw:            # 空白 = 沒標，不是第 0 格
                continue
            try:
                marks[e] = int(float(raw))
            except ValueError as exc:
                raise DatasetError(
                    f"{name} 的 {e!r} 欄位不是整數影格編號：{raw!r}"
                ) from exc
        if not marks:
            skip("no_events_marked")
            continue

        sequence = load_sequence(path)
        pose = to_canonical(sequence.keypoints, LAYOUT)
        n = pose.shape[0]
        bad = {e: f for e, f in marks.items() if not 0 <= f < n}
        if bad:
            raise DatasetError(
                f"{name} 的影格編號超出影片長度 {n}：{bad}。"
                "編號應以原始影片為準（0 起算）"
            )

        ordered = [e for e in spec.events if e in marks]
        frames = [marks[e] for e in ordered]
        if any(b < a for a, b in zip(frames, frames[1:])):
            raise DatasetError(
                f"{name} 的標註違反 {sport!r} 宣告的事件時序："
                + ", ".join(f"{e}={marks[e]}" for e in ordered)
            )

        try:
            signals = compute(
                pose, sequence.fps, handedness_sensitive=spec.handedness_sensitive
            )
        except KineticChainError as exc:
            logger.debug("%s: 特徵計算失敗 %s", name, exc)
            skip("features_failed")
            continue
        if signals.coverage < min_coverage:
            skip("low_coverage")
            continue

        attempt = (row.get("attempt") or "1").strip() or "1"
        clip = Clip(
            clip_id=f"{DATASET_NAME}/{name}#{attempt}",
            sport=sport,
            pose=pose,
            fps=sequence.fps,
            events=marks,
            label_source="human",
            dataset=DATASET_NAME,
            coverage=signals.coverage,
        )
        clip._signals = signals
        clips.append(clip)

    logger.info(
        "人工標註：讀入 %d 段（%d 個事件欄位）；捨棄 %s",
        len(clips), len(events), dict(sorted(skipped.items())),
    )
    return clips
