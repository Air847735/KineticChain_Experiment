"""把偵測到的關鍵時間點畫成一張圖，用眼睛檢查結果。

分數告訴你平均誤差多少，但看不出模型錯在哪裡、錯得合不合理。這支腳本把每個
事件對應的那一格截出來排成一列，預測與真值上下對照，一眼就能看出是「差一格」
還是「整個抓錯位置」。

    python scripts/visualize.py --checkpoint runs/golf/model.pt --val-fold 1 --count 4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from kinetic_chain.data import split_clips
from kinetic_chain.evaluate import predict_clips
from kinetic_chain.metrics import tolerance
from kinetic_chain.train import load_checkpoint

logger = logging.getLogger("visualize")

TILE = 160
LABEL_HEIGHT = 34
ROW_LABEL_WIDTH = 52

# BGR
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
HIT = (120, 220, 120)
MISS = (110, 110, 245)
BACKGROUND = (24, 24, 24)


def read_frames(video: Path, indices: list[int]) -> dict[int, np.ndarray]:
    """只取需要的影格。逐格讀比 seek 可靠——短片段的關鍵格常落在關鍵影格之間。"""
    import cv2

    wanted = set(indices)
    capture = cv2.VideoCapture(str(video))
    frames: dict[int, np.ndarray] = {}
    position = 0
    try:
        while wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if position in wanted:
                frames[position] = frame
                wanted.discard(position)
            position += 1
    finally:
        capture.release()
    return frames


def _short(event: str) -> str:
    """把事件 id 縮成放得進 160 px 的標籤。"""
    return (
        event.replace("golf_", "")
        .replace("_rotation", "")
        .replace("_velocity", "")
        .replace("release_impact", "impact")
        .replace("follow_through_mid", "follow-thru")
        .replace("stride_foot_contact", "foot-contact")
        .replace("pelvis_peak", "pelvis")
        .replace("torso_peak", "torso")
        .replace("arm_peak", "arm")
    )


def build_sheet(
    video: Path,
    events: list[str],
    predicted: dict[str, int],
    truth: dict[str, int] | None,
    tol: int,
) -> np.ndarray:
    """一列預測、（有真值時）一列真值，欄位對齊同一個事件。"""
    import cv2

    rows = 2 if truth else 1
    indices = list(predicted.values()) + (list(truth.values()) if truth else [])
    frames = read_frames(video, indices)
    blank = np.full((TILE, TILE, 3), 60, dtype=np.uint8)

    width = ROW_LABEL_WIDTH + TILE * len(events)
    height = LABEL_HEIGHT + rows * (TILE + LABEL_HEIGHT)
    sheet = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)

    for column, event in enumerate(events):
        x = ROW_LABEL_WIDTH + column * TILE
        cv2.putText(
            sheet, _short(event)[:19], (x + 4, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA,
        )

    for row in range(rows):
        source = predicted if row == 0 else truth
        assert source is not None
        y = LABEL_HEIGHT + row * (TILE + LABEL_HEIGHT)
        cv2.putText(
            sheet, "pred" if row == 0 else "true", (6, y + TILE // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA,
        )
        for column, event in enumerate(events):
            x = ROW_LABEL_WIDTH + column * TILE
            frame = frames.get(source[event])
            sheet[y : y + TILE, x : x + TILE] = blank if frame is None else frame

            caption = f"f{source[event]}"
            colour = WHITE
            if row == 0 and truth is not None:
                delta = abs(source[event] - truth[event])
                colour = HIT if delta <= tol else MISS
                caption = f"f{source[event]}  d={delta}"
            cv2.putText(
                sheet, caption, (x + 4, y + TILE + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA,
            )
    return sheet


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--golfdb-annotations", type=Path, default=Path("data/raw/golfDB.pkl"))
    parser.add_argument("--golfdb-cache", type=Path, default=Path("data/cache/golfdb_pose"))
    parser.add_argument("--golfdb-videos", type=Path, default=Path("data/raw/videos_160"))
    parser.add_argument("--val-fold", type=int, default=1)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/visualise"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from kinetic_chain.datasets import golfdb

    clips = golfdb.load(args.golfdb_annotations, args.golfdb_cache)
    _, val_clips = split_clips(clips, val_fold=args.val_fold)
    if not val_clips:
        raise SystemExit(f"第 {args.val_fold} 折沒有驗證片段")

    model = load_checkpoint(args.checkpoint, device=args.device)
    predictions = predict_clips(model, val_clips, device=args.device)

    # 挑最好、中位、最差各一批，避免只看漂亮的例子
    scored = []
    for clip, prediction in zip(val_clips, predictions):
        tol = tolerance(clip.events, order=clip.ordered_events)
        hits = sum(
            abs(prediction[e] - clip.events[e]) <= tol for e in clip.ordered_events
        )
        scored.append((hits / len(clip.ordered_events), clip, prediction, tol))
    scored.sort(key=lambda row: row[0], reverse=True)

    picks = []
    n = len(scored)
    for label, index in (
        ("best", 0),
        ("median", n // 2),
        ("worst", n - 1),
    ):
        picks.append((label, *scored[index]))
    for offset in range(1, max(args.count - 3, 0) + 1):
        picks.append(("median", *scored[min(n // 2 + offset, n - 1)]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, score, clip, prediction, tol in picks[: args.count]:
        clip_number = clip.clip_id.split("/")[-1]
        video = args.golfdb_videos / f"{clip_number}.mp4"
        if not video.is_file():
            logger.warning("找不到影片 %s", video)
            continue
        sheet = build_sheet(
            video, list(clip.ordered_events), prediction, clip.events, tol
        )
        path = args.output_dir / f"{label}_{clip_number}.png"
        cv2.imwrite(str(path), sheet)
        logger.info(
            "%-6s clip %-5s  命中 %d/%d（容忍 %d 格）→ %s",
            label,
            clip_number,
            round(score * len(clip.ordered_events)),
            len(clip.ordered_events),
            tol,
            path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
