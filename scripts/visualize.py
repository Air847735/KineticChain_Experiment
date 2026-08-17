"""把偵測到的關鍵時間點畫成一張圖，用眼睛檢查結果。

分數告訴你平均誤差多少，但看不出模型錯在哪裡、錯得合不合理。這支腳本把每個
事件對應的那一格截出來排成一列，預測與真值上下對照，一眼就能看出是「差一格」
還是「整個抓錯位置」。

    python scripts/visualize.py --checkpoint runs/golf/model.pt --val-fold 1 --count 4
    python scripts/visualize.py --checkpoint runs/pitch/model.pt \\
        --sport baseball_pitch --count 3

輸出含可辨識的運動員影像，**不進版控**（`runs/` 已在 .gitignore）。
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


def read_frames(source: Path, indices: list[int]) -> dict[int, np.ndarray]:
    """取出指定影格。

    ``source`` 可以是影片檔（GolfDB 的 mp4）或影格目錄（Penn Action 的 jpg 序列）。
    影片一律逐格讀而不 seek——短片段的關鍵格常落在關鍵影格之間，seek 會跳到最近的
    關鍵影格而悄悄取錯格。
    """
    import cv2

    if source.is_dir():
        files = sorted(source.glob("*.jpg"))
        frames = {}
        for index in indices:
            if 0 <= index < len(files):
                image = cv2.imread(str(files[index]))
                if image is not None:
                    frames[index] = image
        return frames

    wanted = set(indices)
    capture = cv2.VideoCapture(str(source))
    frames = {}
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
    reference_label: str = "true",
) -> np.ndarray:
    """一列預測、一列參照，欄位對齊同一個事件。

    ``reference_label`` 讓呼叫端標明第二列是什麼：GolfDB 是人工標註（``true``），
    Penn Action 是規則推導的弱標註（``rule``）。兩者不是同一種東西，圖上必須分得出來。
    """
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
            sheet, "pred" if row == 0 else reference_label, (6, y + TILE // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA,
        )
        for column, event in enumerate(events):
            x = ROW_LABEL_WIDTH + column * TILE
            frame = frames.get(source[event])
            if frame is None:
                sheet[y : y + TILE, x : x + TILE] = blank
            else:
                sheet[y : y + TILE, x : x + TILE] = cv2.resize(frame, (TILE, TILE))

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
    parser.add_argument("--pennaction-root", type=Path, default=Path("data/raw/Penn_Action"))
    parser.add_argument("--sport", default="golf_swing")
    parser.add_argument("--val-fold", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/visualise"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.sport == "golf_swing":
        from kinetic_chain.datasets import golfdb

        clips = golfdb.load(args.golfdb_annotations, args.golfdb_cache)
        _, val_clips = split_clips(clips, val_fold=args.val_fold)
        media_for = lambda cid: args.golfdb_videos / f"{cid}.mp4"  # noqa: E731
        reference_label = "true"
    else:
        from kinetic_chain.datasets import pennaction

        clips = pennaction.load(args.pennaction_root, sports=[args.sport])
        _, val_clips = split_clips(clips, val_fraction=args.val_fraction, seed=0)
        media_for = lambda cid: args.pennaction_root / "frames" / cid  # noqa: E731
        # Penn Action 的第二列是規則推導的弱標註，不是人工標註，標籤必須不同
        reference_label = "rule"

    if not val_clips:
        raise SystemExit(f"{args.sport} 沒有驗證片段")

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

    # 先取最好、最差、中位，其餘依序遞補——並非每個片段都有影像來源
    # （Penn Action 的 frames/ 通常只解壓一部分），缺的要能跳過而不是就此空手。
    n = len(scored)
    priority = [0, n - 1, n // 2]
    order = priority + [i for i in range(n) if i not in priority]

    def label_for(index: int) -> str:
        return "best" if index == 0 else "worst" if index == n - 1 else "median"

    picks = [(label_for(i), *scored[i]) for i in order]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for label, score, clip, prediction, tol in picks:
        if written >= args.count:
            break
        clip_number = clip.clip_id.split("/")[-1]
        media = media_for(clip_number)
        if not media.exists():
            logger.warning("找不到影像來源 %s，略過", media)
            continue
        sheet = build_sheet(
            media, list(clip.ordered_events), prediction, clip.events, tol,
            reference_label=reference_label,
        )
        written += 1
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
