"""產生（預先填好的）人工標註範本，並輸出每一格的縮圖供核對。

標註不從零開始：先用現有模型把每個事件的預測填進 CSV，標註者只要**改錯的**。
實測模型在頭尾事件上已有 0.75–0.875 的準確度，那幾欄多半不用動；
真正要修的是中段。

    python scripts/make_annotation_template.py \
        --videos /srv/datasets/weight --cache data/cache/own_weight \
        --sport clean_and_jerk --checkpoint runs/lift/model.pt \
        --output annotations/weight.csv

輸出兩樣東西：

``annotations/weight.csv``
    寬表 CSV，一支影片一列。影格編號以**原始影片**為準。
``annotations/preview/<video>.jpg``
    每支影片一張縮圖列，把預測的那幾格排出來，改的時候對照著看。
    **含可辨識個人，不進版控。**
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from kinetic_chain.datasets.annotations import write_template
from kinetic_chain.events import get_sport

logger = logging.getLogger("template")


def preview(video: Path, frames: dict[str, int], path: Path, tile: int = 190) -> None:
    """把預測的那幾格排成一列存成圖。"""
    import cv2

    wanted = sorted(set(frames.values()))
    capture = cv2.VideoCapture(str(video))
    grabbed: dict[int, np.ndarray] = {}
    pending = set(wanted)
    position = 0
    while pending:
        ok, frame = capture.read()
        if not ok:
            break
        if position in pending:
            grabbed[position] = frame
            pending.discard(position)
        position += 1
    capture.release()

    names = list(frames)
    width = int(tile * 0.58)
    sheet = np.full((tile + 40, width * len(names), 3), 24, dtype=np.uint8)
    for i, name in enumerate(names):
        frame = grabbed.get(frames[name])
        x = i * width
        if frame is not None:
            h, w = frame.shape[:2]
            scale = tile / h
            small = cv2.resize(frame, (int(w * scale), tile))
            take = min(small.shape[1], width)
            sheet[:tile, x : x + take] = small[:, :take]
        label = name.replace("clean_", "").replace("_peak_velocity", "_drive")[:13]
        cv2.putText(sheet, label, (x + 3, tile + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"f{frames[name]}", (x + 3, tile + 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 220, 160), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sport", required=True)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="有給就用模型預測預先填入；沒給則輸出空白範本")
    parser.add_argument("--events", nargs="*", default=None,
                        help="只標這幾個事件（想減少階段數時用）。預設為該運動宣告的全部")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("annotations/template.csv"))
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from kinetic_chain.datasets import local_video

    spec = get_sport(args.sport)
    events = list(args.events) if args.events else list(spec.events)
    unknown = [e for e in events if e not in spec.events]
    if unknown:
        raise SystemExit(f"{args.sport!r} 沒有宣告這些事件：{unknown}")

    videos = local_video.iter_videos(args.videos)
    if not videos:
        raise SystemExit(f"{args.videos} 底下沒有影片")

    predictions: dict[str, dict[str, int]] = {}
    fps_of: dict[str, float] = {}
    if args.checkpoint is not None:
        from kinetic_chain.evaluate import predict_clips
        from kinetic_chain.pose import load_sequence
        from kinetic_chain.train import load_checkpoint

        clips = local_video.load(args.videos, args.cache, args.sport, auto=True)
        model = load_checkpoint(args.checkpoint, device=args.device)
        for clip, guess in zip(clips, predict_clips(model, clips, device=args.device)):
            name = clip.clip_id.split("/")[-1]
            sequence = load_sequence(args.cache / f"{name}.npz")
            # 預測是在裁切後的片段上做的，要換算回原始影片的影格編號
            from kinetic_chain.features import compute
            from kinetic_chain.skeleton import to_canonical

            signals = compute(
                to_canonical(sequence.keypoints, local_video.LAYOUT),
                sequence.fps,
                handedness_sensitive=spec.handedness_sensitive,
            )
            offset, _ = local_video.auto_trim(signals, sequence.fps)
            predictions[name] = {e: offset + f for e, f in guess.items() if e in events}
            fps_of[name] = sequence.fps
        logger.info("以 %s 預先填入 %d 支影片的預測", args.checkpoint, len(predictions))

    rows = []
    for video in videos:
        name = video.stem
        cached = args.cache / f"{name}.npz"
        if not cached.is_file():
            logger.warning("%s 沒有姿態快取，先跑 local_video.extract_poses", name)
            continue
        if name not in fps_of:
            from kinetic_chain.pose import load_sequence

            fps_of[name] = load_sequence(cached).fps
        guess = predictions.get(name, {})
        rows.append({
            "video": name,
            "attempt": 1,
            "fps": round(fps_of[name], 2),
            "note": "" if guess else "模型未預測，需完整標註",
            **{e: guess.get(e, "") for e in events},
        })
        if guess and not args.no_preview:
            preview(video, {e: guess[e] for e in events if e in guess},
                    args.output.parent / "preview" / f"{name}.jpg")

    write_template(args.output, rows, events)
    logger.info("→ %s（%d 列 × %d 個事件欄位）", args.output, len(rows), len(events))
    if not args.no_preview and predictions:
        logger.info("→ %s（核對用縮圖，含可辨識個人，不進版控）",
                    args.output.parent / "preview")
    print(f"\n標註方式：開啟 {args.output}，逐列核對影格編號。")
    print("  · 編號以原始影片為準（0 起算）")
    print("  · 標不出來的格子留空白，不要填 0——空白會被遮罩掉，0 會被當成第 0 格")
    print("  · 想減少階段數就刪掉那幾欄")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
