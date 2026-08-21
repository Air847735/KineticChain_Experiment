"""棒球打者：逐段的資料與偵測結果。

`scripts/run_experiments.py` 之類的腳本只輸出彙總數字；要看「每一段影片實際偵測到
什麼」得有逐段紀錄。本腳本把驗證集的每一段展開成一列：弱標註影格、模型偵測影格、
兩者差、是否落在容忍度內，另外附上分期佔比。

輸出只含 Penn Action 的片段 id 與影格索引，**不含任何影像**。

    python scripts/bat_report.py --checkpoint runs/bat/model.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from kinetic_chain.analysis import analyse
from kinetic_chain.data import split_clips
from kinetic_chain.evaluate import evaluate_clips, predict_clips
from kinetic_chain.events import get_sport
from kinetic_chain.metrics import tolerance
from kinetic_chain.train import load_checkpoint

logger = logging.getLogger("bat")
SPORT = "baseball_swing"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/bat/model.pt"))
    parser.add_argument("--pennaction-root", type=Path, default=Path("data/raw/Penn_Action"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("runs/bat_report.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from kinetic_chain.datasets import pennaction

    clips = pennaction.load(args.pennaction_root, sports=[SPORT])
    spec = get_sport(SPORT)
    model = load_checkpoint(args.checkpoint, device=args.device)
    train_clips, val_clips = split_clips(
        clips, val_fraction=args.val_fraction, seed=args.seed
    )
    predictions = predict_clips(model, val_clips, device=args.device)
    reports = evaluate_clips(model, val_clips, device=args.device)

    # 每個運動的標註來源必須分開報，這裡驗證集全部是弱標註。
    sources = sorted({c.label_source for c in clips})
    if sources != ["weak"]:
        raise SystemExit(f"預期全部為弱標註，實際為 {sources}")

    records = []
    for clip, pred in zip(val_clips, predictions):
        tol = tolerance(clip.events, order=spec.events)
        chain = analyse(clip.clip_id, spec, pred, clip.fps)
        events = []
        for name in spec.events:
            if name not in clip.events:
                continue
            truth = int(clip.events[name])
            got = int(pred[name])
            events.append(
                {
                    "event": name,
                    "truth": truth,
                    "pred": got,
                    "delta": got - truth,
                    "hit": abs(got - truth) <= tol,
                }
            )
        records.append(
            {
                "clip_id": clip.clip_id,
                "frames": int(clip.num_frames),
                "fps": float(clip.fps),
                "coverage": float(clip.coverage),
                "label_source": clip.label_source,
                "tolerance": tol,
                "hits": sum(e["hit"] for e in events),
                "num_events": len(events),
                "max_abs_delta": max(abs(e["delta"]) for e in events) if events else 0,
                "events": events,
                "segments": [
                    {
                        "name": s.name,
                        "frames": s.frames,
                        "seconds": round(s.seconds, 3),
                        "percent": round(s.percent_of_throw, 1),
                    }
                    for s in chain.segments
                ],
                # 解碼器強制順序，所以這一欄恆為 True。它證明的是解碼器沒有出錯，
                # **不是**近端到遠端序列在資料上成立——後者要看
                # scripts/rotation_chain_by_sport.py 的無約束量測。
                "sequence_from_decoder": " → ".join(
                    e.replace("_peak_rotation", "").replace("_peak_velocity", "")
                    for e in chain.sequence
                ),
                "sequence_order_violations": 0 if chain.sequence_ok else 1,
            }
        )
    records.sort(key=lambda r: (-r["hits"] / max(r["num_events"], 1), r["max_abs_delta"]))

    key = f"{SPORT}/weak"
    lengths = np.array([c.num_frames for c in clips], dtype=float)
    fps_values = sorted({float(c.fps) for c in clips})
    result = {
        "sport": SPORT,
        "label_source": "weak",
        "note": (
            "每一列的 sequence_* 來自解碼器輸出，順序是被強制的，不能當成序列成立的證據；"
            "無約束的量測見 scripts/rotation_chain_by_sport.py。"
        ),
        "dataset": {
            "name": "Penn Action",
            "clips_loaded": len(clips),
            "train": len(train_clips),
            "val": len(val_clips),
            "fps": fps_values,
            "frames_min": int(lengths.min()),
            "frames_median": float(np.median(lengths)),
            "frames_max": int(lengths.max()),
            "coverage_mean": float(np.mean([c.coverage for c in clips])),
            "events_declared": list(spec.events),
        },
        "detection": {
            "pce_vs_weak_labels": reports[key].pce,
            "mean_tolerance_frames": reports[key].mean_tolerance,
            "per_event": {
                name: {"pce": s.pce, "median_delta": s.median_delta}
                for name, s in reports[key].per_event.items()
            },
        },
        "clips": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "驗證集 %d 段，PCE %.4f，寫入 %s", len(records), reports[key].pce, args.output
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
