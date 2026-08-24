"""把各運動的偵測誤差放在同一把尺上比較，回答「效果到底差在哪」。

PCE 不能跨運動比較。它的容忍度是 `(準備→觸球) / 30`，所以片段越短容忍度越嚴：
高爾夫容忍 6 格，打擊只容忍 1 格。同樣差 2 格，在高爾夫算命中，在打擊算沒中。
拿 PCE 說「打擊比高爾夫差」是拿兩把不同的尺量。

本腳本改報**不受容忍度影響**的量：誤差影格數與毫秒。並且把同一個模型在不同
容忍度下的 PCE 一起列出，藉此把「模型不準」與「標準太嚴」分開。

    python scripts/error_budget.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from kinetic_chain.data import Clip, split_clips
from kinetic_chain.evaluate import predict_clips
from kinetic_chain.events import get_sport
from kinetic_chain.metrics import tolerance
from kinetic_chain.train import load_checkpoint

logger = logging.getLogger("error_budget")

#: 運動 → (權重, 資料來源)。GolfDB 是唯一有真人標註的。
RUNS: tuple[tuple[str, str, str], ...] = (
    ("golf_swing", "runs/golf/model.pt", "human"),
    ("baseball_pitch", "runs/pitch/model.pt", "weak"),
    ("baseball_swing", "runs/bat/model.pt", "weak"),
    ("clean_and_jerk", "runs/lift/model.pt", "weak"),
)

#: 額外檢查的固定容忍度（影格）。用固定值而非相對值，跨運動才可比。
FIXED_TOLERANCES = (1, 2, 3, 6)


def measure(clips: list[Clip], predictions: list[dict[str, int]], sport: str) -> dict:
    spec = get_sport(sport)
    deltas: list[int] = []
    per_event: dict[str, list[int]] = {}
    tolerances: list[int] = []
    fps_values: list[float] = []
    spans: list[int] = []

    for clip, prediction in zip(clips, predictions):
        tolerances.append(tolerance(clip.events, order=spec.events))
        fps_values.append(clip.fps)
        present = [clip.events[e] for e in spec.events if e in clip.events]
        spans.append(max(present) - min(present))
        for event in spec.events:
            if event not in prediction or event not in clip.events:
                continue
            delta = abs(prediction[event] - clip.events[event])
            deltas.append(delta)
            per_event.setdefault(event, []).append(delta)

    array = np.asarray(deltas, dtype=float)
    fps = float(np.median(fps_values))
    return {
        "sport": sport,
        "label_source": {c.label_source for c in clips}.pop(),
        "val_clips": len(clips),
        "events_scored": len(deltas),
        "fps": fps,
        "action_span_frames_median": float(np.median(spans)),
        "tolerance_frames_mean": float(np.mean(tolerances)),
        "abs_error_frames": {
            "median": float(np.median(array)),
            "mean": round(float(array.mean()), 2),
            "p75": float(np.percentile(array, 75)),
            "p90": float(np.percentile(array, 90)),
        },
        # 毫秒才是真正跨運動可比的單位：影格數要除以該資料集的 fps
        "abs_error_ms": {
            "median": round(1000 * float(np.median(array)) / fps, 1),
            "mean": round(1000 * float(array.mean()) / fps, 1),
            "p90": round(1000 * float(np.percentile(array, 90)) / fps, 1),
        },
        "pce_at_fixed_tolerance": {
            str(t): round(float((array <= t).mean()), 3) for t in FIXED_TOLERANCES
        },
        "pce_at_own_tolerance": round(
            float(
                np.mean(
                    [
                        abs(prediction[e] - clip.events[e])
                        <= tolerance(clip.events, order=spec.events)
                        for clip, prediction in zip(clips, predictions)
                        for e in spec.events
                        if e in prediction and e in clip.events
                    ]
                )
            ),
            3,
        ),
        "worst_events": sorted(
            (
                (event, round(float(np.median(values)), 1))
                for event, values in per_event.items()
            ),
            key=lambda item: -item[1],
        )[:3],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pennaction-root", type=Path, default=Path("data/raw/Penn_Action"))
    parser.add_argument("--golfdb", type=Path, default=Path("data/raw/golfDB.pkl"))
    parser.add_argument("--golfdb-cache", type=Path, default=Path("data/cache/golfdb_pose"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/error_budget.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from kinetic_chain.datasets import golfdb, pennaction

    rows = []
    for sport, checkpoint, expected_source in RUNS:
        path = Path(checkpoint)
        if not path.exists():
            logger.warning("缺少權重 %s，略過 %s", path, sport)
            continue
        if sport == "golf_swing":
            clips = golfdb.load(args.golfdb, args.golfdb_cache)
            _, val_clips = split_clips(clips, val_fold=1)
        else:
            clips = pennaction.load(args.pennaction_root, sports=[sport])
            _, val_clips = split_clips(clips, val_fraction=0.2, seed=0)

        sources = {c.label_source for c in val_clips}
        if sources != {expected_source}:
            raise SystemExit(f"{sport} 的標註來源為 {sources}，預期 {expected_source}")

        model = load_checkpoint(path, device=args.device)
        predictions = predict_clips(model, val_clips, device=args.device)
        rows.append(measure(val_clips, predictions, sport))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"runs": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(
        "%-16s %-6s %5s %8s %9s %9s %8s",
        "運動", "標註", "容忍", "誤差(格)", "誤差(ms)", "自身PCE", "PCE@2格",
    )
    for row in rows:
        logger.info(
            "%-16s %-6s %5.1f %8.1f %9.0f %9.3f %8.3f",
            row["sport"], row["label_source"], row["tolerance_frames_mean"],
            row["abs_error_frames"]["median"], row["abs_error_ms"]["median"],
            row["pce_at_own_tolerance"], row["pce_at_fixed_tolerance"]["2"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
