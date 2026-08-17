"""棒球投球的動力鏈時序分析。

跑三件事，答案彼此獨立：

1. **模型能不能重現弱標註** —— 在 Penn Action 投球片段上的 PCE。這只說明模型學會了
   規則，不說明規則正確。
2. **動力鏈時序指標** —— 由偵測到的事件算出各段的分離時間與佔比。
3. **近端到遠端序列到底成不成立** —— 直接量原始訊號的峰值，**不套任何順序假設**。
   這是唯一沒有循環論證的問法：弱標註為了讓標註可用，把骨盆／軀幹的搜尋範圍限制在
   上肢峰值之前，那條路徑必然得出正確順序。

    python scripts/pitch_analysis.py --checkpoint runs/pitch/model.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from kinetic_chain.analysis import (
    CHAIN_LINKS,
    analyse,
    format_report,
    sequence_rate,
    summarise,
    unconstrained_sequence,
)
from kinetic_chain.data import split_clips
from kinetic_chain.evaluate import evaluate_clips, predict_clips
from kinetic_chain.events import get_sport
from kinetic_chain.train import load_checkpoint

logger = logging.getLogger("pitch")
SPORT = "baseball_pitch"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/pitch/model.pt"))
    parser.add_argument("--pennaction-root", type=Path, default=Path("data/raw/Penn_Action"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/pitch_analysis.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from kinetic_chain.datasets import pennaction

    clips = pennaction.load(args.pennaction_root, sports=[SPORT])
    spec = get_sport(SPORT)
    model = load_checkpoint(args.checkpoint, device=args.device)
    _, val_clips = split_clips(clips, val_fraction=0.2, seed=args.seed)

    results: dict = {"sport": SPORT, "clips": len(clips), "val_clips": len(val_clips)}

    # ---------------------------------------------------------------- 1
    reports = evaluate_clips(model, val_clips, device=args.device)
    key = f"{SPORT}/weak"
    results["detection"] = {
        "pce_vs_weak_labels": reports[key].pce,
        "mean_tolerance_frames": reports[key].mean_tolerance,
        "per_event": {
            name: {"pce": s.pce, "median_delta": s.median_delta}
            for name, s in reports[key].per_event.items()
        },
    }
    logger.info("模型 vs 弱標註 PCE %.4f（%d 段）", reports[key].pce, reports[key].num_clips)

    # ---------------------------------------------------------------- 2
    predictions = predict_clips(model, val_clips, device=args.device)
    detected = [
        analyse(c.clip_id, spec, p, c.fps) for c, p in zip(val_clips, predictions)
    ]
    reference = [analyse(c.clip_id, spec, c.events, c.fps) for c in clips]

    results["timing_detected"] = summarise(detected)
    results["timing_reference"] = summarise(reference)
    logger.info("時序指標：%d 段（模型偵測）/ %d 段（弱標註）", len(detected), len(reference))

    # ---------------------------------------------------------------- 3
    expected = tuple(name for name, _ in CHAIN_LINKS)
    unconstrained = {"whole_clip": [], "acceleration_phase": []}
    for clip in clips:
        signals = clip.signals()
        whole = unconstrained_sequence(signals)
        # 加速階段：前腳著地 → 出手。用弱標註的這兩個邊界，但兩者都不是由順序
        # 約束推出來的（著地看踝高度、出手看腕速減速），所以不構成循環。
        lo = clip.events.get("stride_foot_contact", 0)
        hi = clip.events.get("release_impact", signals.pose.shape[0] - 1) + 1
        window = unconstrained_sequence(signals, window=(lo, hi))
        for name, peaks in (("whole_clip", whole), ("acceleration_phase", window)):
            observed = tuple(sorted(peaks, key=lambda e: peaks[e]))
            unconstrained[name].append(
                {
                    "clip_id": clip.clip_id,
                    "peaks": peaks,
                    "order_ok": observed == expected,
                    "order": " → ".join(
                        e.replace("_peak_rotation", "").replace("_peak_velocity", "")
                        for e in observed
                    ),
                }
            )

    results["unconstrained"] = {}
    for name, records in unconstrained.items():
        counts: dict[str, int] = {}
        for r in records:
            counts[r["order"]] = counts.get(r["order"], 0) + 1
        results["unconstrained"][name] = {
            "n": len(records),
            "proximal_to_distal_rate": float(np.mean([r["order_ok"] for r in records])),
            "orders": {k: v / len(records) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])},
        }
        logger.info(
            "無約束量測（%s）：近端到遠端成立 %.1f%%",
            name,
            results["unconstrained"][name]["proximal_to_distal_rate"] * 100,
        )

    results["sequence_rate_detected"] = sequence_rate(detected)

    print()
    print(format_report(results["timing_detected"], results["sequence_rate_detected"],
                        title="投球動力鏈時序（模型偵測，驗證集）"))
    print()
    print(format_report(results["timing_reference"], sequence_rate(reference),
                        title="投球動力鏈時序（弱標註，全部片段）"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已寫入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
