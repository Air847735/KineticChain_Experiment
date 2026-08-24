"""誤差是否隨**弱標註規則的種類**而不同？

`scripts/failure_modes.py` 顯示每個運動最容易大錯的事件都是同一類：門檻型
（`rest_start`、`foot_contact`、`signal_onset`）與姿勢極值型（`signal_extreme`），
而速度峰值型（`signal_peak`）幾乎不出錯。本腳本把事件依規則種類分組驗證這件事。

若成立，改善的方向就不是換模型，而是換這些事件的定義。

    python scripts/rule_type_errors.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from kinetic_chain.data import split_clips
from kinetic_chain.evaluate import predict_clips
from kinetic_chain.events import get_sport
from kinetic_chain.train import load_checkpoint

logger = logging.getLogger("rule_type")

RUNS: tuple[tuple[str, str], ...] = (
    ("golf_swing", "runs/golf/model.pt"),
    ("baseball_pitch", "runs/pitch/model.pt"),
    ("baseball_swing", "runs/bat/model.pt"),
    ("clean_and_jerk", "runs/lift/model.pt"),
)

#: 規則 → 分組。分組依據是「這個時間點由什麼決定」。
RULE_GROUP: dict[str, str] = {
    "signal_peak": "速度峰值",
    "post_peak_decel": "速度峰值",
    "signal_extreme": "姿勢極值",
    "signal_crossing": "門檻",
    "signal_onset": "門檻",
    "foot_contact": "門檻",
    "rest_start": "邊界",
    "rest_end": "邊界",
    "midpoint": "推算",
}

BAD_FRAMES = 5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pennaction-root", type=Path, default=Path("data/raw/Penn_Action"))
    parser.add_argument("--golfdb", type=Path, default=Path("data/raw/golfDB.pkl"))
    parser.add_argument("--golfdb-cache", type=Path, default=Path("data/cache/golfdb_pose"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/rule_type_errors.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from kinetic_chain.datasets import golfdb, pennaction

    # GolfDB 是真人標註，事件沒有「規則」；仍照同一套分組是為了看
    # 同一類事件在真人標註下是否也難——難的是事件本身，還是規則？
    by_group: dict[str, dict[str, list[int]]] = {}

    for sport, checkpoint in RUNS:
        path = Path(checkpoint)
        if not path.exists():
            logger.warning("缺少權重 %s，略過", path)
            continue
        if sport == "golf_swing":
            clips = golfdb.load(args.golfdb, args.golfdb_cache)
            _, val_clips = split_clips(clips, val_fold=1)
        else:
            clips = pennaction.load(args.pennaction_root, sports=[sport])
            _, val_clips = split_clips(clips, val_fraction=0.2, seed=0)

        spec = get_sport(sport)
        rule_of = {rule.event: rule.rule for rule in spec.weak_rules}
        model = load_checkpoint(path, device=args.device)
        predictions = predict_clips(model, val_clips, device=args.device)

        source = "human" if sport == "golf_swing" else "weak"
        for clip, prediction in zip(val_clips, predictions):
            for event in spec.events:
                if event not in prediction or event not in clip.events:
                    continue
                kind = rule_of.get(event)
                group = RULE_GROUP.get(kind, "其他") if kind else "運動專屬"
                delta = abs(prediction[event] - clip.events[event])
                by_group.setdefault(group, {}).setdefault(source, []).append(delta)

    summary = {}
    for group, sources in sorted(by_group.items()):
        summary[group] = {}
        for source, deltas in sources.items():
            array = np.asarray(deltas, dtype=float)
            summary[group][source] = {
                "n": len(array),
                "median_frames": float(np.median(array)),
                "mean_frames": round(float(array.mean()), 2),
                "bad_rate": round(float((array > BAD_FRAMES).mean()), 3),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"bad_threshold_frames": BAD_FRAMES, "groups": summary},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("%-10s %-6s %6s %8s %8s %8s", "分組", "標註", "n", "中位", "平均", "大錯率")
    for group, sources in summary.items():
        for source, stats in sources.items():
            logger.info(
                "%-10s %-6s %6d %8.1f %8.2f %8.3f",
                group, source, stats["n"], stats["median_frames"],
                stats["mean_frames"], stats["bad_rate"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
