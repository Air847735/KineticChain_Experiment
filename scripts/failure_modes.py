"""失敗集中在哪裡：整段崩掉，還是特定事件？

`scripts/error_budget.py` 顯示各運動的誤差中位數都只有 1 格，但平均高出 3 到 9 倍，
代表少數片段錯得離譜。要改善就得知道那些錯誤是**整段崩掉**（資料或姿態問題）
還是**特定事件**（事件定義或弱標註問題）——兩者的修法完全不同。

    python scripts/failure_modes.py
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

logger = logging.getLogger("failure_modes")

RUNS: tuple[tuple[str, str], ...] = (
    ("golf_swing", "runs/golf/model.pt"),
    ("baseball_pitch", "runs/pitch/model.pt"),
    ("baseball_swing", "runs/bat/model.pt"),
    ("clean_and_jerk", "runs/lift/model.pt"),
)

#: 超過這個影格數視為「大錯」——不是差一格的取樣誤差，是抓錯位置。
BAD_FRAMES = 5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pennaction-root", type=Path, default=Path("data/raw/Penn_Action"))
    parser.add_argument("--golfdb", type=Path, default=Path("data/raw/golfDB.pkl"))
    parser.add_argument("--golfdb-cache", type=Path, default=Path("data/cache/golfdb_pose"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/failure_modes.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from kinetic_chain.datasets import golfdb, pennaction

    results = []
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
        model = load_checkpoint(path, device=args.device)
        predictions = predict_clips(model, val_clips, device=args.device)

        # (片段, 事件) 的大錯矩陣
        events = list(spec.events)
        grid = np.zeros((len(val_clips), len(events)), dtype=bool)
        scored = np.zeros_like(grid)
        for i, (clip, prediction) in enumerate(zip(val_clips, predictions)):
            for j, event in enumerate(events):
                if event not in prediction or event not in clip.events:
                    continue
                scored[i, j] = True
                grid[i, j] = abs(prediction[event] - clip.events[event]) > BAD_FRAMES

        bad_total = int(grid.sum())
        per_clip = grid.sum(axis=1)
        per_event = grid.sum(axis=0)
        n_events = int(scored[0].sum()) or 1

        # 大錯集中在少數片段，還是散在所有片段？
        clips_with_any = int((per_clip > 0).sum())
        clips_mostly_bad = int((per_clip >= n_events / 2).sum())
        share_from_worst_clips = (
            float(np.sort(per_clip)[::-1][: max(len(val_clips) // 5, 1)].sum() / bad_total)
            if bad_total
            else 0.0
        )

        row = {
            "sport": sport,
            "val_clips": len(val_clips),
            "events_per_clip": n_events,
            "bad_threshold_frames": BAD_FRAMES,
            "bad_rate": round(bad_total / max(int(scored.sum()), 1), 3),
            "clips_with_any_bad": clips_with_any,
            "clips_mostly_bad": clips_mostly_bad,
            "share_of_errors_from_worst_fifth": round(share_from_worst_clips, 3),
            "per_event_bad_rate": {
                events[j]: round(float(per_event[j] / max(int(scored[:, j].sum()), 1)), 3)
                for j in range(len(events))
            },
        }
        results.append(row)

        logger.info(
            "%-16s 大錯率 %.3f｜有大錯的片段 %d/%d，過半崩掉 %d｜最差 1/5 的片段佔全部錯誤 %.0f%%",
            sport, row["bad_rate"], clips_with_any, len(val_clips),
            clips_mostly_bad, 100 * share_from_worst_clips,
        )
        worst = sorted(row["per_event_bad_rate"].items(), key=lambda kv: -kv[1])[:4]
        logger.info("    最容易大錯的事件：%s", worst)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"runs": results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
