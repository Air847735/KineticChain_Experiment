"""打擊表現差，是因為標註是弱標註，還是因為資料只有 88 段？

這兩個原因的修法完全相反：前者要重寫規則或人工標註，後者只要多蒐集影片。
先前無法分辨，因為高爾夫（真人標註、1042 段）與打擊（弱標註、88 段）兩個變因綁在一起。

作法：把高爾夫的訓練集降到與打擊相同的段數，其餘不變，再用**固定容忍度**評估
（PCE 自身的容忍度隨片段長度變動，跨運動不可比）。若降量後的高爾夫掉到打擊的水準，
瓶頸就是資料量；若仍明顯較高，差距要歸到標註品質。

    python scripts/label_vs_data.py
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
from kinetic_chain.train import ModelConfig, TrainConfig, load_checkpoint, train

logger = logging.getLogger("label_vs_data")

#: 打擊的訓練段數，作為降量的目標。
BAT_TRAIN_SIZE = 88

#: 固定容忍度（影格）。2 格 = 67 ms，約為 30 fps 下能分辨的最小單位的兩倍。
FIXED_TOLERANCE = 2

FOLDS = (1, 2, 3, 4)


def subsample(clips: list[Clip], size: int, seed: int) -> list[Clip]:
    if size >= len(clips):
        return clips
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clips))[:size]
    return [clips[i] for i in order]


def fixed_tolerance_pce(
    clips: list[Clip], predictions: list[dict[str, int]], sport: str, tol: int
) -> float:
    spec = get_sport(sport)
    hits = [
        abs(prediction[event] - clip.events[event]) <= tol
        for clip, prediction in zip(clips, predictions)
        for event in spec.events
        if event in prediction and event in clip.events
    ]
    return float(np.mean(hits))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golfdb", type=Path, default=Path("data/raw/golfDB.pkl"))
    parser.add_argument("--golfdb-cache", type=Path, default=Path("data/cache/golfdb_pose"))
    parser.add_argument("--pennaction-root", type=Path, default=Path("data/raw/Penn_Action"))
    parser.add_argument("--bat-checkpoint", type=Path, default=Path("runs/bat/model.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/label_vs_data.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("kinetic_chain.train").setLevel(logging.WARNING)

    from kinetic_chain.datasets import golfdb, pennaction

    golf = golfdb.load(args.golfdb, args.golfdb_cache)
    results: dict = {"fixed_tolerance_frames": FIXED_TOLERANCE, "golf": {}}

    for size_label, size in (("full", None), (f"n={BAT_TRAIN_SIZE}", BAT_TRAIN_SIZE)):
        scores = []
        for fold in FOLDS:
            train_clips, val_clips = split_clips(golf, seed=args.seed, val_fold=fold)
            pool = train_clips if size is None else subsample(train_clips, size, args.seed + fold)
            config = TrainConfig(
                epochs=args.epochs, seed=args.seed, device=args.device, model=ModelConfig()
            )
            model, _ = train(pool, val_clips, config)
            predictions = predict_clips(model, val_clips, device=args.device)
            scores.append(
                fixed_tolerance_pce(val_clips, predictions, "golf_swing", FIXED_TOLERANCE)
            )
            logger.info(
                "golf %-6s fold %d  訓練 %4d 段  PCE@%d格 %.4f",
                size_label, fold, len(pool), FIXED_TOLERANCE, scores[-1],
            )
        results["golf"][size_label] = {
            "train_clips": len(pool),
            "pce_mean": round(float(np.mean(scores)), 4),
            "pce_std": round(float(np.std(scores)), 4),
            "per_fold": [round(s, 4) for s in scores],
        }

    # 打擊用已訓練好的權重，不重訓——比較的是它現有的水準
    bat_clips = pennaction.load(args.pennaction_root, sports=["baseball_swing"])
    bat_train, bat_val = split_clips(bat_clips, val_fraction=0.2, seed=0)
    bat_model = load_checkpoint(args.bat_checkpoint, device=args.device)
    bat_predictions = predict_clips(bat_model, bat_val, device=args.device)
    results["baseball_swing"] = {
        "train_clips": len(bat_train),
        "label_source": "weak",
        "pce": round(
            fixed_tolerance_pce(bat_val, bat_predictions, "baseball_swing", FIXED_TOLERANCE), 4
        ),
    }

    full = results["golf"]["full"]["pce_mean"]
    reduced = results["golf"][f"n={BAT_TRAIN_SIZE}"]["pce_mean"]
    bat = results["baseball_swing"]["pce"]
    results["attribution"] = {
        "cost_of_less_data": round(full - reduced, 4),
        "residual_gap_to_batting": round(reduced - bat, 4),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("=" * 62)
    logger.info("固定容忍度 %d 格，全部可比：", FIXED_TOLERANCE)
    logger.info("  高爾夫 真人標註 1042 段   PCE %.4f", full)
    logger.info("  高爾夫 真人標註   %2d 段   PCE %.4f  （少資料的代價 %.4f）",
                BAT_TRAIN_SIZE, reduced, full - reduced)
    logger.info("  打擊   弱標註     %2d 段   PCE %.4f  （剩餘差距 %.4f）",
                results["baseball_swing"]["train_clips"], bat, reduced - bat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
