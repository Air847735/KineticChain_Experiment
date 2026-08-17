"""新運動需要多少標註資料？從頭訓練 vs 從別的運動微調。

這是「後續每個運動各自訓練或微調」這條路線最實際的問題：手上有一個新運動、
標了 N 段，該從頭訓練還是拿別的運動的權重來微調？

作法：把 GolfDB 的訓練折子取樣成不同大小，同樣的驗證折子上比較兩種起點。
以高爾夫當代理——它是唯一有真人標註的運動，可以放心把它當新運動來模擬。

    python scripts/data_efficiency.py --output runs/data_efficiency.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from kinetic_chain.data import split_clips
from kinetic_chain.evaluate import evaluate_clips
from kinetic_chain.model import ModelConfig
from kinetic_chain.train import TrainConfig, train

logger = logging.getLogger("data_efficiency")

SIZES = (25, 50, 100, 200, 400, None)  # None = 全部
FOLDS = (1, 2, 3, 4)


def subsample(clips: list, size: int | None, seed: int) -> list:
    if size is None or size >= len(clips):
        return clips
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clips))[:size]
    return [clips[i] for i in order]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golfdb-annotations", type=Path, default=Path("data/raw/golfDB.pkl"))
    parser.add_argument("--golfdb-cache", type=Path, default=Path("data/cache/golfdb_pose"))
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path("runs/pretrain_five_sports.pt"),
        help="由 scripts/run_experiments.py 的 finetune_from_others 設定產生",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/data_efficiency.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("kinetic_chain.train").setLevel(logging.WARNING)

    if not args.pretrained.is_file():
        raise SystemExit(
            f"找不到預訓練 checkpoint {args.pretrained}；先跑：\n"
            "  python scripts/run_experiments.py --settings finetune_from_others "
            "--skip-multisport --output runs/experiments_finetune.json"
        )

    from kinetic_chain.datasets import golfdb

    clips = golfdb.load(args.golfdb_annotations, args.golfdb_cache)
    records = []

    for size in SIZES:
        for start in ("scratch", "finetune"):
            scores = []
            for fold in FOLDS:
                train_clips, val_clips = split_clips(clips, seed=args.seed, val_fold=fold)
                pool = subsample(train_clips, size, args.seed + fold)
                config = TrainConfig(
                    epochs=args.epochs,
                    seed=args.seed,
                    device=args.device,
                    model=ModelConfig(),
                    init_from=str(args.pretrained) if start == "finetune" else None,
                )
                model, _ = train(pool, val_clips, config)
                scores.append(evaluate_clips(model, val_clips, device=args.device)["overall"].pce)

            record = {
                "size": size if size is not None else len(pool),
                "start": start,
                "pce_mean": float(np.mean(scores)),
                "pce_std": float(np.std(scores)),
                "per_fold": scores,
            }
            records.append(record)
            logger.info(
                "n=%-5s %-9s PCE %.4f ± %.4f",
                record["size"],
                start,
                record["pce_mean"],
                record["pce_std"],
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"已寫入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
