"""跑齊 `docs/spec.md` 的成功標準所需的實驗，把結果寫成 JSON。

一次載入資料，跑完全部設定——資料載入（含 1400 段姿態快取與弱標註推導）比訓練
本身還久，每個設定重載一次是浪費。

    python scripts/run_experiments.py --output runs/experiments.json

實驗設計：

S2  高爾夫 PCE 與 SwingNet 對照。用 GolfDB **官方四折**，每折訓練一次、在該折驗證，
    四折平均——這是 SwingNet 論文的協定，換成自訂的隨機切分就沒得比。
S4  聯合訓練 vs 單運動訓練。同樣的四折、同樣的超參數，唯一差別是訓練集。三個設定：

    ``golf_only``            只有 GolfDB（真人標註）
    ``joint``                GolfDB + Penn Action 全部六個運動
    ``joint_no_penn_golf``   GolfDB + Penn Action 的**其他五個**運動

    第三個設定用來隔離兩種可能的干擾來源：跨運動共用本身，還是「Penn Action 的
    高爾夫弱標註與 GolfDB 的真人標註灌進同一個事件槽，但兩者的事件定義不一致」。
    少了它，joint 比 golf_only 差時無從判斷該怪誰。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from kinetic_chain.data import split_clips
from kinetic_chain.evaluate import evaluate_clips, per_event_delta_table, predict_clips, sequence_violations
from kinetic_chain.model import ModelConfig
from kinetic_chain.train import TrainConfig, save_checkpoint, train

logger = logging.getLogger("experiments")

FOLDS = (1, 2, 3, 4)
SETTINGS = ("joint", "golf_only", "joint_no_penn_golf", "finetune_from_others")


def select_training_pool(clips: list, setting: str) -> list:
    """依設定挑出可用於訓練的片段。驗證集不受影響（一律只用 GolfDB 真人標註）。"""
    if setting == "golf_only":
        return [c for c in clips if c.dataset == "golfdb"]
    if setting == "joint_no_penn_golf":
        return [
            c for c in clips
            if not (c.dataset == "penn_action" and c.sport == "golf_swing")
        ]
    if setting in ("joint", "finetune_from_others"):
        return list(clips)
    raise ValueError(f"未知的設定 {setting!r}")


def load_all(args: argparse.Namespace) -> list:
    from kinetic_chain.datasets import golfdb, pennaction

    clips = golfdb.load(args.golfdb_annotations, args.golfdb_cache)
    clips += pennaction.load(args.pennaction_root)
    logger.info("共 %d 段", len(clips))
    return clips


def pretrain_on_others(
    clips: list, *, epochs: int, seed: int, device: str, output: Path
) -> Path:
    """在**不含高爾夫**的五個運動上預訓練，供 finetune 設定當起點。

    刻意排除高爾夫：這樣微調時看到的高爾夫資料只有 GolfDB 的真人標註，
    不會混進 Penn Action 高爾夫那份定義衝突的弱標註。
    """
    if output.is_file():
        return output
    pool = [
        c for c in clips
        if c.dataset == "penn_action" and c.sport != "golf_swing"
    ]
    train_clips, val_clips = split_clips(pool, val_fraction=0.1, seed=seed)
    config = TrainConfig(epochs=epochs, seed=seed, device=device, model=ModelConfig())
    model, _ = train(train_clips, val_clips, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, config, output)
    logger.info("預訓練完成（%d 段，五個運動）→ %s", len(train_clips), output)
    return output


def run_one(
    clips: list,
    *,
    fold: int,
    setting: str,
    epochs: int,
    seed: int,
    device: str,
    pretrained: Path | None = None,
) -> dict:
    if setting == "finetune_from_others":
        # 微調：訓練資料只有 GolfDB，但權重從五個運動的預訓練模型出發
        subset = [c for c in clips if c.dataset == "golfdb"]
    else:
        subset = select_training_pool(clips, setting)
    train_clips, val_clips = split_clips(subset, seed=seed, val_fold=fold)
    # 驗證集一律只看 GolfDB 的真人標註，各設定才可比
    val_human = [c for c in val_clips if c.label_source == "human"]

    config = TrainConfig(
        epochs=epochs,
        seed=seed,
        device=device,
        model=ModelConfig(),
        init_from=str(pretrained) if setting == "finetune_from_others" else None,
    )
    model, history = train(train_clips, val_human, config)

    reports = evaluate_clips(model, val_human, device=device)
    predictions = predict_clips(model, val_human, device=device)
    return {
        "fold": fold,
        "setting": setting,
        "train_clips": len(train_clips),
        "val_clips": len(val_human),
        "pce": reports["overall"].pce,
        "mean_tolerance": reports["overall"].mean_tolerance,
        "per_event": {
            name: {"pce": score.pce, "median_delta": score.median_delta}
            for name, score in reports["overall"].per_event.items()
        },
        "order_violations": sequence_violations(val_human, predictions),
        "final_train_loss": history["epochs"][-1]["train_loss"],
    }


def run_multisport(clips: list, *, epochs: int, seed: int, device: str) -> dict:
    """S1：單一組權重同時輸出多個運動的事件。

    這一次用隨機分層切分（Penn Action 沒有官方切分），只回報每個運動的分數，
    並保持 human / weak 分開。
    """
    train_clips, val_clips = split_clips(clips, val_fraction=0.2, seed=seed)
    config = TrainConfig(epochs=epochs, seed=seed, device=device, model=ModelConfig())
    model, _ = train(train_clips, val_clips, config)

    reports = evaluate_clips(model, val_clips, device=device)
    predictions = predict_clips(model, val_clips, device=device)
    return {
        "train_clips": len(train_clips),
        "val_clips": len(val_clips),
        "pce": {name: report.pce for name, report in sorted(reports.items())},
        "clips_per_group": {
            name: report.num_clips for name, report in sorted(reports.items())
        },
        "order_violations": sequence_violations(val_clips, predictions),
        "per_event_delta": per_event_delta_table(val_clips, predictions),
        "parameters": model.num_parameters,
        "receptive_field": model.config.receptive_field,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golfdb-annotations", type=Path, default=Path("data/raw/golfDB.pkl"))
    parser.add_argument("--golfdb-cache", type=Path, default=Path("data/cache/golfdb_pose"))
    parser.add_argument("--pennaction-root", type=Path, default=Path("data/raw/Penn_Action"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/experiments.json"))
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=SETTINGS)
    parser.add_argument("--skip-multisport", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("kinetic_chain.train").setLevel(logging.WARNING)

    clips = load_all(args)
    results: dict = {"folds": [], "multisport": None}

    pretrained = None
    if "finetune_from_others" in args.settings:
        pretrained = pretrain_on_others(
            clips,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            output=args.output.parent / "pretrain_five_sports.pt",
        )

    for fold in FOLDS:
        for setting in args.settings:
            record = run_one(
                clips,
                fold=fold,
                setting=setting,
                epochs=args.epochs,
                seed=args.seed,
                device=args.device,
                pretrained=pretrained,
            )
            logger.info(
                "fold %d  %-20s  PCE %.4f  (%d train / %d val)",
                fold,
                record["setting"],
                record["pce"],
                record["train_clips"],
                record["val_clips"],
            )
            results["folds"].append(record)

    for setting in args.settings:
        scores = [r["pce"] for r in results["folds"] if r["setting"] == setting]
        results[f"{setting}_mean_pce"] = float(np.mean(scores))
        results[f"{setting}_std_pce"] = float(np.std(scores))
        logger.info("%s 四折平均 PCE %.4f ± %.4f", setting, np.mean(scores), np.std(scores))

    if not args.skip_multisport:
        results["multisport"] = run_multisport(
            clips, epochs=args.epochs, seed=args.seed, device=args.device
        )
        logger.info("多運動： %s", results["multisport"]["pce"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已寫入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
