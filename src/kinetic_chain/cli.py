"""命令列介面。薄封裝，不含任何業務邏輯。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .data import split_clips
from .errors import KineticChainError
from .events import get_sport, registered_sports
from .segment import ACTIVE_FRACTION, MIN_ACTION_SECONDS

DEFAULT_GOLFDB_ANNOTATIONS = Path("data/raw/golfDB.pkl")
DEFAULT_GOLFDB_VIDEOS = Path("data/raw/videos_160")
DEFAULT_GOLFDB_CACHE = Path("data/cache/golfdb_pose")
DEFAULT_PENN_ROOT = Path("data/raw/Penn_Action")


def _load_dataset(args: argparse.Namespace) -> list:
    """依旗標載入資料集並合併。"""
    from .datasets import golfdb, pennaction

    clips = []
    if not args.no_golfdb:
        clips.extend(
            golfdb.load(
                args.golfdb_annotations,
                args.golfdb_cache,
                limit=args.limit,
            )
        )
    if not args.no_pennaction:
        clips.extend(
            pennaction.load(
                args.pennaction_root,
                sports=args.sports or None,
                limit=args.limit,
            )
        )
    if args.sports:
        wanted = set(args.sports)
        clips = [c for c in clips if c.sport in wanted]
    return clips


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--golfdb-annotations", type=Path, default=DEFAULT_GOLFDB_ANNOTATIONS)
    parser.add_argument("--golfdb-cache", type=Path, default=DEFAULT_GOLFDB_CACHE)
    parser.add_argument("--pennaction-root", type=Path, default=DEFAULT_PENN_ROOT)
    parser.add_argument("--no-golfdb", action="store_true", help="不載入 GolfDB")
    parser.add_argument("--no-pennaction", action="store_true", help="不載入 Penn Action")
    parser.add_argument(
        "--sport",
        dest="sports",
        nargs="+",
        default=None,
        metavar="SPORT",
        help="要訓練／評估的運動項目 id。一般用法是指定單一項目——"
        "實測顯示單運動訓練優於多運動聯合訓練。給多個則為聯合訓練。",
    )
    parser.add_argument("--limit", type=int, default=None, help="每個資料集只讀前 N 段")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--val-fold",
        type=int,
        default=None,
        help="以資料集官方切分的第 N 折當驗證集（GolfDB 為 1–4）。"
        "沒有官方切分的資料集仍照 --val-fraction 隨機切。",
    )
    parser.add_argument("--seed", type=int, default=0)


def cmd_sports(args: argparse.Namespace) -> int:
    """列出已註冊的運動項目與其事件。"""
    for sport_id in registered_sports():
        spec = get_sport(sport_id)
        print(f"{sport_id}  ({spec.display_name})")
        for i, event in enumerate(spec.events):
            print(f"  {i}. {event}")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    """對 GolfDB 影片跑姿態抽取並存快取。"""
    from .datasets import golfdb

    stats = golfdb.extract_poses(
        args.golfdb_annotations,
        args.golfdb_videos,
        args.golfdb_cache,
        device=args.pose_device,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(stats, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .model import ModelConfig
    from .train import TrainConfig, train

    clips = _load_dataset(args)
    if not clips:
        print("沒有載入任何片段", file=sys.stderr)
        return 1

    train_clips, val_clips = split_clips(
        clips,
        val_fraction=args.val_fraction,
        seed=args.seed,
        val_fold=args.val_fold,
    )
    sports = sorted({c.sport for c in clips})
    print(
        f"運動項目 {sports}；訓練 {len(train_clips)} 段 / 驗證 {len(val_clips)} 段"
    )

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        augment=not args.no_augment,
        model=ModelConfig(hidden=args.hidden, num_layers=args.layers),
        init_from=str(args.init_from) if args.init_from else None,
        freeze_backbone=args.freeze_backbone,
    )
    _, history = train(train_clips, val_clips, config, output_dir=args.output)
    print(f"best selection score: {history['best_score']:.4f}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_clips, format_reports, predict_clips, sequence_violations
    from .train import load_checkpoint

    model = load_checkpoint(args.checkpoint, device=args.device)
    clips = _load_dataset(args)
    if not clips:
        print("沒有載入任何片段", file=sys.stderr)
        return 1

    if args.split in ("train", "val"):
        train_clips, val_clips = split_clips(
            clips,
            val_fraction=args.val_fraction,
            seed=args.seed,
            val_fold=args.val_fold,
        )
        clips = val_clips if args.split == "val" else train_clips

    reports = evaluate_clips(model, clips, device=args.device)
    print(format_reports(reports))

    predictions = predict_clips(model, clips, device=args.device)
    violations = sequence_violations(clips, predictions)
    print(f"\n順序違反：{violations} / {len(clips)} 段（解碼器保證應為 0）")
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    from .infer import predict_video
    from .train import load_checkpoint

    model = load_checkpoint(args.checkpoint, device=args.device)
    result = predict_video(
        model,
        args.video,
        args.sport,
        device=args.device,
        pose_device=args.pose_device,
        bbox_strategy=args.bbox_strategy,
    )
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.format())
    return 0


def cmd_segment(args: argparse.Namespace) -> int:
    """把一支長影片切成一次次的動作，可選擇對每一段跑推論。"""
    from .features import compute
    from .pose import PoseExtractor
    from .segment import find_actions
    from .skeleton import to_canonical

    extractor = PoseExtractor(device=args.pose_device, bbox_strategy=args.bbox_strategy)
    sequence = extractor.extract_video(args.video, progress=True)
    pose = to_canonical(sequence.keypoints, sequence.layout)
    fps = sequence.fps
    signals = compute(pose, fps, handedness_sensitive=False)
    report = find_actions(
        signals, fps,
        activity_signal=args.activity_signal,
        active_fraction=args.active_fraction,
        min_action_seconds=args.min_action_seconds,
    )

    payload = report.as_dict()
    payload["video"] = str(args.video)
    payload["fps"] = fps
    payload["num_frames"] = int(pose.shape[0])
    for entry, seg in zip(payload["segments"], report.segments):
        lo, hi = seg.seconds(fps)
        entry["start_time"] = round(lo, 3)
        entry["end_time"] = round(hi, 3)

    # 每一段各自跑一次推論。切分只給邊界，事件仍由既有的模型與解碼負責。
    if args.checkpoint is not None:
        if args.sport is None:
            raise KineticChainError("給了 --checkpoint 就必須同時給 --sport")
        from .infer import predict_pose_sequence
        from .train import load_checkpoint

        model = load_checkpoint(args.checkpoint, device=args.device)
        for entry, seg in zip(payload["segments"], report.segments):
            result = predict_pose_sequence(
                model, pose[seg.start : seg.end], fps, args.sport, device=args.device
            )
            entry["events"] = result.as_dict()["events"]

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"{args.video}  {payload['num_frames']} 影格 @ {fps:.2f} fps")
        print(f"  切出 {len(report.segments)} 段（活動量訊號 {report.activity_signal}）")
        print(f"  {'可信' if report.should_trust else '**不可信**'}：{report.reason}")
        for index, (entry, seg) in enumerate(zip(payload["segments"], report.segments), 1):
            lo, hi = seg.seconds(fps)
            print(
                f"  第 {index:>2} 段  f{seg.start:>6}–{seg.end:<6}"
                f"  {lo:>7.2f}–{hi:<7.2f} 秒  ({seg.num_frames} 格)"
                f"  峰值強度 {seg.peak_activity:.2f}"
            )
    return 0 if report.should_trust else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kinetic-chain",
        description="以運動項目為條件，從影片偵測動力鏈關鍵時間點",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("sports", help="列出已註冊的運動項目").set_defaults(func=cmd_sports)

    extract = sub.add_parser("extract", help="對 GolfDB 影片跑姿態抽取")
    extract.add_argument("--golfdb-annotations", type=Path, default=DEFAULT_GOLFDB_ANNOTATIONS)
    extract.add_argument("--golfdb-videos", type=Path, default=DEFAULT_GOLFDB_VIDEOS)
    extract.add_argument("--golfdb-cache", type=Path, default=DEFAULT_GOLFDB_CACHE)
    extract.add_argument("--pose-device", default="cuda")
    extract.add_argument("--limit", type=int, default=None)
    extract.add_argument("--overwrite", action="store_true")
    extract.set_defaults(func=cmd_extract)

    trainer = sub.add_parser("train", help="訓練模型")
    _add_dataset_args(trainer)
    trainer.add_argument("--epochs", type=int, default=60)
    trainer.add_argument("--batch-size", type=int, default=16)
    trainer.add_argument("--learning-rate", type=float, default=3e-4)
    trainer.add_argument("--hidden", type=int, default=128)
    trainer.add_argument("--layers", type=int, default=6)
    trainer.add_argument("--device", default="cuda")
    trainer.add_argument("--no-augment", action="store_true")
    trainer.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="從既有 checkpoint 的權重出發（跨運動微調），而不是隨機初始化",
    )
    trainer.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="只訓練輸出頭與運動項目 embedding；新運動資料量小時使用",
    )
    trainer.add_argument("--output", type=Path, default=Path("runs/latest"))
    trainer.set_defaults(func=cmd_train)

    evaluator = sub.add_parser("eval", help="評估 checkpoint")
    _add_dataset_args(evaluator)
    evaluator.add_argument("--checkpoint", type=Path, required=True)
    evaluator.add_argument("--device", default="cuda")
    evaluator.add_argument("--split", choices=["all", "train", "val"], default="val")
    evaluator.set_defaults(func=cmd_eval)

    infer = sub.add_parser("infer", help="對單支影片推論")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--video", type=Path, required=True)
    infer.add_argument("--sport", required=True)
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--pose-device", default="cuda")
    infer.add_argument("--bbox-strategy", choices=["detect", "whole_frame"], default="detect")
    infer.add_argument("--json", action="store_true")
    infer.set_defaults(func=cmd_infer)

    segment = sub.add_parser(
        "segment",
        help="把未裁切的長影片切成一次次的動作；給 --checkpoint 就順便對每段推論",
    )
    segment.add_argument("--video", type=Path, required=True)
    segment.add_argument(
        "--checkpoint", type=Path, default=None,
        help="給了就對每一段跑事件推論；不給就只輸出邊界",
    )
    segment.add_argument("--sport", default=None, help="搭配 --checkpoint 使用")
    segment.add_argument(
        "--activity-signal", default="body_speed",
        help="當活動量的訊號。器械類動作身體位移小時改用 wrist_speed",
    )
    segment.add_argument("--active-fraction", type=float, default=ACTIVE_FRACTION)
    segment.add_argument("--min-action-seconds", type=float, default=MIN_ACTION_SECONDS)
    segment.add_argument("--device", default="cuda")
    segment.add_argument("--pose-device", default="cuda")
    segment.add_argument(
        "--bbox-strategy", choices=["detect", "whole_frame"], default="detect"
    )
    segment.add_argument("--json", action="store_true")
    segment.set_defaults(func=cmd_segment)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.func(args))
    except KineticChainError as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
