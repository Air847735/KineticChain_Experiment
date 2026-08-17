"""訓練迴圈。

訓練程式不認識任何特定運動：它只看到 :class:`~kinetic_chain.data.Clip`、
運動項目索引與事件遮罩。新增運動項目不需要動這個檔案。
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .data import Batch, Clip, collate
from .errors import DatasetError
from .evaluate import evaluate_clips
from .model import KineticChainNet, ModelConfig, event_loss, soft_targets

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """訓練超參數。整份存進 checkpoint，實驗才可重現。"""

    epochs: int = 60
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    sigma_seconds: float = 0.05
    grad_clip: float = 5.0
    seed: int = 0
    device: str = "cuda"
    # 資料增強
    time_scale_range: tuple[float, float] = (0.8, 1.25)
    feature_noise: float = 0.01
    augment: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    # 微調：從既有 checkpoint 的權重出發，而不是隨機初始化
    init_from: str | None = None
    freeze_backbone: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resample(features: np.ndarray, factor: float) -> np.ndarray:
    """沿時間軸線性重取樣，模擬不同的動作速度。

    比在特徵上加雜訊更貼近真實變異：同一個動作由不同人做、或用不同幀率拍攝，
    事件之間的間隔本來就會伸縮。
    """
    n = features.shape[0]
    target = max(8, int(round(n * factor)))
    source = np.linspace(0.0, n - 1.0, target)
    lower = np.floor(source).astype(np.int64)
    upper = np.minimum(lower + 1, n - 1)
    weight = (source - lower)[:, None].astype(np.float32)
    return features[lower] * (1.0 - weight) + features[upper] * weight


def augment_clip(clip: Clip, config: TrainConfig, rng: np.random.Generator) -> Clip:
    """回傳一個增強過的淺複本。原片段不被修改。"""
    features = clip.features()
    n = features.shape[0]

    low, high = config.time_scale_range
    factor = float(rng.uniform(low, high))
    resampled = _resample(features, factor)
    ratio = (resampled.shape[0] - 1) / max(n - 1, 1)
    events = {
        name: int(np.clip(round(frame * ratio), 0, resampled.shape[0] - 1))
        for name, frame in clip.events.items()
    }

    if config.feature_noise > 0:
        scale = config.feature_noise * (np.std(resampled, axis=0, keepdims=True) + 1e-6)
        resampled = resampled + rng.normal(0.0, 1.0, resampled.shape).astype(np.float32) * scale

    augmented = Clip(
        clip_id=clip.clip_id,
        sport=clip.sport,
        # 共用原片段的姿態陣列（唯讀，不複製）。增強後的特徵直接塞進快取，
        # 這份姿態不會再被讀到，但保留它才能讓 Clip 的長度檢查有意義。
        pose=clip.pose,
        fps=clip.fps * factor,
        events=events,
        label_source=clip.label_source,
        dataset=clip.dataset,
        coverage=clip.coverage,
    )
    augmented._features = resampled.astype(np.float32)
    return augmented


class _BucketedBatches:
    """依長度分桶再組批，減少 padding。

    片段長度從數十到數百影格不等，隨機組批會讓短片段陪著最長的片段一起補到
    同樣長度，多數計算浪費在 padding 上。分桶後每批的長度接近。
    """

    def __init__(self, clips: Sequence[Clip], batch_size: int, *, shuffle: bool = True) -> None:
        self.clips = list(clips)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        order = sorted(range(len(self.clips)), key=lambda i: self.clips[i].num_frames)
        batches = [
            [self.clips[i] for i in order[start : start + self.batch_size]]
            for start in range(0, len(order), self.batch_size)
        ]
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return math.ceil(len(self.clips) / self.batch_size)


def _step(
    model: KineticChainNet,
    batch: Batch,
    device: torch.device,
) -> torch.Tensor:
    batch = batch.to(device)
    logits = model(batch.features, batch.sport_ids, batch.frame_mask)
    targets = soft_targets(
        batch.targets, logits.shape[-1], batch.sigma, batch.frame_mask
    )
    return event_loss(logits, targets, batch.event_mask)


def _initialise_from(model: KineticChainNet, path: str, device: torch.device) -> None:
    """以既有 checkpoint 的權重初始化，供跨運動微調使用。

    形狀不合時直接拋錯而不是部分載入：部分載入會靜默留下一半隨機權重，
    看起來有在微調，實際上是在半隨機的起點上重訓。
    """
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    source = ModelConfig(**payload["model_config"])
    if asdict(source) != asdict(model.config):
        raise DatasetError(
            "微調來源的模型結構與目前設定不同，權重無法對應。\n"
            f"來源: {asdict(source)}\n目前: {asdict(model.config)}"
        )
    model.load_state_dict(payload["state_dict"])
    logger.info("以 %s 的權重初始化", path)


def _freeze_backbone(model: KineticChainNet) -> int:
    """凍結特徵抽取層，只留輸出頭與運動項目 embedding 可訓練。

    資料量小的新運動用得上：骨幹已經在別的運動上學會怎麼讀動作訊號，
    只需要重學「哪些訊號對應到哪個事件」。
    """
    frozen = 0
    for module in (model.input_proj, model.blocks):
        for parameter in module.parameters():
            parameter.requires_grad = False
            frozen += parameter.numel()
    return frozen


def train(
    train_clips: Sequence[Clip],
    val_clips: Sequence[Clip],
    config: TrainConfig | None = None,
    *,
    output_dir: Path | str | None = None,
) -> tuple[KineticChainNet, dict]:
    """訓練模型並回傳 ``(model, history)``。

    每個 epoch 結束後在驗證集上算 PCE；以「真人標註」分組的 PCE 為挑選最佳
    checkpoint 的依據，沒有真人標註時退回整體 PCE——弱標註上的分數不足以當作
    模型好壞的判準。
    """
    config = config or TrainConfig()
    if not train_clips:
        raise DatasetError("訓練集是空的")
    set_seed(config.seed)

    device = torch.device(
        config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if str(device) != config.device:
        logger.warning("要求的裝置 %s 不可用，改用 %s", config.device, device)

    model = KineticChainNet(config.model).to(device)
    if config.init_from:
        _initialise_from(model, config.init_from, device)
    if config.freeze_backbone:
        frozen = _freeze_backbone(model)
        logger.info("凍結骨幹：%d 個參數不更新", frozen)
    logger.info(
        "模型參數 %d（可訓練 %d）；感受野 %d 影格",
        model.num_parameters,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        model.config.receptive_field,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    batches = _BucketedBatches(train_clips, config.batch_size)
    steps_per_epoch = len(batches)

    def lr_scale(epoch: int) -> float:
        if epoch < config.warmup_epochs:
            return (epoch + 1) / max(config.warmup_epochs, 1)
        progress = (epoch - config.warmup_epochs) / max(
            config.epochs - config.warmup_epochs, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    rng = np.random.default_rng(config.seed)
    history: dict = {"config": asdict(config), "epochs": []}
    best_score = -math.inf
    best_state: dict | None = None

    for epoch in range(config.epochs):
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * lr_scale(epoch)

        model.train()
        started = time.time()
        total = 0.0
        for clips in batches:
            if config.augment:
                clips = [augment_clip(clip, config, rng) for clip in clips]
            loss = _step(model, collate(clips, sigma_seconds=config.sigma_seconds), device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            total += float(loss.detach())

        train_loss = total / max(steps_per_epoch, 1)
        reports = evaluate_clips(model, val_clips, device=device) if val_clips else {}
        human = [k for k in reports if k.endswith("/human")]
        score = (
            float(np.mean([reports[k].pce for k in human]))
            if human
            else reports.get("overall").pce
            if "overall" in reports
            else -math.inf
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "seconds": time.time() - started,
            "lr": optimizer.param_groups[0]["lr"],
            "pce": {k: report.pce for k, report in sorted(reports.items())},
            "selection_score": score,
        }
        history["epochs"].append(record)
        logger.info(
            "epoch %3d  loss %.4f  score %.4f  %.1fs  %s",
            epoch,
            train_loss,
            score,
            record["seconds"],
            {k: round(v, 3) for k, v in record["pce"].items()},
        )

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_score"] = best_score

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, config, output_dir / "model.pt")
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("已寫入 %s", output_dir)

    return model, history


def save_checkpoint(model: KineticChainNet, config: TrainConfig, path: Path | str) -> None:
    """存權重與重建模型所需的設定，外加運動項目清單。

    運動項目清單必須一起存：``sport_index`` 依註冊表排序決定，註冊表改變時
    舊 checkpoint 的 embedding 索引就對不上了，載入時要能發現這件事。
    """
    from .events import registered_sports

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": asdict(model.config),
            "train_config": asdict(config),
            "sports": list(registered_sports()),
            "version": 1,
        },
        Path(path),
    )


def load_checkpoint(path: Path | str, *, device: str = "cpu") -> KineticChainNet:
    """讀回 :func:`save_checkpoint` 存下的 checkpoint。"""
    from .events import registered_sports

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    stored = list(payload.get("sports", []))
    current = list(registered_sports())
    if stored and stored != current:
        raise DatasetError(
            "checkpoint 的運動項目清單與目前的註冊表不一致，sport embedding 索引"
            f"會對不上。\ncheckpoint: {stored}\n目前:       {current}"
        )
    model = KineticChainNet(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model
