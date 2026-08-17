"""Sport-conditioned 時序事件偵測模型。

一組權重、一個輸出頭，所有運動共用。運動項目以 FiLM 條件進入**每一層**，
而不是拼在輸入上——後者經過幾層卷積後影響會被稀釋，模型會退化成忽略條件的
單一通用模型。

輸出是 ``(B, E, T)``：每個事件槽在每個影格的 logit。時間軸上做 softmax 後即
「這個事件發生在哪一格」的分布。運動項目沒有宣告的事件槽由遮罩排除，不參與
損失也不參與解碼。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .events import NUM_EVENT_SLOTS, registered_sports
from .features import NUM_FEATURES


@dataclass
class ModelConfig:
    """模型超參數。存進 checkpoint 以便重建同樣的結構。"""

    num_features: int = NUM_FEATURES
    num_events: int = NUM_EVENT_SLOTS
    num_sports: int = 0  # 0 表示建構時以註冊表大小填入
    hidden: int = 128
    sport_embedding: int = 32
    num_layers: int = 6
    kernel_size: int = 5
    dropout: float = 0.1

    def resolved(self) -> "ModelConfig":
        if self.num_sports == 0:
            return ModelConfig(**{**asdict(self), "num_sports": len(registered_sports())})
        return self

    @property
    def receptive_field(self) -> int:
        """雙向感受野（影格）。dilation 逐層加倍。"""
        span = 1
        for layer in range(self.num_layers):
            span += 2 * (self.kernel_size - 1) * (2**layer)
        return span


class FiLM(nn.Module):
    """由運動項目 embedding 產生逐通道的仿射調變參數。"""

    def __init__(self, embedding_dim: int, channels: int) -> None:
        super().__init__()
        self.to_gamma = nn.Linear(embedding_dim, channels)
        self.to_beta = nn.Linear(embedding_dim, channels)
        # 初始化成恆等轉換：訓練初期條件不干擾，之後再逐漸分化
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: torch.Tensor, sport: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(sport).unsqueeze(-1)
        beta = self.to_beta(sport).unsqueeze(-1)
        return gamma * x + beta


class ConditionedBlock(nn.Module):
    """帶 FiLM 條件的膨脹殘差卷積塊。

    非因果（雙向）卷積：離線分析看得到未來影格，沒有理由自綁因果限制。
    padding 用 ``dilation * (kernel - 1) // 2`` 保持長度不變，逐影格解析度不損失。
    """

    def __init__(
        self, channels: int, kernel_size: int, dilation: int, embedding_dim: int, dropout: float
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.film = FiLM(embedding_dim, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sport: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.norm(h)
        h = self.film(h, sport)
        h = self.activation(h)
        h = self.dropout(h)
        return x + h


class KineticChainNet(nn.Module):
    """特徵序列 + 運動項目 → 每個事件槽的逐影格 logits。

    Parameters
    ----------
    config:
        見 :class:`ModelConfig`。
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = (config or ModelConfig()).resolved()
        cfg = self.config

        self.sport_embedding = nn.Embedding(cfg.num_sports, cfg.sport_embedding)
        self.input_proj = nn.Conv1d(cfg.num_features, cfg.hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            ConditionedBlock(
                cfg.hidden, cfg.kernel_size, 2**i, cfg.sport_embedding, cfg.dropout
            )
            for i in range(cfg.num_layers)
        )
        self.head = nn.Conv1d(cfg.hidden, cfg.num_events, kernel_size=1)

    def forward(
        self,
        features: torch.Tensor,
        sport_ids: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features:
            ``(B, T, F)``。
        sport_ids:
            ``(B,)`` 的 long tensor，值為 :func:`events.sport_index` 的輸出。
        frame_mask:
            ``(B, T)`` 布林張量，``True`` 表示該影格有效（非 padding）。
            padding 位置的 logits 會被設成 ``-inf``，讓時間軸 softmax 完全忽略它們。

        Returns
        -------
        ``(B, E, T)`` 的 logits。
        """
        x = features.transpose(1, 2)  # (B, F, T)
        sport = self.sport_embedding(sport_ids)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, sport)
        logits = self.head(h)

        if frame_mask is not None:
            invalid = ~frame_mask.unsqueeze(1)  # (B, 1, T)
            logits = logits.masked_fill(invalid, float("-inf"))
        return logits

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def soft_targets(
    frames: torch.Tensor,
    length: int,
    sigma: torch.Tensor | float,
    frame_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """以真值影格為中心的離散高斯目標分布。

    用軟目標而非 one-hot：事件真值本身帶標註誤差（GolfDB 標註者對「擊球」的
    判定容忍度約 ±1 影格），硬目標會逼模型去擬合標註雜訊，也讓損失對「差一格」
    和「差五十格」給出同樣的懲罰。

    Parameters
    ----------
    frames:
        ``(B, E)`` 的真值影格索引。
    length:
        時間軸長度 ``T``。
    sigma:
        高斯標準差（影格）。可為純量或 ``(B, 1)``（依各片段 fps 縮放）。
    frame_mask:
        ``(B, T)``，``True`` 為有效影格。padding 位置的機率歸零後重新正規化。

    Returns
    -------
    ``(B, E, T)``，時間軸上總和為 1。
    """
    device = frames.device
    positions = torch.arange(length, device=device, dtype=torch.float32)
    centre = frames.unsqueeze(-1).to(torch.float32)  # (B, E, 1)
    sigma_t = torch.as_tensor(sigma, device=device, dtype=torch.float32)
    if sigma_t.ndim == 2:
        sigma_t = sigma_t.unsqueeze(-1)
    sigma_t = sigma_t.clamp(min=0.5)

    log_weights = -0.5 * ((positions - centre) / sigma_t) ** 2
    if frame_mask is not None:
        log_weights = log_weights.masked_fill(~frame_mask.unsqueeze(1), float("-inf"))
    return torch.softmax(log_weights, dim=-1)


def event_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    event_mask: torch.Tensor,
) -> torch.Tensor:
    """時間軸上的 KL 散度，只計算 active 的事件槽。

    Parameters
    ----------
    logits:
        ``(B, E, T)``，padding 位置應已為 ``-inf``。
    targets:
        ``(B, E, T)``，時間軸上總和為 1。
    event_mask:
        ``(B, E)`` 布林張量，``True`` 表示該片段有這個事件的標註。

    Returns
    -------
    純量：所有 active 事件槽的平均 KL。
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    # targets 為 0 的位置貢獻為 0；先乘再處理 -inf * 0 的 NaN
    per_event = -(targets * log_probs).nan_to_num(nan=0.0, neginf=0.0).sum(dim=-1)
    entropy = -(targets * torch.log(targets.clamp(min=1e-12))).sum(dim=-1)
    kl = per_event - entropy

    active = event_mask.to(kl.dtype)
    denom = active.sum().clamp(min=1.0)
    return (kl * active).sum() / denom
