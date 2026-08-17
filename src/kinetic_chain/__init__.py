"""KineticChain — 以運動項目為條件，從影片偵測動力鏈關鍵時間點。

一組權重涵蓋多個運動項目：運動項目的差異放在資料與條件輸入，不放在模型結構。

匯入本套件不會拉進 ``torch`` 以外的重量級相依。姿態抽取（``rtmlib``/``cv2``）
只在 :mod:`kinetic_chain.pose` 中發生，資料集轉接的 ``pandas``/``scipy.io``
只在實際載入時才匯入。
"""

from __future__ import annotations

from .data import Batch, Clip, ClipDataset, collate, split_clips
from .decode import decode
from .errors import (
    ClipTooShortError,
    DatasetError,
    KineticChainError,
    PoseExtractionError,
    SportSpecError,
    UnknownEventError,
    UnknownSportError,
    WeakLabelError,
)
from .events import (
    ALL_EVENTS,
    CANONICAL_EVENTS,
    NUM_EVENT_SLOTS,
    SPORT_SPECIFIC_EVENTS,
    SportSpec,
    get_sport,
    register_sport,
    registered_sports,
)
from .metrics import PCEReport, evaluate_predictions
from .model import KineticChainNet, ModelConfig, event_loss, soft_targets

__all__ = [
    "ALL_EVENTS",
    "CANONICAL_EVENTS",
    "NUM_EVENT_SLOTS",
    "SPORT_SPECIFIC_EVENTS",
    "Batch",
    "Clip",
    "ClipDataset",
    "ClipTooShortError",
    "DatasetError",
    "KineticChainError",
    "KineticChainNet",
    "ModelConfig",
    "PCEReport",
    "PoseExtractionError",
    "SportSpec",
    "SportSpecError",
    "UnknownEventError",
    "UnknownSportError",
    "WeakLabelError",
    "collate",
    "decode",
    "evaluate_predictions",
    "event_loss",
    "get_sport",
    "register_sport",
    "registered_sports",
    "soft_targets",
    "split_clips",
]

__version__ = "0.1.0"
