"""統一的片段記錄與批次組裝。

所有資料集轉接層的輸出都是 :class:`Clip`，模型與訓練迴圈只認識 :class:`Clip`。
新增資料集不需要動訓練程式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np
import torch

from .errors import ClipTooShortError, DatasetError
from .events import NUM_EVENT_SLOTS, SportSpec, event_index, get_sport, sport_index
from .features import MIN_FRAMES, NUM_FEATURES, PoseSignals, build, compute

LabelSource = Literal["human", "weak"]


@dataclass
class Clip:
    """一段已裁切的單人動作片段。

    Attributes
    ----------
    clip_id:
        來源資料集內的唯一 id。
    sport:
        運動項目 id，須已註冊。
    pose:
        ``(T, 13, 3)`` canonical 布局的關鍵點，最後一維為 ``x, y, confidence``。
    fps:
        影格率。
    events:
        事件 id → 影格索引。可以是該運動宣告事件的子集。
    label_source:
        ``"human"``（真人標註）或 ``"weak"``（規則推導）。**必填**，因為評估報表
        必須據此分開統計，合併會讓兩種完全不同的東西看起來一樣。
    dataset:
        來源資料集名稱，用於報表分組。
    coverage:
        關鍵點信心足夠的影格比例，由特徵計算時填入。
    fold:
        資料集自帶的官方切分編號（GolfDB 有 1–4 四折）。有官方切分時要照用，
        隨機切分會讓數字無法與已發表結果對照。``None`` 表示該資料集沒有官方切分。
    """

    clip_id: str
    sport: str
    pose: np.ndarray
    fps: float
    events: dict[str, int]
    label_source: LabelSource
    dataset: str = ""
    coverage: float = 1.0
    fold: int | None = None
    _signals: PoseSignals | None = field(default=None, repr=False, compare=False)
    _features: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.label_source not in ("human", "weak"):
            raise DatasetError(
                f"{self.clip_id!r} 的 label_source 必須是 'human' 或 'weak'，"
                f"收到 {self.label_source!r}"
            )
        self.spec  # 提早驗證運動項目已註冊
        for name in self.events:
            event_index(name)
        if self.num_frames < MIN_FRAMES:
            raise ClipTooShortError(
                f"{self.clip_id!r} 只有 {self.num_frames} 影格，少於最低要求 {MIN_FRAMES}"
            )

    @property
    def spec(self) -> SportSpec:
        return get_sport(self.sport)

    @property
    def num_frames(self) -> int:
        """片段的時間長度。

        以已快取的特徵為準，而不是原始姿態：時間軸的資料增強會重取樣特徵，
        使兩者長度不同。事件影格索引指的是特徵序列上的位置，padding 與解碼也是，
        所以特徵長度才是權威。
        """
        if self._features is not None:
            return int(self._features.shape[0])
        return int(self.pose.shape[0])

    @property
    def ordered_events(self) -> tuple[str, ...]:
        """該片段實際具備的事件，依該運動宣告的時序排列。"""
        return tuple(e for e in self.spec.events if e in self.events)

    def signals(self) -> PoseSignals:
        """算一次、快取起來——特徵與弱標註共用同一份訊號。"""
        if self._signals is None:
            self._signals = compute(
                self.pose,
                self.fps,
                handedness_sensitive=self.spec.handedness_sensitive,
            )
            self.coverage = self._signals.coverage
        return self._signals

    def features(self) -> np.ndarray:
        if self._features is None:
            self._features = build(self.pose, self.fps, signals=self.signals())
        return self._features

    def release_cache(self) -> None:
        """丟掉快取的中間結果，減少長時間持有的記憶體。"""
        self._signals = None
        self._features = None


@dataclass
class Batch:
    """組裝好的一批資料。所有張量的批次維度都在最前面。"""

    features: torch.Tensor      # (B, T, F) float32
    sport_ids: torch.Tensor     # (B,) long
    frame_mask: torch.Tensor    # (B, T) bool，True 為有效影格
    event_mask: torch.Tensor    # (B, E) bool，True 為該片段有標註的事件槽
    targets: torch.Tensor       # (B, E) long，事件影格索引；未標註處為 0
    sigma: torch.Tensor         # (B, 1) float32，軟目標的標準差（影格）
    clip_ids: tuple[str, ...]

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            features=self.features.to(device),
            sport_ids=self.sport_ids.to(device),
            frame_mask=self.frame_mask.to(device),
            event_mask=self.event_mask.to(device),
            targets=self.targets.to(device),
            sigma=self.sigma.to(device),
            clip_ids=self.clip_ids,
        )


def collate(clips: Sequence[Clip], *, sigma_seconds: float = 0.05) -> Batch:
    """把不等長的片段補齊成一批。

    padding 用 0 填特徵，並在 ``frame_mask`` 標成無效；模型會把這些位置的 logits
    設成 ``-inf``，因此不會被時間軸 softmax 取到。

    ``sigma_seconds`` 以秒為單位再換算成影格：不同 fps 的片段在時間上得到一致的
    軟化程度，而不是在影格數上一致。
    """
    if not clips:
        raise DatasetError("collate 收到空的片段列表")

    max_frames = max(clip.num_frames for clip in clips)
    batch_size = len(clips)

    features = torch.zeros(batch_size, max_frames, NUM_FEATURES, dtype=torch.float32)
    frame_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    event_mask = torch.zeros(batch_size, NUM_EVENT_SLOTS, dtype=torch.bool)
    targets = torch.zeros(batch_size, NUM_EVENT_SLOTS, dtype=torch.long)
    sport_ids = torch.zeros(batch_size, dtype=torch.long)
    sigma = torch.zeros(batch_size, 1, dtype=torch.float32)

    for i, clip in enumerate(clips):
        matrix = clip.features()
        n = matrix.shape[0]
        features[i, :n] = torch.from_numpy(matrix)
        frame_mask[i, :n] = True
        sport_ids[i] = sport_index(clip.sport)
        sigma[i, 0] = max(0.5, sigma_seconds * clip.fps)
        for name in clip.ordered_events:
            slot = event_index(name)
            frame = int(clip.events[name])
            if not 0 <= frame < n:
                raise DatasetError(
                    f"{clip.clip_id!r} 的事件 {name!r} 落在影格 {frame}，"
                    f"超出片段長度 {n}"
                )
            event_mask[i, slot] = True
            targets[i, slot] = frame

    return Batch(
        features=features,
        sport_ids=sport_ids,
        frame_mask=frame_mask,
        event_mask=event_mask,
        targets=targets,
        sigma=sigma,
        clip_ids=tuple(clip.clip_id for clip in clips),
    )


class ClipDataset(torch.utils.data.Dataset):
    """把 :class:`Clip` 列表包成 torch Dataset。特徵在第一次取用時計算並快取。"""

    def __init__(self, clips: Sequence[Clip]) -> None:
        self.clips = list(clips)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> Clip:
        clip = self.clips[index]
        clip.features()  # 觸發快取
        return clip


def split_clips(
    clips: Iterable[Clip],
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
    val_fold: int | None = None,
) -> tuple[list[Clip], list[Clip]]:
    """依運動項目分層切分訓練／驗證集。

    分層而非全域隨機：資料集之間片段數差距很大（GolfDB 約 1400 段，Penn Action
    的單一運動只有一兩百段），全域隨機切會讓小類別的驗證集小到沒有意義。

    ``val_fold`` 指定時，帶有官方切分編號的片段（``Clip.fold``）一律照官方切分走，
    只有 ``fold`` 為 ``None`` 的片段才隨機切。這是與已發表數字對照的前提——
    自訂的隨機切分無法跟別人的四折交叉驗證比。
    """
    clips = list(clips)
    train: list[Clip] = []
    val: list[Clip] = []

    if val_fold is not None:
        folded = [c for c in clips if c.fold is not None]
        clips = [c for c in clips if c.fold is None]
        train.extend(c for c in folded if c.fold != val_fold)
        val.extend(c for c in folded if c.fold == val_fold)

    by_sport: dict[str, list[Clip]] = {}
    for clip in clips:
        by_sport.setdefault(clip.sport, []).append(clip)

    rng = np.random.default_rng(seed)
    for sport in sorted(by_sport):
        group = sorted(by_sport[sport], key=lambda c: c.clip_id)
        order = rng.permutation(len(group))
        cut = int(round(len(group) * (1.0 - val_fraction)))
        cut = min(max(cut, 1), len(group) - 1) if len(group) > 1 else len(group)
        train.extend(group[i] for i in order[:cut])
        val.extend(group[i] for i in order[cut:])
    return train, val
