"""順序約束解碼。

模型對每個事件槽獨立輸出逐影格分數。獨立取 argmax 會產生力學上不可能的結果
（例如擊球早於上桿頂點）。這裡以動態規劃求出**滿足時序約束的全域最佳解**，
把順序變成硬保證而不是模型「大致上會學會」的傾向。

本模組沒有可學參數，純函式，可完整單元測試。
"""

from __future__ import annotations

import numpy as np

from .errors import ClipTooShortError

NEG_INF = -np.inf


def constrained_argmax(scores: np.ndarray) -> np.ndarray:
    """在 ``t_0 <= t_1 <= ... <= t_{k-1}`` 的限制下最大化 ``sum_i scores[i, t_i]``。

    Parameters
    ----------
    scores:
        ``(k, T)``，第 ``i`` 列是第 ``i`` 個事件（依時序排列）在各影格的對數分數。

    Returns
    -------
    ``(k,)`` 的 int64 影格索引，保證非遞減。

    Notes
    -----
    遞迴式 ``D[i][t] = scores[i][t] + max_{t' <= t} D[i-1][t']``。內層的
    ``max_{t' <= t}`` 以前綴最大值一次掃過，故每列 ``O(T)``，總計 ``O(kT)``
    時間與空間。

    允許相等（``<=`` 而非 ``<``）：像棒球投球的「手腕速度峰值」與「出手」在
    低幀率影片中可能落在同一影格，硬性要求嚴格遞增會逼出錯誤的解。
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"scores 的形狀應為 (k, T)，收到 {scores.shape}")
    k, n_frames = scores.shape
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    if n_frames == 0:
        raise ClipTooShortError("片段沒有任何影格")

    best = scores[0].copy()
    backpointers = np.empty((k, n_frames), dtype=np.int64)
    backpointers[0] = -1

    positions = np.arange(n_frames, dtype=np.int64)
    for i in range(1, k):
        prefix_max = np.maximum.accumulate(best)
        # best[t] == prefix_max[t] 的位置就是新的前綴最大值出現處；其餘填 0 後
        # 取累積最大值，等同於「沿用上一個達到前綴最大值的索引」。
        backpointers[i] = np.maximum.accumulate(
            np.where(best >= prefix_max, positions, 0)
        )
        best = scores[i] + prefix_max

    frames = np.empty(k, dtype=np.int64)
    frames[k - 1] = int(np.argmax(best))
    for i in range(k - 1, 0, -1):
        frames[i - 1] = backpointers[i][frames[i]]
    return frames


def decode(
    logits: np.ndarray,
    slots: tuple[int, ...],
    *,
    valid_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """把模型輸出解碼成事件影格與信心。

    Parameters
    ----------
    logits:
        ``(E, T)``，模型對全部事件槽的逐影格 logits。
    slots:
        該運動的事件槽索引，**依時序排列**（即 ``SportSpec.slots``）。
    valid_length:
        padding 前的真實影格數。``None`` 表示整段都有效。

    Returns
    -------
    ``(frames, confidence)``，長度皆為 ``len(slots)``。``confidence`` 是該事件
    在被選中影格上的機率（時間軸 softmax 後的值）。
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"logits 的形狀應為 (E, T)，收到 {logits.shape}")
    if valid_length is not None:
        logits = logits[:, :valid_length]
    if logits.shape[1] == 0:
        raise ClipTooShortError("片段沒有任何有效影格")

    selected = logits[list(slots), :]
    log_probs = selected - _logsumexp(selected, axis=1, keepdims=True)
    frames = constrained_argmax(log_probs)
    confidence = np.exp(log_probs[np.arange(len(slots)), frames])
    return frames, confidence


def _logsumexp(x: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    peak = np.max(x, axis=axis, keepdims=True)
    out = peak + np.log(np.sum(np.exp(x - peak), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)
