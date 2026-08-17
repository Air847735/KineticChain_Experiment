"""順序約束解碼的正確性。

這是整個管線裡唯一以硬保證形式存在的不變式（架構文件的 I1），因此測得最細：
不只測「通常正確」，而是對小規模輸入以窮舉法比對全域最佳解。
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from kinetic_chain.decode import constrained_argmax, decode
from kinetic_chain.errors import ClipTooShortError


def brute_force(scores: np.ndarray) -> float:
    """窮舉所有非遞減的組合，回傳最佳總分。只用於小規模驗證。"""
    k, n = scores.shape
    best = -np.inf
    for combo in itertools.combinations_with_replacement(range(n), k):
        best = max(best, sum(scores[i, t] for i, t in enumerate(combo)))
    return best


def test_matches_brute_force_on_random_inputs():
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = int(rng.integers(1, 5))
        n = int(rng.integers(k, 9))
        scores = rng.normal(size=(k, n))
        frames = constrained_argmax(scores)
        total = sum(scores[i, t] for i, t in enumerate(frames))
        assert total == pytest.approx(brute_force(scores))


def test_output_is_always_non_decreasing():
    rng = np.random.default_rng(1)
    for _ in range(200):
        scores = rng.normal(size=(int(rng.integers(2, 8)), int(rng.integers(8, 40))))
        frames = constrained_argmax(scores)
        assert np.all(np.diff(frames) >= 0)


def test_independent_argmax_can_violate_order_but_decoder_cannot():
    # 第 0 個事件的獨立最佳解在最後一格，第 1 個在第一格——順序完全相反
    scores = np.array(
        [
            [0.0, 0.0, 0.0, 5.0],
            [4.0, 0.0, 0.0, 0.0],
        ]
    )
    assert list(np.argmax(scores, axis=1)) == [3, 0]  # 獨立取 argmax 違反順序
    frames = constrained_argmax(scores)
    assert frames[0] <= frames[1]


def test_allows_equal_frames():
    """兩個事件落在同一影格是合法的（低幀率下峰值與釋放可能同格）。"""
    scores = np.array([[0.0, 9.0, 0.0], [0.0, 9.0, 0.0]])
    assert list(constrained_argmax(scores)) == [1, 1]


def test_single_event_reduces_to_argmax():
    scores = np.array([[1.0, 7.0, 3.0]])
    assert list(constrained_argmax(scores)) == [1]


def test_empty_event_set_returns_empty():
    assert constrained_argmax(np.zeros((0, 5))).size == 0


def test_zero_frames_is_an_error():
    with pytest.raises(ClipTooShortError):
        constrained_argmax(np.zeros((2, 0)))


def test_decode_selects_declared_slots_and_normalises_confidence():
    logits = np.full((13, 20), -10.0)
    logits[3, 5] = 10.0   # slot 3 峰值在影格 5
    logits[7, 12] = 10.0  # slot 7 峰值在影格 12
    frames, confidence = decode(logits, slots=(3, 7))
    assert list(frames) == [5, 12]
    assert np.all(confidence > 0.9)


def test_decode_respects_valid_length():
    """padding 區域即使分數最高也不能被選中。"""
    logits = np.full((13, 30), -10.0)
    logits[0, 25] = 50.0  # 落在 padding 內
    logits[0, 4] = 1.0
    frames, _ = decode(logits, slots=(0,), valid_length=10)
    assert frames[0] == 4
