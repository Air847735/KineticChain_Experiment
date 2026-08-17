"""特徵的不變性。

特徵設計的整個前提是「與相機距離、畫面位置、慣用邊無關」。這裡直接測那個前提，
而不是測某個數值等於某個常數。
"""

from __future__ import annotations

import numpy as np
import pytest

from kinetic_chain.errors import ClipTooShortError
from kinetic_chain.features import (
    FEATURE_NAMES,
    NUM_FEATURES,
    build,
    compute,
    normalize,
)
from kinetic_chain.skeleton import JOINT_INDEX

from .conftest import synthetic_pose


def test_feature_names_match_matrix_width(pose, fps):
    matrix = build(pose, fps)
    assert matrix.shape == (pose.shape[0], NUM_FEATURES)
    assert len(FEATURE_NAMES) == NUM_FEATURES
    assert len(set(FEATURE_NAMES)) == NUM_FEATURES


def test_invariant_to_translation(pose, fps):
    shifted = pose.copy()
    shifted[..., 0] += 250.0
    shifted[..., 1] -= 130.0
    assert np.allclose(build(pose, fps), build(shifted, fps), atol=1e-4)


def test_invariant_to_scale(pose, fps):
    """相機拉遠一倍不該改變特徵——所有長度都以身體尺度為單位。"""
    scaled = pose.copy()
    scaled[..., :2] *= 2.5
    assert np.allclose(build(pose, fps), build(scaled, fps), atol=1e-3)


def test_handedness_normalisation_flips_reversed_motion(fps):
    """左右相反的同一個動作，正規化後特徵應該一致。"""
    pose = synthetic_pose()
    mirrored = pose.copy()
    mirrored[..., 0] = 1000.0 - mirrored[..., 0]
    for part in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
        li, ri = JOINT_INDEX[f"left_{part}"], JOINT_INDEX[f"right_{part}"]
        mirrored[:, [li, ri], :] = mirrored[:, [ri, li], :]

    original = compute(pose, fps, handedness_sensitive=True)
    flipped = compute(mirrored, fps, handedness_sensitive=True)
    assert original.flipped != flipped.flipped
    assert np.allclose(original.pose, flipped.pose, atol=1e-3)


def test_coverage_reflects_missing_keypoints(pose, fps):
    damaged = pose.copy()
    damaged[10:30, :, 2] = 0.0  # 20 影格完全偵測不到
    signals = compute(damaged, fps)
    assert signals.coverage == pytest.approx(1.0 - 20 / pose.shape[0], abs=1e-6)


def test_missing_keypoints_are_interpolated_not_zeroed(pose, fps):
    damaged = pose.copy()
    damaged[40:45, JOINT_INDEX["right_wrist"], 2] = 0.0
    coords, _, _, _, _ = normalize(damaged)
    # 內插後的位置應落在缺口前後兩端之間，而不是塌到原點
    before = coords[39, JOINT_INDEX["right_wrist"], 0]
    after = coords[45, JOINT_INDEX["right_wrist"], 0]
    gap = coords[40:45, JOINT_INDEX["right_wrist"], 0]
    assert np.all(gap >= min(before, after) - 1e-6)
    assert np.all(gap <= max(before, after) + 1e-6)


def test_features_are_finite_even_for_degenerate_input(fps):
    """所有關鍵點重疊在同一點時尺度會塌掉，仍不得產生 NaN 或 inf。"""
    pose = np.zeros((40, 13, 3), dtype=np.float32)
    pose[..., 2] = 1.0
    matrix = build(pose, fps)
    assert np.all(np.isfinite(matrix))


def test_too_short_clip_is_rejected(fps):
    pose = np.zeros((4, 13, 3), dtype=np.float32)
    pose[..., 2] = 1.0
    with pytest.raises(ClipTooShortError):
        compute(pose, fps)


def test_fps_out_of_range_falls_back_without_crashing(pose):
    assert np.all(np.isfinite(build(pose, 0.0)))
    assert np.all(np.isfinite(build(pose, float("nan"))))
