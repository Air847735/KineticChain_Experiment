"""共用的測試素材。

全部合成，不需要 GPU、不需要資料集、不需要網路。
"""

from __future__ import annotations

import numpy as np
import pytest

from kinetic_chain.skeleton import CANONICAL_JOINTS, JOINT_INDEX, NUM_JOINTS


def synthetic_pose(
    num_frames: int = 120,
    *,
    scale: float = 100.0,
    offset: tuple[float, float] = (300.0, 200.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """造一段有明確近端到遠端時序的假動作。

    骨盆先轉、軀幹後轉、手腕最後加速，前腳在中段落地。這樣弱標註規則的輸出
    可以直接對照已知的真值，不必依賴任何外部資料。
    """
    rng = rng or np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, num_frames)

    def bump(centre: float, width: float) -> np.ndarray:
        return np.exp(-0.5 * ((t - centre) / width) ** 2)

    pelvis_angle = 1.2 * np.cumsum(bump(0.55, 0.05)) / num_frames
    torso_angle = 1.6 * np.cumsum(bump(0.62, 0.05)) / num_frames

    pose = np.zeros((num_frames, NUM_JOINTS, 3), dtype=np.float32)
    pose[..., 2] = 1.0

    hip_half = 0.18
    shoulder_half = 0.22
    for name in CANONICAL_JOINTS:
        index = JOINT_INDEX[name]
        if name.endswith("hip"):
            sign = -1.0 if name.startswith("left") else 1.0
            pose[:, index, 0] = sign * hip_half * np.cos(pelvis_angle)
            pose[:, index, 1] = sign * hip_half * np.sin(pelvis_angle)
        elif name.endswith("shoulder"):
            sign = -1.0 if name.startswith("left") else 1.0
            pose[:, index, 0] = sign * shoulder_half * np.cos(torso_angle)
            pose[:, index, 1] = -0.55 + sign * shoulder_half * np.sin(torso_angle)
        elif name == "head":
            pose[:, index, 1] = -0.85

    # 右腕：先後拉，再快速前擺；速度峰值落在 0.70
    swing = 0.9 * np.tanh(12.0 * (t - 0.70))
    pose[:, JOINT_INDEX["right_wrist"], 0] = swing
    pose[:, JOINT_INDEX["right_wrist"], 1] = -0.45 - 0.35 * bump(0.35, 0.10)
    pose[:, JOINT_INDEX["left_wrist"], 0] = 0.3 * swing
    pose[:, JOINT_INDEX["left_wrist"], 1] = -0.45
    for side, sign in (("left", -1.0), ("right", 1.0)):
        pose[:, JOINT_INDEX[f"{side}_elbow"], 0] = sign * 0.25
        pose[:, JOINT_INDEX[f"{side}_elbow"], 1] = -0.35
        pose[:, JOINT_INDEX[f"{side}_knee"], 0] = sign * 0.15
        pose[:, JOINT_INDEX[f"{side}_knee"], 1] = 0.45
        pose[:, JOINT_INDEX[f"{side}_ankle"], 0] = sign * 0.15
        pose[:, JOINT_INDEX[f"{side}_ankle"], 1] = 0.9

    # 前腳（右）抬起後於 0.45 落地
    lift = 0.5 * np.clip(1.0 - np.abs(t - 0.30) / 0.18, 0.0, None)
    pose[:, JOINT_INDEX["right_ankle"], 1] -= lift
    pose[:, JOINT_INDEX["right_ankle"], 0] += 0.6 * np.clip((t - 0.15) / 0.30, 0.0, 1.0)

    pose[..., :2] *= scale
    pose[..., 0] += offset[0]
    pose[..., 1] += offset[1]
    return pose


@pytest.fixture
def pose() -> np.ndarray:
    return synthetic_pose()


@pytest.fixture
def fps() -> float:
    return 30.0
