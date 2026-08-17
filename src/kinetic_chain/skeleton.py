"""關鍵點布局定義與跨布局對映。

不同來源的關鍵點編號不同（RTMPose 輸出 COCO-17，Penn Action 標註 13 點）。
特徵抽取只認識這裡定義的 *canonical* 13 點布局，其餘布局一律先轉換過來，
使 ``features.py`` 完全不需要知道資料從哪來。

Canonical 布局選 13 點而非 17 點的理由：COCO 的眼、耳對動力鏈沒有作用，
而 Penn Action 沒有這些點。取兩者的交集可以讓兩個資料集共用同一組特徵定義，
不必為了補齊而捏造不存在的關鍵點。
"""

from __future__ import annotations

import numpy as np

from .errors import SportSpecError

#: Canonical 關鍵點順序。索引即為 ``features`` 與 ``weak_labels`` 使用的編號。
CANONICAL_JOINTS: tuple[str, ...] = (
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

NUM_JOINTS = len(CANONICAL_JOINTS)

#: 名稱 → canonical 索引
JOINT_INDEX: dict[str, int] = {name: i for i, name in enumerate(CANONICAL_JOINTS)}

#: 左右鏡射時互換的 canonical 索引配對。
FLIP_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (JOINT_INDEX[f"left_{part}"], JOINT_INDEX[f"right_{part}"])
    for part in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")
)

#: COCO-17 的關鍵點順序（RTMPose 輸出）。
COCO17_JOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

#: Penn Action 標註的 13 點順序。
PENN13_JOINTS: tuple[str, ...] = (
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

#: 各布局的來源關鍵點名稱 → canonical 名稱。未列出的來源關鍵點會被丟棄。
_LAYOUT_SOURCES: dict[str, tuple[tuple[str, ...], dict[str, str]]] = {
    "coco17": (COCO17_JOINTS, {"nose": "head"}),
    "penn13": (PENN13_JOINTS, {}),
    "canonical13": (CANONICAL_JOINTS, {}),
}

LAYOUTS: tuple[str, ...] = tuple(_LAYOUT_SOURCES)


def _gather_indices(layout: str) -> np.ndarray:
    """回傳長度 13 的索引陣列，把來源布局重排成 canonical 順序。"""
    try:
        source_joints, aliases = _LAYOUT_SOURCES[layout]
    except KeyError as exc:
        raise SportSpecError(
            f"未知的關鍵點布局 {layout!r}；可用的有 {LAYOUTS}"
        ) from exc

    position = {}
    for i, name in enumerate(source_joints):
        position[aliases.get(name, name)] = i

    missing = [name for name in CANONICAL_JOINTS if name not in position]
    if missing:
        raise SportSpecError(
            f"布局 {layout!r} 缺少 canonical 關鍵點：{missing}"
        )
    return np.array([position[name] for name in CANONICAL_JOINTS], dtype=np.int64)


_GATHER_CACHE: dict[str, np.ndarray] = {}


def to_canonical(pose: np.ndarray, layout: str) -> np.ndarray:
    """把 ``(T, J, 3)`` 的關鍵點序列轉成 canonical 13 點布局。

    Parameters
    ----------
    pose:
        ``(T, J, 3)``，最後一維為 ``x, y, confidence``。
    layout:
        來源布局名稱，見 :data:`LAYOUTS`。

    Returns
    -------
    ``(T, 13, 3)`` 的 float32 陣列。
    """
    pose = np.asarray(pose)
    if pose.ndim != 3 or pose.shape[-1] != 3:
        raise SportSpecError(
            f"關鍵點序列的形狀應為 (T, J, 3)，收到 {pose.shape}"
        )

    if layout not in _GATHER_CACHE:
        _GATHER_CACHE[layout] = _gather_indices(layout)
    index = _GATHER_CACHE[layout]

    expected = len(_LAYOUT_SOURCES[layout][0])
    if pose.shape[1] != expected:
        raise SportSpecError(
            f"布局 {layout!r} 應有 {expected} 個關鍵點，收到 {pose.shape[1]}"
        )

    return np.ascontiguousarray(pose[:, index, :], dtype=np.float32)


def flip_horizontal(pose: np.ndarray) -> np.ndarray:
    """水平鏡射 canonical 姿態序列：x 取負並交換左右關鍵點。

    輸入須為已中心化的座標（原點在骨盆），否則 x 取負會把人移到畫面外。
    """
    flipped = np.array(pose, dtype=np.float32, copy=True)
    flipped[..., 0] = -flipped[..., 0]
    for left, right in FLIP_PAIRS:
        flipped[:, [left, right], :] = flipped[:, [right, left], :]
    return flipped
