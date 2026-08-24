"""姿態序列 → 力學特徵。

本模組是「訊號」的唯一來源：模型吃的特徵矩陣與弱標註規則吃的一維訊號，
都從同一次計算出來。這樣規則與模型看到的是同一個世界，弱標註不會因為
兩邊各算各的而對不上。

座標約定：正規化之後原點在骨盆中點，**y 軸向上為正**（與影像座標相反），
單位為「肩寬與髖寬的平均」。因此所有特徵與相機距離、畫面位置無關。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from .errors import ClipTooShortError
from .skeleton import JOINT_INDEX, NUM_JOINTS, flip_horizontal

MIN_FRAMES = 8

_L_SHOULDER = JOINT_INDEX["left_shoulder"]
_R_SHOULDER = JOINT_INDEX["right_shoulder"]
_L_HIP = JOINT_INDEX["left_hip"]
_R_HIP = JOINT_INDEX["right_hip"]
_L_WRIST = JOINT_INDEX["left_wrist"]
_R_WRIST = JOINT_INDEX["right_wrist"]
_L_KNEE = JOINT_INDEX["left_knee"]
_R_KNEE = JOINT_INDEX["right_knee"]
_L_ANKLE = JOINT_INDEX["left_ankle"]
_R_ANKLE = JOINT_INDEX["right_ankle"]

#: 關鍵點信心低於此值時視為未量到，沿時間軸內插補值。
#: 匯出成常數是為了讓視覺化能標出哪些關節是補出來的，不必重複這個字面值。
MIN_CONFIDENCE = 0.3

#: 特徵矩陣的欄位名稱，順序即欄位順序。
FEATURE_NAMES: tuple[str, ...] = (
    *(
        f"{axis}_{joint}"
        for joint in (
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
        for axis in ("x", "y")
    ),
    "sin_pelvis_angle",
    "cos_pelvis_angle",
    "sin_torso_angle",
    "cos_torso_angle",
    "sin_separation_angle",
    "cos_separation_angle",
    "pelvis_angular_speed",
    "torso_angular_speed",
    "separation_rate",
    *(
        f"speed_{joint}"
        for joint in (
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
    ),
    "wrist_height",
    "lead_knee_height",
    "lead_ankle_height",
    "lead_ankle_vspeed",
    "body_speed",
    "cos_hip_angle",
    "cos_knee_angle",
    "hip_extension_speed",
    "knee_extension_speed",
    "pose_confidence",
)

NUM_FEATURES = len(FEATURE_NAMES)


@dataclass(frozen=True)
class PoseSignals:
    """正規化後的姿態與由它導出的一維力學訊號。

    Attributes
    ----------
    pose:
        ``(T, 13, 2)``，已中心化、縮放、必要時鏡射，y 軸向上為正。
    signals:
        訊號名稱 → ``(T,)`` 陣列。弱標註規則只認得這裡的名稱。
    scale:
        用來正規化的身體尺度（原始影像單位）。
    flipped:
        是否做過左右鏡射。
    hitting_side:
        判定為「擊球／出手側」的邊，``"left"`` 或 ``"right"``（鏡射前的原始側）。
    coverage:
        關鍵點信心足夠的影格比例。
    """

    pose: np.ndarray
    signals: dict[str, np.ndarray]
    scale: float
    flipped: bool
    hitting_side: str
    coverage: float


def _interpolate_low_confidence(
    pose: np.ndarray, min_confidence: float
) -> tuple[np.ndarray, float]:
    """把信心不足的關鍵點沿時間軸線性內插補值。

    回傳補值後的 ``(T, J, 2)`` 座標與整段的有效影格比例。頭尾缺值以最近的
    有效影格外推（``np.interp`` 的預設行為），全段皆缺的關鍵點填 0。
    """
    coords = np.array(pose[..., :2], dtype=np.float64, copy=True)
    conf = np.asarray(pose[..., 2], dtype=np.float64)
    valid = conf >= min_confidence
    frames = np.arange(coords.shape[0], dtype=np.float64)

    for j in range(coords.shape[1]):
        good = valid[:, j]
        if good.all():
            continue
        if not good.any():
            coords[:, j, :] = 0.0
            continue
        for axis in range(2):
            coords[:, j, axis] = np.interp(
                frames, frames[good], coords[good, j, axis]
            )

    # 一個影格只要有一半以上的關鍵點可信就算有效
    coverage = float(np.mean(valid.mean(axis=1) >= 0.5))
    return coords, coverage


def _smooth(x: np.ndarray, fps: float) -> np.ndarray:
    """沿時間軸做 Savitzky–Golay 平滑；窗長依 fps 縮放。

    用 SG 而非移動平均：SG 以局部多項式擬合，能保留峰值的位置與高度，
    而峰值位置正是本專案要找的東西。移動平均會把峰值壓平並可能位移。
    """
    n = x.shape[0]
    window = int(round(0.1 * fps)) | 1  # 約 100 ms，強制為奇數
    window = max(5, min(window, n if n % 2 else n - 1))
    if window < 5 or n < 5:
        return x.astype(np.float64, copy=False)
    return savgol_filter(x, window_length=window, polyorder=2, axis=0)


def _velocity(x: np.ndarray, fps: float) -> np.ndarray:
    """中央差分求導，單位為「每秒」。"""
    if x.shape[0] < 2:
        return np.zeros_like(x)
    return np.gradient(x, axis=0) * fps


def normalize(
    pose: np.ndarray,
    *,
    handedness_sensitive: bool = True,
    min_confidence: float = MIN_CONFIDENCE,
) -> tuple[np.ndarray, float, bool, str, float]:
    """把 canonical 關鍵點序列正規化成尺度、位置與慣用邊無關的座標。

    Returns
    -------
    ``(pose_norm, scale, flipped, hitting_side, coverage)``
    """
    coords, coverage = _interpolate_low_confidence(pose, min_confidence)

    pelvis = 0.5 * (coords[:, _L_HIP, :] + coords[:, _R_HIP, :])
    centred = coords - pelvis[:, None, :]

    shoulder_width = np.linalg.norm(
        centred[:, _L_SHOULDER, :] - centred[:, _R_SHOULDER, :], axis=-1
    )
    hip_width = np.linalg.norm(centred[:, _L_HIP, :] - centred[:, _R_HIP, :], axis=-1)
    torso_height = np.linalg.norm(
        0.5 * (centred[:, _L_SHOULDER, :] + centred[:, _R_SHOULDER, :]), axis=-1
    )
    # 肩寬與髖寬在正面視角有意義，側面視角會塌成 0；軀幹長度則在兩種視角都存在。
    # 取三者的中位數合成，避免視角造成尺度爆炸。
    scale = float(
        np.median(np.stack([shoulder_width, hip_width, torso_height]).mean(axis=0))
    )
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0

    normed = centred / scale
    normed[..., 1] *= -1.0  # 影像 y 向下 → 改為向上為正

    # 擊球／出手側：整段速度峰值較大的手腕
    speeds = np.linalg.norm(_velocity(normed, 1.0), axis=-1)
    hitting_side = "left" if speeds[:, _L_WRIST].max() > speeds[:, _R_WRIST].max() else "right"

    flipped = False
    if handedness_sensitive:
        wrist = _L_WRIST if hitting_side == "left" else _R_WRIST
        x = normed[:, wrist, 0]
        t = np.arange(x.size, dtype=np.float64)
        # 以線性擬合的斜率判斷水平運動方向，比首尾差值抗雜訊
        slope = np.polyfit(t, x, 1)[0] if x.size >= 2 else 0.0
        if slope < 0:
            normed = flip_horizontal(
                np.concatenate([normed, np.zeros_like(normed[..., :1])], axis=-1)
            )[..., :2]
            flipped = True

    return normed.astype(np.float64), scale, flipped, hitting_side, coverage


def _angle_series(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """向量 ``b - a`` 的方向角，沿時間軸解纏繞。

    警告：這個量在 ``b - a`` 的投影長度趨近零時是病態的——輸入的小誤差被放大成
    輸出的大誤差。髖線與肩線在 2D 投影下會隨身體轉向鏡頭而縮短，因此由此導出的
    角速度不可全信。用 :func:`kinetic_chain.analysis.projection_quality` 診斷。
    """
    d = b - a
    return np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """以 ``b`` 為頂點的三點夾角（弧度，範圍 ``[0, pi]``）。

    與 :func:`_angle_series` 的關鍵差別：夾角的兩條邊是**長節段**（軀幹、大腿、小腿），
    不會像跨身體的髖線那樣在投影下塌成一點，因此對關鍵點雜訊穩健得多。
    伸展型動作（舉重、跳躍）的動力鏈就定義在這種夾角上。

    但它有另一個限制：三點必須落在**看得見的平面**上。正面拍攝時前後彎曲被壓縮，
    髖角量不出來——這是矢狀面版本的同一個投影問題。
    """
    u = a - b
    v = c - b
    norms = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
    cosine = (u * v).sum(axis=-1) / np.maximum(norms, 1e-9)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def compute(
    pose: np.ndarray,
    fps: float,
    *,
    handedness_sensitive: bool = True,
    min_confidence: float = MIN_CONFIDENCE,
) -> PoseSignals:
    """由 canonical 關鍵點序列算出正規化姿態與全部一維訊號。

    Parameters
    ----------
    pose:
        ``(T, 13, 3)``，canonical 布局，最後一維為 ``x, y, confidence``。
    fps:
        影格率。用於把速度換算成「每秒」並決定平滑窗長，使不同幀率的影片可比。
    """
    pose = np.asarray(pose)
    if pose.ndim != 3 or pose.shape[1] != NUM_JOINTS or pose.shape[2] != 3:
        raise ValueError(
            f"姿態序列的形狀應為 (T, {NUM_JOINTS}, 3)，收到 {pose.shape}"
        )
    if pose.shape[0] < MIN_FRAMES:
        raise ClipTooShortError(
            f"片段只有 {pose.shape[0]} 影格，少於最低要求 {MIN_FRAMES}"
        )
    if not np.isfinite(fps) or not 1.0 <= fps <= 1000.0:
        fps = 30.0

    normed, scale, flipped, hitting_side, coverage = normalize(
        pose, handedness_sensitive=handedness_sensitive, min_confidence=min_confidence
    )
    smoothed = _smooth(normed, fps)
    velocity = _velocity(smoothed, fps)
    joint_speed = np.linalg.norm(velocity, axis=-1)

    pelvis_angle = _angle_series(smoothed[:, _L_HIP, :], smoothed[:, _R_HIP, :])
    torso_angle = _angle_series(smoothed[:, _L_SHOULDER, :], smoothed[:, _R_SHOULDER, :])
    separation = torso_angle - pelvis_angle

    # 鏡射後擊球手一律視為 "right" 側；未鏡射時沿用原判定
    wrist = _R_WRIST if (flipped or hitting_side == "right") else _L_WRIST
    lead_knee = _L_KNEE if smoothed[:, _L_KNEE, 0].mean() > smoothed[:, _R_KNEE, 0].mean() else _R_KNEE
    lead_ankle = _L_ANKLE if smoothed[:, _L_ANKLE, 0].mean() > smoothed[:, _R_ANKLE, 0].mean() else _R_ANKLE

    # 伸展型動力鏈：以三點夾角量髖與膝，取雙側平均（舉重等對稱動作最適用；
    # 非對稱動作取平均仍有意義，但不區分前後腳）。
    shoulder_mid = 0.5 * (smoothed[:, _L_SHOULDER, :] + smoothed[:, _R_SHOULDER, :])
    pelvis_origin = np.zeros_like(shoulder_mid)
    knee_mid = 0.5 * (smoothed[:, _L_KNEE, :] + smoothed[:, _R_KNEE, :])
    ankle_mid = 0.5 * (smoothed[:, _L_ANKLE, :] + smoothed[:, _R_ANKLE, :])
    hip_angle = _joint_angle(shoulder_mid, pelvis_origin, knee_mid)
    knee_angle = _joint_angle(pelvis_origin, knee_mid, ankle_mid)

    signals: dict[str, np.ndarray] = {
        "hip_angle": hip_angle,
        "knee_angle": knee_angle,
        # 只取**伸展**方向（夾角變大）。用絕對值會把接槓下蹲的快速屈曲算進來，
        # 而那正好是三重伸展的反向動作，混在一起會讓序列量測完全失真。
        "hip_extension_speed": np.maximum(_velocity(hip_angle, fps), 0.0),
        "knee_extension_speed": np.maximum(_velocity(knee_angle, fps), 0.0),
        "pelvis_angle": pelvis_angle,
        "torso_angle": torso_angle,
        "separation_angle": separation,
        "pelvis_angular_speed": np.abs(_velocity(pelvis_angle, fps)),
        "torso_angular_speed": np.abs(_velocity(torso_angle, fps)),
        "separation_rate": _velocity(separation, fps),
        "wrist_speed": joint_speed[:, wrist],
        "wrist_height": smoothed[:, wrist, 1],
        "lead_knee_height": smoothed[:, lead_knee, 1],
        "lead_ankle_height": smoothed[:, lead_ankle, 1],
        "lead_ankle_vspeed": velocity[:, lead_ankle, 1],
        "body_speed": joint_speed.mean(axis=1),
    }

    return PoseSignals(
        pose=smoothed,
        signals=signals,
        scale=scale,
        flipped=flipped,
        hitting_side=hitting_side,
        coverage=coverage,
    )


def build(
    pose: np.ndarray,
    fps: float,
    *,
    handedness_sensitive: bool = True,
    min_confidence: float = 0.3,
    signals: PoseSignals | None = None,
) -> np.ndarray:
    """組出模型輸入的特徵矩陣 ``(T, NUM_FEATURES)``。

    ``signals`` 可傳入已算好的結果避免重算（訓練時特徵與弱標註共用同一份）。
    """
    if signals is None:
        signals = compute(
            pose,
            fps,
            handedness_sensitive=handedness_sensitive,
            min_confidence=min_confidence,
        )
    s = signals.signals
    conf = np.asarray(pose[..., 2], dtype=np.float64).mean(axis=1)

    columns = [
        signals.pose.reshape(signals.pose.shape[0], -1),
        np.stack(
            [
                np.sin(s["pelvis_angle"]),
                np.cos(s["pelvis_angle"]),
                np.sin(s["torso_angle"]),
                np.cos(s["torso_angle"]),
                np.sin(s["separation_angle"]),
                np.cos(s["separation_angle"]),
                s["pelvis_angular_speed"],
                s["torso_angular_speed"],
                s["separation_rate"],
            ],
            axis=1,
        ),
        np.linalg.norm(_velocity(signals.pose, fps), axis=-1),
        np.stack(
            [
                s["wrist_height"],
                s["lead_knee_height"],
                s["lead_ankle_height"],
                s["lead_ankle_vspeed"],
                s["body_speed"],
                np.cos(s["hip_angle"]),
                np.cos(s["knee_angle"]),
                s["hip_extension_speed"],
                s["knee_extension_speed"],
                conf,
            ],
            axis=1,
        ),
    ]
    matrix = np.concatenate(columns, axis=1).astype(np.float32)
    if matrix.shape[1] != NUM_FEATURES:
        raise AssertionError(
            f"特徵欄位數 {matrix.shape[1]} 與 FEATURE_NAMES 的 {NUM_FEATURES} 不一致"
        )
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
