"""長影片切分的測試。

手上沒有真正的多次反覆影片，所以用合成序列驗證：拿一段合成的單次動作，
中間插入靜止段接成 N 次反覆，檢查切分器找回正確的段數與大致位置。
這證明得了演算法本身，證明不了它在真實訓練影片上的表現——後者見
`docs/architecture.md` 的 Known Gaps。
"""

from __future__ import annotations

import numpy as np
import pytest

from kinetic_chain.errors import KineticChainError
from kinetic_chain.features import compute
from kinetic_chain.segment import find_actions
from kinetic_chain.skeleton import NUM_JOINTS

FPS = 30.0


def _still_pose(num_frames: int) -> np.ndarray:
    """靜止站姿，關節座標不隨時間變化。"""
    base = np.array(
        [
            [0.0, 1.70],   # head
            [-0.20, 1.45], [0.20, 1.45],   # shoulders
            [-0.28, 1.15], [0.28, 1.15],   # elbows
            [-0.30, 0.90], [0.30, 0.90],   # wrists
            [-0.12, 0.95], [0.12, 0.95],   # hips
            [-0.13, 0.52], [0.13, 0.52],   # knees
            [-0.13, 0.05], [0.13, 0.05],   # ankles
        ],
        dtype=np.float32,
    )
    assert base.shape[0] == NUM_JOINTS
    pose = np.repeat(base[None, :, :], num_frames, axis=0)
    return np.concatenate([pose, np.ones((num_frames, NUM_JOINTS, 1), np.float32)], axis=2)


def _action_pose(num_frames: int) -> np.ndarray:
    """一次動作：雙腕由低到高再放下，其餘關節小幅跟隨。"""
    pose = _still_pose(num_frames)
    t = np.linspace(0.0, 1.0, num_frames)
    lift = np.sin(np.pi * t) ** 2  # 兩端為 0，中間最大
    for wrist in (5, 6):
        pose[:, wrist, 1] += 0.85 * lift
    for joint in (3, 4):
        pose[:, joint, 1] += 0.45 * lift
    for joint in (9, 10):
        pose[:, joint, 1] -= 0.10 * lift
    return pose


def _session(num_reps: int, action_frames: int = 45, rest_frames: int = 60) -> np.ndarray:
    """接成 N 次反覆，前後與中間都夾靜止段。"""
    parts = [_still_pose(rest_frames)]
    for _ in range(num_reps):
        parts.append(_action_pose(action_frames))
        parts.append(_still_pose(rest_frames))
    return np.concatenate(parts, axis=0)


@pytest.mark.parametrize("num_reps", [1, 2, 3, 5])
def test_finds_each_repetition(num_reps: int) -> None:
    signals = compute(_session(num_reps), FPS, handedness_sensitive=False)
    report = find_actions(signals, FPS, activity_signal="wrist_speed")
    assert len(report.segments) == num_reps
    assert report.should_trust


def test_segments_are_ordered_and_cover_the_action() -> None:
    action, rest = 45, 60
    signals = compute(_session(3, action, rest), FPS, handedness_sensitive=False)
    report = find_actions(signals, FPS, activity_signal="wrist_speed")

    for index, seg in enumerate(report.segments):
        expected_start = rest + index * (action + rest)
        expected_end = expected_start + action
        assert seg.start <= expected_start + action // 2 < seg.end
        # 峰值必須落在該次動作的範圍內。不檢查它靠不靠近中點——
        # 活動量是速度，sin² 位移的速度峰值在四分點而非中點，那是對的。
        assert expected_start <= seg.peak < expected_end
    starts = [s.start for s in report.segments]
    assert starts == sorted(starts)


def test_segments_do_not_overlap() -> None:
    signals = compute(_session(4), FPS, handedness_sensitive=False)
    report = find_actions(signals, FPS, activity_signal="wrist_speed")
    for earlier, later in zip(report.segments, report.segments[1:]):
        assert earlier.end <= later.start


def test_continuous_action_is_flagged_untrustworthy() -> None:
    """反覆之間不停下來時必須自己說不可信，而不是給出看似正確的邊界。

    用連續正弦而非把 `_action_pose` 接起來：後者兩端速度為零，接起來仍有間隔，
    切分器會（正確地）切開它，測不到要測的情況。
    """
    num_frames = 300
    pose = _still_pose(num_frames)
    t = np.arange(num_frames) / FPS
    wave = 0.45 * np.sin(2 * np.pi * 0.8 * t)
    for wrist in (5, 6):
        pose[:, wrist, 1] += wave
    signals = compute(pose, FPS, handedness_sensitive=False)
    report = find_actions(signals, FPS, activity_signal="wrist_speed")
    assert not report.should_trust
    assert "貼齊影片邊界" in report.reason


def test_action_running_to_the_video_edge_is_flagged() -> None:
    """影片從動作中途開始時，那一段的起點無從驗證，必須說出來。"""
    pose = np.concatenate([_action_pose(45)[20:], _still_pose(90)], axis=0)
    signals = compute(pose, FPS, handedness_sensitive=False)
    report = find_actions(signals, FPS, activity_signal="wrist_speed")
    assert not report.should_trust
    assert "貼齊影片邊界" in report.reason


def test_all_still_reports_no_segments() -> None:
    signals = compute(_still_pose(300), FPS, handedness_sensitive=False)
    report = find_actions(signals, FPS, activity_signal="wrist_speed")
    assert report.segments == ()
    assert not report.should_trust


def test_unknown_signal_raises() -> None:
    signals = compute(_session(2), FPS, handedness_sensitive=False)
    with pytest.raises(KineticChainError, match="未知的活動量訊號"):
        find_actions(signals, FPS, activity_signal="not_a_signal")


def test_too_short_raises() -> None:
    signals = compute(_still_pose(8), FPS, handedness_sensitive=False)
    with pytest.raises(KineticChainError, match="放不下"):
        find_actions(signals, FPS, activity_signal="wrist_speed")


def test_report_serialises() -> None:
    signals = compute(_session(2), FPS, handedness_sensitive=False)
    payload = find_actions(signals, FPS, activity_signal="wrist_speed").as_dict()
    assert payload["num_segments"] == 2
    assert len(payload["segments"]) == 2
    assert set(payload["segments"][0]) == {
        "start", "end", "frames", "peak", "peak_activity"
    }
