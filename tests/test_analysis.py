"""動力鏈時序分析。

重點在兩件事：區段換算正確，以及**無約束量測真的不受順序假設影響**——
後者是整個分析誠實與否的關鍵，若它偷偷套了順序，近端到遠端的結論就變成循環論證。
"""

from __future__ import annotations

import numpy as np
import pytest

from kinetic_chain.analysis import (
    CHAIN_LINKS,
    SEGMENTS,
    analyse,
    sequence_rate,
    summarise,
    unconstrained_sequence,
)
from kinetic_chain.events import CANONICAL_EVENTS, get_sport
from kinetic_chain.features import compute

from .conftest import synthetic_pose

SPEC = get_sport("baseball_pitch")

EVENTS = {
    "address": 0,
    "loading_start": 10,
    "loading_peak": 30,
    "stride_foot_contact": 45,
    "pelvis_peak_rotation": 51,
    "torso_peak_rotation": 54,
    "arm_peak_velocity": 57,
    "release_impact": 60,
    "follow_through_mid": 75,
    "finish": 90,
}


def test_segments_use_frame_differences():
    result = analyse("t/0", SPEC, EVENTS, fps=30.0)
    assert result.segment("骨盆→軀幹").frames == 3
    assert result.segment("軀幹→上肢").frames == 3
    assert result.segment("加速期").frames == 15
    assert result.segment("蓄力").frames == 20


def test_seconds_scale_with_fps():
    at30 = analyse("t/0", SPEC, EVENTS, fps=30.0).segment("加速期")
    at60 = analyse("t/0", SPEC, EVENTS, fps=60.0).segment("加速期")
    assert at30.frames == at60.frames
    assert at30.seconds == pytest.approx(2 * at60.seconds)


def test_percent_normalises_to_address_release_span():
    """參考區間是最早事件到 release_impact，不含隨勢——隨勢長度受裁切影響。"""
    result = analyse("t/0", SPEC, EVENTS, fps=30.0)
    assert result.throw_frames == 60
    assert result.segment("加速期").percent_of_throw == pytest.approx(100 * 15 / 60)
    # 隨勢比 100% 長是合理的：它落在參考區間之外
    assert result.segment("隨勢").percent_of_throw > 0


def test_timeline_places_release_at_100_percent():
    result = analyse("t/0", SPEC, EVENTS, fps=30.0)
    assert result.timeline["address"] == pytest.approx(0.0)
    assert result.timeline["release_impact"] == pytest.approx(100.0)


def test_missing_events_are_skipped_not_estimated():
    partial = {k: v for k, v in EVENTS.items() if k != "stride_foot_contact"}
    result = analyse("t/0", SPEC, partial, fps=30.0)
    assert result.segment("跨步") is None
    assert result.segment("加速期") is None
    assert result.segment("骨盆→軀幹") is not None


def test_sequence_detects_violation():
    scrambled = dict(EVENTS)
    scrambled["pelvis_peak_rotation"] = 58  # 骨盆跑到上肢之後
    result = analyse("t/0", SPEC, scrambled, fps=30.0)
    assert not result.sequence_ok
    assert result.sequence == (
        "torso_peak_rotation",
        "arm_peak_velocity",
        "pelvis_peak_rotation",
    )


def test_sequence_ok_for_proximal_to_distal():
    assert analyse("t/0", SPEC, EVENTS, fps=30.0).sequence_ok


# --------------------------------------------------------------------------
# 無約束量測：分析誠實性的關鍵
# --------------------------------------------------------------------------


def test_unconstrained_measurement_can_report_a_reversed_order():
    """把訊號做成遠端先達峰，量測必須如實回報順序相反，不得偷偷排序。"""
    signals = compute(synthetic_pose(120), 30.0)
    fake = dict(signals.signals)
    n = signals.pose.shape[0]
    # 手腕最早、軀幹次之、骨盆最晚——完全顛倒
    for signal_name, peak in (
        ("wrist_speed", 20),
        ("torso_angular_speed", 50),
        ("pelvis_angular_speed", 80),
    ):
        curve = np.zeros(n)
        curve[peak] = 1.0
        fake[signal_name] = curve
    object.__setattr__(signals, "signals", fake)

    peaks = unconstrained_sequence(signals)
    assert peaks["arm_peak_velocity"] == 20
    assert peaks["torso_peak_rotation"] == 50
    assert peaks["pelvis_peak_rotation"] == 80
    order = sorted(peaks, key=lambda e: peaks[e])
    assert order[0] == "arm_peak_velocity"  # 如實回報，沒有被重排


def test_unconstrained_window_restricts_the_search():
    signals = compute(synthetic_pose(120), 30.0)
    fake = dict(signals.signals)
    n = signals.pose.shape[0]
    curve = np.zeros(n)
    curve[10] = 2.0   # 全域最大值在窗外
    curve[70] = 1.0   # 窗內的最大值
    fake["wrist_speed"] = curve
    object.__setattr__(signals, "signals", fake)

    assert unconstrained_sequence(signals)["arm_peak_velocity"] == 10
    assert unconstrained_sequence(signals, window=(60, 100))["arm_peak_velocity"] == 70


def test_unconstrained_window_is_clamped_to_the_clip():
    signals = compute(synthetic_pose(60), 30.0)
    peaks = unconstrained_sequence(signals, window=(-50, 9999))
    assert all(0 <= v < signals.pose.shape[0] for v in peaks.values())


# --------------------------------------------------------------------------
# 統計
# --------------------------------------------------------------------------


def test_summarise_uses_median_not_mean():
    """單機 2D 姿態偶爾有離群估計，平均會被少數壞樣本拉走。"""
    normal = [analyse(f"c{i}", SPEC, EVENTS, fps=30.0) for i in range(9)]
    outlier_events = dict(EVENTS)
    outlier_events["torso_peak_rotation"] = 500
    outlier_events["arm_peak_velocity"] = 501
    outlier_events["release_impact"] = 502
    outlier = analyse("bad", SPEC, outlier_events, fps=30.0)

    summary = summarise(normal + [outlier])
    assert summary["骨盆→軀幹"]["n"] == 10
    assert summary["骨盆→軀幹"]["median_frames"] == pytest.approx(3.0)


def test_sequence_rate_counts_each_observed_permutation():
    good = [analyse(f"g{i}", SPEC, EVENTS, fps=30.0) for i in range(3)]
    bad_events = dict(EVENTS)
    bad_events["pelvis_peak_rotation"] = 58
    bad = [analyse("b", SPEC, bad_events, fps=30.0)]
    rates = sequence_rate(good + bad)
    assert rates["pelvis → torso → arm"] == pytest.approx(0.75)
    assert rates["torso → arm → pelvis"] == pytest.approx(0.25)


def test_segments_only_reference_canonical_events():
    """區段定義不得引用只有單一運動才有的事件，否則換運動就壞掉。"""
    for _, first, second in SEGMENTS:
        assert first in CANONICAL_EVENTS
        assert second in CANONICAL_EVENTS


def test_chain_links_are_ordered_proximal_to_distal():
    assert [name for name, _ in CHAIN_LINKS] == [
        e
        for e in CANONICAL_EVENTS
        if e in {name for name, _ in CHAIN_LINKS}
    ]
