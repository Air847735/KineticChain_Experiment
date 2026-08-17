"""PCE 與容忍度。

容忍度的定義直接決定所有數字能不能跟 SwingNet 對照，因此對照 GolfDB 原始公式
``tol = int(max(round((impact - address) / 30), 1))`` 逐項驗證。
"""

from __future__ import annotations

import pytest

from kinetic_chain.metrics import (
    TOLERANCE_DIVISOR,
    PCEReport,
    evaluate_predictions,
    tolerance,
)


def test_tolerance_matches_golfdb_formula():
    truth = {"address": 100, "release_impact": 220}
    assert tolerance(truth) == max(round((220 - 100) / TOLERANCE_DIVISOR), 1) == 4


def test_tolerance_is_at_least_one_frame():
    assert tolerance({"address": 10, "release_impact": 12}) == 1


def test_tolerance_falls_back_to_event_span():
    truth = {"loading_peak": 20, "finish": 140}
    order = ("loading_peak", "finish")
    assert tolerance(truth, order=order) == 4


def test_tolerance_with_a_single_event_is_one():
    assert tolerance({"address": 5}) == 1


def test_pce_counts_events_within_tolerance():
    truth = {"address": 0, "release_impact": 60, "finish": 90}
    order = ("address", "release_impact", "finish")
    report = PCEReport()
    report.add(truth, {"address": 2, "release_impact": 60, "finish": 80}, order=order)
    # tolerance = round(60/30) = 2；address 差 2 通過，finish 差 10 不通過
    assert report.clip_pce == [pytest.approx(2 / 3)]
    assert report.per_event["finish"].correct == 0


def test_pce_ignores_events_missing_from_either_side():
    truth = {"address": 0, "release_impact": 30}
    report = PCEReport()
    report.add(truth, {"address": 0}, order=("address", "release_impact"))
    assert report.per_event.keys() == {"address"}
    assert report.clip_pce == [1.0]


def test_clip_pce_averages_per_clip_not_per_event():
    """GolfDB 的 PCE 先算每段的正確率再平均；長短片段權重相同。"""
    order = ("address", "release_impact")
    report = PCEReport()
    report.add({"address": 0, "release_impact": 60}, {"address": 0, "release_impact": 60}, order=order)
    report.add({"address": 0, "release_impact": 60}, {"address": 40, "release_impact": 50}, order=order)
    assert report.pce == pytest.approx(0.5)


def test_groups_keep_human_and_weak_labels_separate():
    order = ("address", "release_impact")
    truth = {"address": 0, "release_impact": 60}
    reports = evaluate_predictions(
        [
            ("golf_swing/human", truth, truth, order),
            ("baseball_pitch/weak", truth, {"address": 55, "release_impact": 5}, order),
        ]
    )
    assert reports["golf_swing/human"].pce == pytest.approx(1.0)
    assert reports["baseball_pitch/weak"].pce == pytest.approx(0.0)
    assert reports["overall"].num_clips == 2


def test_empty_report_reports_nan_not_zero():
    """沒有資料時回 NaN，不能回 0——0 會被誤讀成「全錯」。"""
    import math

    assert math.isnan(PCEReport().pce)
