"""人工標註 CSV 的讀寫與驗證。

這是唯一能在自備影片上產生真值的路徑，格式的每一條規則都直接影響資料品質，
所以驗證行為要測得比一般轉接層細——特別是「空白不等於 0」這條。
"""

from __future__ import annotations

import pytest

from kinetic_chain.datasets.annotations import (
    META_COLUMNS,
    event_columns,
    read,
    write_template,
)
from kinetic_chain.errors import DatasetError, UnknownEventError
from kinetic_chain.events import get_sport

SPORT = "clean_and_jerk"
EVENTS = ["address", "clean_liftoff", "clean_catch", "finish"]


def test_event_columns_ignores_metadata():
    header = ["video", "attempt", "fps", *EVENTS, "note"]
    assert event_columns(header) == EVENTS


def test_event_columns_rejects_unknown_event():
    """欄名打錯而被靜默忽略會變成無聲的資料損失，所以要直接報錯。"""
    with pytest.raises(UnknownEventError):
        event_columns(["video", "fps", "clean_liftof"])  # 少一個 f


def test_event_columns_requires_at_least_one_event():
    with pytest.raises(DatasetError):
        event_columns(list(META_COLUMNS))


def test_round_trip(tmp_path):
    path = write_template(
        tmp_path / "a.csv",
        [{"video": "v1", "attempt": 1, "fps": 60.0,
          "address": 10, "clean_liftoff": 20, "clean_catch": 30, "finish": 90}],
        EVENTS,
    )
    events, rows = read(path)
    assert events == EVENTS
    assert rows[0]["video"] == "v1"
    assert rows[0]["address"] == "10"


def test_blank_cells_survive_the_round_trip(tmp_path):
    """空白必須保持空白——寫成 0 會讓「沒標」變成「標在第 0 格」。"""
    path = write_template(
        tmp_path / "b.csv",
        [{"video": "v1", "fps": 60.0, "address": 10, "finish": 90}],
        EVENTS,
    )
    _, rows = read(path)
    assert rows[0]["clean_liftoff"] == ""
    assert rows[0]["clean_catch"] == ""


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(DatasetError):
        read(tmp_path / "nope.csv")


def test_template_columns_are_ordered_and_complete(tmp_path):
    path = write_template(tmp_path / "c.csv", [], EVENTS)
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == ["video", "attempt", "fps", *EVENTS, "note"]


def test_reduced_event_set_is_allowed(tmp_path):
    """想少標幾個階段就刪欄位——格式不寫死事件數。"""
    fewer = ["address", "clean_catch", "finish"]
    path = write_template(tmp_path / "d.csv", [{"video": "v", "address": 1}], fewer)
    events, _ = read(path)
    assert events == fewer
    assert all(e in get_sport(SPORT).events for e in events)
