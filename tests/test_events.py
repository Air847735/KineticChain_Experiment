"""事件註冊表的一致性。

這些測試守的是跨運動的契約：事件槽是共用的，一個運動宣告錯了會影響所有運動。
"""

from __future__ import annotations

import numpy as np
import pytest

from kinetic_chain.errors import SportSpecError, UnknownEventError, UnknownSportError
from kinetic_chain.events import (
    ALL_EVENTS,
    CANONICAL_EVENTS,
    NUM_EVENT_SLOTS,
    SPORT_SPECIFIC_EVENTS,
    UNIVERSAL_ORDER,
    SportSpec,
    WeakRule,
    event_index,
    event_mask,
    get_sport,
    registered_sports,
    sport_index,
)


def test_event_vocabulary_has_no_duplicates():
    assert len(set(ALL_EVENTS)) == len(ALL_EVENTS) == NUM_EVENT_SLOTS
    assert not set(CANONICAL_EVENTS) & set(SPORT_SPECIFIC_EVENTS)


def test_sport_specific_events_are_prefixed_by_a_registered_sport():
    """專屬事件必須看得出屬於哪個運動，否則會被誤當成可共用的 canonical 事件。"""
    sports = registered_sports()
    for event in SPORT_SPECIFIC_EVENTS:
        assert any(event.startswith(sport.split("_")[0]) for sport in sports), event


@pytest.mark.parametrize("sport_id", registered_sports())
def test_every_sport_satisfies_the_universal_order(sport_id):
    """跨運動成立的那幾條順序約束，每個運動都必須遵守。

    共用輸出槽只有在各運動對同一事件的力學認知一致時才有意義。反之，
    ``loading_peak`` 與 ``stride_foot_contact`` 的相對位置本來就因運動而異
    （投擲類先舉腿到頂，擊球類先落地），所以不列入約束。
    """
    spec = get_sport(sport_id)
    position = {event: i for i, event in enumerate(spec.events)}
    for earlier, later in UNIVERSAL_ORDER:
        if earlier in position and later in position:
            assert position[earlier] < position[later], f"{earlier} 應排在 {later} 之前"


def test_sports_genuinely_disagree_on_non_universal_order():
    """確認 UNIVERSAL_ORDER 的取捨有實際依據，而不是一開始就沒人會違反。

    投擲類與擊球類對 loading_peak / stride_foot_contact 的順序相反；若哪天兩邊
    變成一致，這條約束就該重新檢討是不是可以收進 UNIVERSAL_ORDER。
    """
    def order_of(sport: str) -> tuple[int, int]:
        events = get_sport(sport).events
        return events.index("loading_peak"), events.index("stride_foot_contact")

    pitch_peak, pitch_contact = order_of("baseball_pitch")
    swing_peak, swing_contact = order_of("baseball_swing")
    assert pitch_peak < pitch_contact
    assert swing_contact < swing_peak


def test_universal_order_only_mentions_canonical_events():
    """跨運動的約束不得引用只有單一運動才有的事件。"""
    for earlier, later in UNIVERSAL_ORDER:
        assert earlier in CANONICAL_EVENTS
        assert later in CANONICAL_EVENTS


def test_spec_rejects_an_order_violating_the_universal_constraint():
    with pytest.raises(SportSpecError):
        SportSpec("bad", "壞順序", ("release_impact", "arm_peak_velocity"))


@pytest.mark.parametrize("sport_id", registered_sports())
def test_weak_rules_cover_declared_events_without_cycles(sport_id):
    spec = get_sport(sport_id)
    produced = {rule.event for rule in spec.weak_rules}
    assert produced <= set(spec.events)

    # 依賴關係必須可拓撲排序
    remaining = list(spec.weak_rules)
    resolved: set[str] = set()
    while remaining:
        ready = [
            r
            for r in remaining
            if not {
                v
                for k, v in r.params.items()
                if k in ("after", "before", "start", "end") and isinstance(v, str)
            }
            - resolved
        ]
        assert ready, f"{sport_id} 的弱標註規則有循環依賴"
        resolved |= {r.event for r in ready}
        remaining = [r for r in remaining if r not in ready]


@pytest.mark.parametrize("sport_id", registered_sports())
def test_slots_are_consistent_with_event_index(sport_id):
    spec = get_sport(sport_id)
    assert spec.slots == tuple(event_index(e) for e in spec.events)


def test_sport_index_is_stable_and_dense():
    sports = registered_sports()
    assert list(sports) == sorted(sports)
    assert [sport_index(s) for s in sports] == list(range(len(sports)))


def test_event_mask_selects_only_declared_events():
    mask = event_mask("baseball_pitch", {"address": 0, "release_impact": 10})
    assert mask.sum() == 2
    assert mask[event_index("address")]
    assert mask[event_index("release_impact")]


def test_event_mask_rejects_events_the_sport_does_not_declare():
    with pytest.raises(UnknownEventError):
        event_mask("tennis_serve", {"golf_toe_up": 3})


def test_unknown_sport_is_an_error_not_a_fuzzy_match():
    with pytest.raises(UnknownSportError):
        get_sport("golf")  # 'golf_swing' 存在，但不做模糊比對


def test_spec_rejects_duplicate_events():
    with pytest.raises(SportSpecError):
        SportSpec("x", "X", ("address", "address"))


def test_spec_rejects_unknown_event():
    with pytest.raises(UnknownEventError):
        SportSpec("x", "X", ("not_an_event",))


def test_spec_rejects_weak_rule_for_undeclared_event():
    with pytest.raises(SportSpecError):
        SportSpec("x", "X", ("address",), weak_rules=(WeakRule("finish", "rest_end"),))


def test_weak_rule_params_are_immutable():
    rule = WeakRule("address", "rest_start", {"signal": "body_speed"})
    with pytest.raises(TypeError):
        rule.params["signal"] = "other"  # type: ignore[index]
