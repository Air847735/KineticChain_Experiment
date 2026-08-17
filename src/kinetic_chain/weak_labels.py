"""由力學訊號程式化推導事件（**弱標註**）。

Penn Action 有 2326 段影片與每影格的人工關節標註，但**沒有事件標註**。要讓
「一個模型多運動」成立，就得有這些運動的事件時間點；自行人工標註兩千段影片
不切實際，因此改以確定性的運動學規則推導。

**這些規則產生的不是真值。** 模型在弱標註上的分數只說明模型學會了這些規則，
不說明規則本身正確。任何評估報表都必須把弱標註與真人標註分開統計
（見 :func:`kinetic_chain.metrics.evaluate_predictions`）。

規則本身是宣告式的，寫在 :mod:`kinetic_chain.events` 的 ``SportSpec.weak_rules``；
這裡只實作規則的語意。新增運動項目時寫規則，不寫程式。
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from .errors import WeakLabelError
from .events import SportSpec
from .features import PoseSignals

#: 判定「靜止」的門檻，佔該片段身體速度全距的比例。
REST_FRACTION = 0.15

#: ``signal_onset`` 判定「開始移動」的門檻，佔起點到極值差距的比例。
ONSET_FRACTION = 0.10

#: ``foot_contact`` 判定「已著地」的門檻，佔踝高度全距的比例。
CONTACT_FRACTION = 0.15

#: 允許擠在同一影格的事件數上限。
#:
#: **三路平手是取樣率的極限，不是規則失敗**：動力鏈的骨盆／軀幹／上肢峰值在
#: 30 fps 下本來就相隔不到一格（見 `docs/pitch-analysis.md`），實測每個運動都有
#: 三分之一左右的片段如此。門檻設 2 會砍掉一半資料而且砍錯對象。
#: 四路以上才是搜尋窗互相塌陷的徵兆。
MAX_TIED_EVENTS = 3

Resolver = Callable[[PoseSignals, Mapping[str, object], dict[str, int]], int]


def _window(
    params: Mapping[str, object], resolved: dict[str, int], length: int
) -> tuple[int, int]:
    """把 ``after`` / ``before`` 參數換算成 ``[lo, hi)`` 的搜尋區間。

    參照的事件尚未解出時視為無限制——依賴由 :func:`derive` 的迭代解析處理，
    這裡不該看到未解的依賴，但保守處理避免 KeyError。

    區間空掉（``after`` 解出的影格晚於 ``before``）時退化成 ``lo`` 單格，而不是
    放寬回整段：區間空掉本身就代表順序已經被違反，放寬只會掩蓋問題，讓
    :func:`derive` 的順序檢查失去作用。
    """
    lo = 0
    hi = length
    after = params.get("after")
    before = params.get("before")
    if isinstance(after, str) and after in resolved:
        lo = max(lo, resolved[after])
    if isinstance(before, str) and before in resolved:
        hi = min(hi, resolved[before] + 1)
    if hi <= lo:
        hi = min(lo + 1, length)
    return lo, hi


def _signal(signals: PoseSignals, params: Mapping[str, object]) -> np.ndarray:
    name = params.get("signal")
    if not isinstance(name, str):
        raise WeakLabelError(f"規則缺少 signal 參數：{dict(params)}")
    try:
        return signals.signals[name]
    except KeyError as exc:
        raise WeakLabelError(
            f"未知的訊號 {name!r}；可用的有 {sorted(signals.signals)}"
        ) from exc


def _rest_threshold(speed: np.ndarray) -> float:
    lo = float(np.percentile(speed, 5))
    hi = float(np.percentile(speed, 95))
    return lo + REST_FRACTION * (hi - lo)


def _rule_rest_start(
    signals: PoseSignals, params: Mapping[str, object], resolved: dict[str, int]
) -> int:
    """準備姿勢：動作開始前的最後一個靜止影格。"""
    speed = signals.signals["body_speed"]
    moving = np.flatnonzero(speed > _rest_threshold(speed))
    if moving.size == 0:
        return 0
    return int(max(0, moving[0] - 1))


def _rule_rest_end(
    signals: PoseSignals, params: Mapping[str, object], resolved: dict[str, int]
) -> int:
    """結束姿勢：動作停止後的第一個靜止影格。"""
    speed = signals.signals["body_speed"]
    moving = np.flatnonzero(speed > _rest_threshold(speed))
    if moving.size == 0:
        return int(speed.size - 1)
    return int(min(speed.size - 1, moving[-1] + 1))


def _rule_signal_peak(
    signals: PoseSignals, params: Mapping[str, object], resolved: dict[str, int]
) -> int:
    """訊號在搜尋區間內的最大值位置。"""
    values = _signal(signals, params)
    lo, hi = _window(params, resolved, values.size)
    return int(lo + np.argmax(values[lo:hi]))


def _rule_signal_extreme(
    signals: PoseSignals, params: Mapping[str, object], resolved: dict[str, int]
) -> int:
    """訊號在搜尋區間內的極值位置，``mode`` 決定取最大或最小。"""
    values = _signal(signals, params)
    lo, hi = _window(params, resolved, values.size)
    segment = values[lo:hi]
    mode = params.get("mode", "max")
    if mode == "min":
        return int(lo + np.argmin(segment))
    if mode == "max":
        return int(lo + np.argmax(segment))
    raise WeakLabelError(f"signal_extreme 的 mode 只能是 max 或 min，收到 {mode!r}")


def _rule_signal_onset(
    signals: PoseSignals, params: Mapping[str, object], resolved: dict[str, int]
) -> int:
    """訊號開始朝極值移動的位置。

    從搜尋區間的終點（通常是該訊號的極值所在）往回找，最後一個仍停留在起點
    附近的影格即為起始點。往回找而不是往前找：起點附近可能有微幅晃動，
    正向掃描容易被雜訊觸發。
    """
    values = _signal(signals, params)
    lo, hi = _window(params, resolved, values.size)
    end = hi - 1
    baseline = float(values[lo])
    span = abs(float(values[end]) - baseline)
    if span < 1e-9:
        return lo
    threshold = ONSET_FRACTION * span
    near_baseline = np.flatnonzero(np.abs(values[lo : end + 1] - baseline) <= threshold)
    if near_baseline.size == 0:
        return lo
    return int(lo + near_baseline[-1])


def _rule_foot_contact(
    signals: PoseSignals, params: Mapping[str, object], resolved: dict[str, int]
) -> int:
    """前腳著地：前踝抬起後首次回到接近最低高度的影格。

    用「回到最低高度」而不是「垂直速度過零」：踝關節在整段動作中多數時間都貼地，
    速度過零的點很多；抬起後的下降段只會有一次落回底部。
    """
    height = signals.signals["lead_ankle_height"]
    lo, hi = _window(params, resolved, height.size)
    segment = height[lo:hi]
    if segment.size == 0:
        return lo
    floor = float(np.min(segment))
    ceiling = float(np.max(segment))
    span = ceiling - floor
    if span < 1e-9:
        return int(lo + segment.size // 2)
    peak = int(np.argmax(segment))
    threshold = floor + CONTACT_FRACTION * span
    landed = np.flatnonzero(segment[peak:] <= threshold)
    if landed.size == 0:
        return int(lo + peak)
    return int(lo + peak + landed[0])


def _rule_post_peak_decel(
    signals: PoseSignals, params: Mapping[str, object], resolved: dict[str, int]
) -> int:
    """出手／擊球：遠端速度峰值之後減速最劇烈的影格。

    球離手或球具觸球之後，手部速度會急遽下降；減速的極值比速度峰值本身更貼近
    真正的釋放瞬間。
    """
    values = _signal(signals, params)
    lo, hi = _window(params, resolved, values.size)
    segment = values[lo:hi]
    if segment.size < 3:
        return int(lo + np.argmax(segment)) if segment.size else lo
    peak = int(np.argmax(segment))
    tail = segment[peak:]
    if tail.size < 2:
        return int(lo + peak)
    return int(lo + peak + np.argmin(np.gradient(tail)))


def _rule_signal_crossing(
    signals: PoseSignals, params: Mapping[str, object], resolved: dict[str, int]
) -> int:
    """兩條訊號交叉的影格：``signal`` 由下方升到 ``reference`` 之上。

    用於「槓鈴通過膝蓋」這類以相對高度定義的事件。比絕對門檻穩健——
    兩條訊號共用同一個身體尺度，個體差異會互相抵消。
    """
    values = _signal(signals, params)
    other = params.get("reference")
    if not isinstance(other, str):
        raise WeakLabelError(f"signal_crossing 需要 reference 參數：{dict(params)}")
    try:
        baseline = signals.signals[other]
    except KeyError as exc:
        raise WeakLabelError(f"未知的參考訊號 {other!r}") from exc

    lo, hi = _window(params, resolved, values.size)
    difference = (values - baseline)[lo:hi]
    above = np.flatnonzero(difference > 0)
    if above.size == 0:
        return int(lo + np.argmax(difference))
    return int(lo + above[0])


def _rule_midpoint(
    signals: PoseSignals, params: Mapping[str, object], resolved: dict[str, int]
) -> int:
    """兩個已解出事件的中點。"""
    start = params.get("start")
    end = params.get("end")
    if not isinstance(start, str) or not isinstance(end, str):
        raise WeakLabelError(f"midpoint 需要 start 與 end 參數：{dict(params)}")
    if start not in resolved or end not in resolved:
        raise KeyError((start, end))
    return int(round((resolved[start] + resolved[end]) / 2))


RULES: dict[str, Resolver] = {
    "rest_start": _rule_rest_start,
    "rest_end": _rule_rest_end,
    "signal_peak": _rule_signal_peak,
    "signal_extreme": _rule_signal_extreme,
    "signal_onset": _rule_signal_onset,
    "foot_contact": _rule_foot_contact,
    "post_peak_decel": _rule_post_peak_decel,
    "signal_crossing": _rule_signal_crossing,
    "midpoint": _rule_midpoint,
}


def _dependencies(params: Mapping[str, object]) -> set[str]:
    return {
        value
        for key, value in params.items()
        if key in ("after", "before", "start", "end") and isinstance(value, str)
    }


def derive(
    signals: PoseSignals,
    spec: SportSpec,
    *,
    enforce_order: bool = True,
    max_tied_events: int = MAX_TIED_EVENTS,
) -> dict[str, int]:
    """套用運動項目宣告的弱標註規則，回傳事件 id → 影格索引。

    規則之間可以互相依賴（``after`` / ``before`` / ``start`` / ``end``），依賴關係
    以迭代解析，不要求宣告順序即為求解順序。

    Raises
    ------
    WeakLabelError
        規則之間有循環依賴、規則名稱未知，或 ``enforce_order`` 為真時推導結果
        違反該運動宣告的事件時序。
    """
    if not spec.weak_rules:
        raise WeakLabelError(f"{spec.sport_id!r} 沒有宣告任何弱標註規則")

    length = signals.pose.shape[0]
    resolved: dict[str, int] = {}
    pending = list(spec.weak_rules)

    while pending:
        progressed = False
        deferred = []
        for rule in pending:
            try:
                resolver = RULES[rule.rule]
            except KeyError as exc:
                raise WeakLabelError(
                    f"未知的規則 {rule.rule!r}；可用的有 {sorted(RULES)}"
                ) from exc
            missing = _dependencies(rule.params) - resolved.keys()
            if missing:
                deferred.append(rule)
                continue
            frame = resolver(signals, rule.params, resolved)
            resolved[rule.event] = int(np.clip(frame, 0, length - 1))
            progressed = True
        if not progressed:
            unresolved = sorted(r.event for r in deferred)
            raise WeakLabelError(
                f"{spec.sport_id!r} 的弱標註規則有循環或缺失的依賴：{unresolved}"
            )
        pending = deferred

    ordered = [e for e in spec.events if e in resolved]
    if enforce_order:
        frames = [resolved[e] for e in ordered]
        if any(b < a for a, b in zip(frames, frames[1:])):
            raise WeakLabelError(
                f"{spec.sport_id!r} 的弱標註違反宣告的事件時序："
                + ", ".join(f"{e}={resolved[e]}" for e in ordered)
            )

        counts: dict[int, int] = {}
        for frame in frames:
            counts[frame] = counts.get(frame, 0) + 1
        worst = max(counts.values(), default=0)
        if worst > max_tied_events:
            at = max(counts, key=lambda f: counts[f])
            raise WeakLabelError(
                f"{spec.sport_id!r} 有 {worst} 個事件擠在影格 {at}，超過上限 "
                f"{max_tied_events}——搜尋窗互相塌陷，推出來的是邊界值不是事件："
                + ", ".join(f"{e}={resolved[e]}" for e in ordered)
            )

        # 只有 address 與 finish 有理由落在片段的頭尾。其他事件被釘在邊界，
        # 代表它的搜尋窗塌到 `_window` 的退化分支，取到的是邊界值不是極值。
        last = length - 1
        pinned = [
            e for e in ordered
            if e not in ("address", "finish") and resolved[e] in (0, last)
        ]
        if len(pinned) >= 2:
            raise WeakLabelError(
                f"{spec.sport_id!r} 有 {len(pinned)} 個中段事件被釘在片段邊界 "
                f"{pinned}——搜尋窗塌到邊界，取到的不是極值"
            )
    return {e: resolved[e] for e in ordered}
