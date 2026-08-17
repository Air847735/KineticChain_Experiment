"""由事件時間點算出動力鏈的時序指標。

偵測事件只是中間產物；真正要看的是**力量沿著動力鏈往遠端傳遞的時序**——
骨盆先轉、軀幹跟上、上肢最後爆發，各段之間隔了多久，佔整個動作的多少比例。
這一層才是教練會看的東西。

本模組完全 sport-agnostic：它只認得 :mod:`kinetic_chain.events` 的 canonical 事件，
哪些區段算得出來由該運動宣告了哪些事件決定。新增運動不需要改這裡。

**一個必須說在前面的限制**：弱標註（`weak_labels.py`）推導骨盆／軀幹／上肢峰值時，
刻意把搜尋範圍限制在加速階段內，因此近端到遠端的順序是**被建構進去的**。
用弱標註訓練出來的模型再回頭「驗證」近端到遠端序列成立，是循環論證。
要問「這批資料的序列到底成不成立」，必須用 :func:`unconstrained_sequence`
直接量原始訊號，那條路徑不受任何順序假設影響。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .events import CANONICAL_EVENTS, SportSpec
from .features import PoseSignals

#: 動力鏈上依序傳遞的三個環節，近端到遠端。
CHAIN_LINKS: tuple[tuple[str, str], ...] = (
    ("pelvis_peak_rotation", "pelvis_angular_speed"),
    ("torso_peak_rotation", "torso_angular_speed"),
    ("arm_peak_velocity", "wrist_speed"),
)

#: 要量的區段：`(顯示名稱, 起, 迄)`。兩端事件都存在時才計算。
#: 宣告式定義——新增運動時若有新的有意義區段，加在這裡即可。
SEGMENTS: tuple[tuple[str, str, str], ...] = (
    ("蓄力", "loading_start", "loading_peak"),
    ("跨步", "loading_peak", "stride_foot_contact"),
    ("軀幹加速", "stride_foot_contact", "torso_peak_rotation"),
    ("骨盆→軀幹", "pelvis_peak_rotation", "torso_peak_rotation"),
    ("軀幹→上肢", "torso_peak_rotation", "arm_peak_velocity"),
    ("上肢→出手", "arm_peak_velocity", "release_impact"),
    ("加速期", "stride_foot_contact", "release_impact"),
    ("隨勢", "release_impact", "finish"),
)


@dataclass(frozen=True)
class Segment:
    """兩個事件之間的一段時間。"""

    name: str
    start_event: str
    end_event: str
    frames: int
    seconds: float
    percent_of_throw: float


@dataclass(frozen=True)
class ChainAnalysis:
    """一次動作的動力鏈時序。

    Attributes
    ----------
    sequence:
        該動作實際觀察到的近端到遠端環節，依時間排序。
    sequence_ok:
        觀察到的順序是否等於力學上預期的近端到遠端順序。
    throw_frames:
        參考區間長度（起始事件到 ``release_impact``），用來正規化。
    timeline:
        事件 id → 佔參考區間的百分比。負值代表在起始事件之前。
    """

    clip_id: str
    sport: str
    fps: float
    events: Mapping[str, int]
    segments: tuple[Segment, ...]
    sequence: tuple[str, ...]
    sequence_ok: bool
    throw_frames: int
    timeline: Mapping[str, float]

    @property
    def throw_seconds(self) -> float:
        return self.throw_frames / self.fps if self.fps > 0 else float("nan")

    def segment(self, name: str) -> Segment | None:
        for item in self.segments:
            if item.name == name:
                return item
        return None


def _reference_span(events: Mapping[str, int], order: Sequence[str]) -> tuple[int, int]:
    """參考區間 `(起, 迄)`：最早的事件到 ``release_impact``。

    用 ``release_impact`` 當終點而不是 ``finish``：隨勢的長度受拍攝裁切影響很大，
    拿它正規化會讓同一個投球在不同剪法下算出不同比例。
    """
    present = [e for e in order if e in events]
    if not present:
        return 0, 0
    start = events[present[0]]
    end = events.get("release_impact", events[present[-1]])
    return start, end


def analyse(
    clip_id: str,
    spec: SportSpec,
    events: Mapping[str, int],
    fps: float,
) -> ChainAnalysis:
    """由事件影格算出動力鏈時序指標。

    Parameters
    ----------
    events:
        事件 id → 影格索引。可以是該運動宣告事件的子集；算不出來的區段直接略過，
        不以估計值補。
    """
    order = spec.events
    start, end = _reference_span(events, order)
    span = max(end - start, 1)

    segments = []
    for name, first, second in SEGMENTS:
        if first not in events or second not in events:
            continue
        frames = int(events[second] - events[first])
        segments.append(
            Segment(
                name=name,
                start_event=first,
                end_event=second,
                frames=frames,
                seconds=frames / fps if fps > 0 else float("nan"),
                percent_of_throw=100.0 * frames / span,
            )
        )

    links = [name for name, _ in CHAIN_LINKS if name in events]
    observed = tuple(sorted(links, key=lambda e: events[e]))

    timeline = {
        event: 100.0 * (frame - start) / span
        for event, frame in events.items()
        if event in order
    }

    return ChainAnalysis(
        clip_id=clip_id,
        sport=spec.sport_id,
        fps=fps,
        events=dict(events),
        segments=tuple(segments),
        sequence=observed,
        sequence_ok=observed == tuple(links),
        throw_frames=int(span),
        timeline=timeline,
    )


def unconstrained_sequence(
    signals: PoseSignals,
    *,
    window: tuple[int, int] | None = None,
) -> dict[str, int]:
    """直接取原始訊號的峰值，**不套任何順序假設**。

    這是唯一能誠實回答「這批資料的近端到遠端序列成不成立」的量法。弱標註為了
    讓標註可用，把骨盆／軀幹的搜尋範圍限制在上肢峰值之前；那條路徑必然得出
    正確順序，不能拿來當證據。

    Parameters
    ----------
    window:
        `(lo, hi)` 影格範圍。``None`` 表示整段。給定加速階段的範圍時，量到的是
        「投球動作內的峰值」；不給則包含隨勢，兩者的答案可能不同——差異本身
        就是結果的一部分。
    """
    peaks: dict[str, int] = {}
    for event, signal_name in CHAIN_LINKS:
        values = signals.signals[signal_name]
        lo, hi = (0, values.size) if window is None else window
        lo = max(0, min(lo, values.size - 1))
        hi = max(lo + 1, min(hi, values.size))
        peaks[event] = int(lo + np.argmax(values[lo:hi]))
    return peaks


def summarise(analyses: Sequence[ChainAnalysis]) -> dict[str, dict[str, float]]:
    """一批動作的區段統計。

    回傳每個區段的中位數、四分位距與樣本數。用中位數而非平均：單機 2D 姿態
    偶爾會有離群的估計，平均會被少數壞樣本拉走。
    """
    buckets: dict[str, list[Segment]] = {}
    for item in analyses:
        for segment in item.segments:
            buckets.setdefault(segment.name, []).append(segment)

    summary: dict[str, dict[str, float]] = {}
    for name, items in buckets.items():
        frames = np.array([s.frames for s in items], dtype=float)
        seconds = np.array([s.seconds for s in items], dtype=float)
        percent = np.array([s.percent_of_throw for s in items], dtype=float)
        summary[name] = {
            "n": float(len(items)),
            "median_frames": float(np.median(frames)),
            "median_ms": float(np.median(seconds) * 1000.0),
            "iqr_ms": float((np.percentile(seconds, 75) - np.percentile(seconds, 25)) * 1000.0),
            "median_percent": float(np.median(percent)),
        }
    return summary


def sequence_rate(analyses: Sequence[ChainAnalysis]) -> dict[str, float]:
    """符合近端到遠端順序的比例，以及各種實際觀察到的排列。"""
    if not analyses:
        return {}
    counts: dict[str, int] = {}
    for item in analyses:
        key = " → ".join(e.replace("_peak_rotation", "").replace("_peak_velocity", "")
                         for e in item.sequence)
        counts[key] = counts.get(key, 0) + 1
    total = len(analyses)
    return {key: count / total for key, count in sorted(counts.items(), key=lambda kv: -kv[1])}


def format_report(
    summary: Mapping[str, Mapping[str, float]],
    rates: Mapping[str, float],
    *,
    title: str = "動力鏈時序",
) -> str:
    """人看的文字報表。"""
    lines = [title, "=" * len(title) * 2, ""]
    lines.append(f"  {'區段':<12}{'n':>5}{'中位(格)':>10}{'中位(ms)':>11}{'IQR(ms)':>10}{'佔比%':>9}")
    for name, stats in summary.items():
        lines.append(
            f"  {name:<12}{stats['n']:>5.0f}{stats['median_frames']:>10.1f}"
            f"{stats['median_ms']:>11.0f}{stats['iqr_ms']:>10.0f}{stats['median_percent']:>9.1f}"
        )
    lines.append("")
    lines.append("  觀察到的近端→遠端排列：")
    for key, rate in rates.items():
        lines.append(f"    {rate * 100:5.1f}%  {key}")
    return "\n".join(lines)
