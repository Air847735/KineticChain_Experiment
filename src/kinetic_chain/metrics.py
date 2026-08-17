"""評估指標。

主指標為 PCE（Percentage of Correct Events）：預測影格與真值影格的距離在容忍度
之內即算正確。容忍度的定義沿用 GolfDB 的作法，使高爾夫的數字能與已發表的
SwingNet 基準直接對照。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

#: GolfDB 的容忍度定義：以「準備到擊球」的影格數除以此常數。
#: 出自 ``wmcnally/golfdb`` 的 ``util.correct_preds``：
#: ``tol = int(max(np.round((events[5] - events[0]) / 30), 1))``
TOLERANCE_DIVISOR = 30


def tolerance(
    truth: Mapping[str, int],
    *,
    start_event: str = "address",
    end_event: str = "release_impact",
    order: Sequence[str] | None = None,
) -> int:
    """由參考區間長度算出容忍度（影格）。

    參考區間預設為「準備 → 擊球／出手」，即動作的主體長度。用相對長度而非固定
    影格數，慢動作與正常速度的影片才能用同一個標準比較。

    缺少參考事件時退回使用該片段最早與最晚的事件。
    """
    if start_event in truth and end_event in truth:
        span = truth[end_event] - truth[start_event]
    else:
        keys = list(order) if order else list(truth)
        present = [truth[k] for k in keys if k in truth]
        if len(present) < 2:
            return 1
        span = max(present) - min(present)
    return int(max(round(abs(span) / TOLERANCE_DIVISOR), 1))


@dataclass
class EventScore:
    """單一事件槽的累計統計。"""

    total: int = 0
    correct: int = 0
    deltas: list[int] = field(default_factory=list)

    @property
    def pce(self) -> float:
        return self.correct / self.total if self.total else float("nan")

    @property
    def mean_delta(self) -> float:
        return float(np.mean(self.deltas)) if self.deltas else float("nan")

    @property
    def median_delta(self) -> float:
        return float(np.median(self.deltas)) if self.deltas else float("nan")


@dataclass
class PCEReport:
    """一次評估的完整結果。

    ``per_event`` 與 ``clip_pce`` 分開保留：前者回答「哪個事件難」，後者是與
    SwingNet 可比的整體數字（GolfDB 的 PCE 是先算每個片段的正確率再平均）。
    """

    per_event: dict[str, EventScore] = field(default_factory=dict)
    clip_pce: list[float] = field(default_factory=list)
    tolerances: list[int] = field(default_factory=list)

    @property
    def pce(self) -> float:
        """片段平均 PCE。與 GolfDB 的 ``eval.py`` 同定義。"""
        return float(np.mean(self.clip_pce)) if self.clip_pce else float("nan")

    @property
    def num_clips(self) -> int:
        return len(self.clip_pce)

    @property
    def mean_tolerance(self) -> float:
        return float(np.mean(self.tolerances)) if self.tolerances else float("nan")

    def add(
        self,
        truth: Mapping[str, int],
        predicted: Mapping[str, int],
        *,
        order: Sequence[str] | None = None,
    ) -> None:
        """累計一個片段。只統計真值與預測都存在的事件。"""
        shared = [e for e in (order or truth) if e in truth and e in predicted]
        if not shared:
            return
        tol = tolerance(truth, order=order)
        correct = 0
        for event in shared:
            delta = int(abs(int(predicted[event]) - int(truth[event])))
            score = self.per_event.setdefault(event, EventScore())
            score.total += 1
            score.deltas.append(delta)
            if delta <= tol:
                score.correct += 1
                correct += 1
        self.clip_pce.append(correct / len(shared))
        self.tolerances.append(tol)

    def format(self, title: str = "PCE") -> str:
        """人看的表格。"""
        lines = [
            f"{title}: {self.pce:.4f}  ({self.num_clips} clips, "
            f"mean tolerance {self.mean_tolerance:.2f} frames)",
            f"  {'event':<22}{'n':>6}{'PCE':>9}{'median Δ':>11}{'mean Δ':>9}",
        ]
        for event, score in self.per_event.items():
            lines.append(
                f"  {event:<22}{score.total:>6}{score.pce:>9.4f}"
                f"{score.median_delta:>11.1f}{score.mean_delta:>9.2f}"
            )
        return "\n".join(lines)


def evaluate_predictions(
    records: Iterable[tuple[str, Mapping[str, int], Mapping[str, int], Sequence[str]]],
) -> dict[str, PCEReport]:
    """依分組鍵彙總多個片段的預測結果。

    Parameters
    ----------
    records:
        ``(group, truth, predicted, order)`` 的可迭代物。``group`` 通常是
        ``f"{sport}/{label_source}"``——弱標註與真人標註的數字**必須**分開，
        合併成單一數字會讓「模型學會了規則」看起來像「模型找對了時間點」。

    Returns
    -------
    分組鍵 → :class:`PCEReport`，另含鍵 ``"overall"`` 的全體彙總。
    """
    reports: dict[str, PCEReport] = defaultdict(PCEReport)
    for group, truth, predicted, order in records:
        reports[group].add(truth, predicted, order=order)
        reports["overall"].add(truth, predicted, order=order)
    return dict(reports)
