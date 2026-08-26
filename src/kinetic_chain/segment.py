"""把未裁切的長影片切成一次次的動作片段。

管線其他部分都假設「一段片段恰好包含一次完整動作」（`docs/architecture.md` 的 A1）。
一支十分鐘的訓練影片不滿足這個假設：裡面有 N 次反覆，中間夾著休息、走位、換槓片、
與人講話。直接餵進模型的話，順序約束解碼會強迫它在整支影片上只輸出一組事件。

本模組只回答「動作發生在哪幾段」，不回答「事件在第幾格」——後者仍由既有的
模型與解碼負責，對每一段各跑一次。

**作法是門檻式的，不是學習式的。** 以全身速度當活動量，低於休息水位的區間視為間隔。
理由：`weak_labels` 的 `rest_start` / `rest_end` 早就用同一個概念判定動作的起訖，
沿用同一套語意可以少一個需要驗證的東西。代價是遇到「反覆之間不停下來」的動作
（連續揮棒練習）會切不開，見 `should_trust`。

核心層模組，只用 numpy，不匯入任何影像後端。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import KineticChainError
from .features import PoseSignals

#: 休息水位取活動量的這個百分位，不取最小值。
#:
#: 取最小值會被單一影格的姿態抖動釘死——長影片裡總有一格估得特別穩。
#: 本專案在 `sagittal_visibility` 上已經犯過一次同樣的錯（見
#: `docs/lift-analysis.md`），那次用 min 導致 11 段片段的髖角低於解剖上可能的值。
REST_PERCENTILE = 20.0

#: 活動量高於「休息水位 + 這個比例 × 動態範圍」才算在動作中。
ACTIVE_FRACTION = 0.18

#: 兩段動作之間短於這個秒數的間隔會被合併——同一次動作中間的減速不該切成兩段。
#:
#: 初版設 0.5 秒，在 12 支自備舉重影片（每支恰好一次舉，正解就是 1 段）上只有
#: 6/12 正確，其餘被切成 2 到 5 段。挺舉的上膊與上挺之間本來就停 1 到 2 秒。
#: 掃描結果（wrist_speed / body_speed 各 12 支恰好切出 1 段的數量）：
#:
#:     0.5s → 6 / 6      1.0s → 8 / 6      1.5s → 9 / 10
#:     2.0s → 9 / 10     2.5s → 10 / 11    3.0s → 10 / 11
#:
#: 取 1.5 秒：比動作內部的停頓長，但遠短於反覆之間的休息（舉重通常十秒以上），
#: 不至於把兩次反覆併成一次。再往上調只多對 1 支，卻提高併錯的風險。
MERGE_GAP_SECONDS = 1.5

#: 短於這個秒數的活動視為雜訊（調整站位、擦手），不算一次動作。
MIN_ACTION_SECONDS = 0.7

#: 每段前後各留這麼多秒，確保 `address` 與 `finish` 落在片段內。
MARGIN_SECONDS = 0.4

#: 判定「反覆之間有停下來」的門檻：間隔總長至少要佔影片這個比例。
#: 低於此值代表動作是連續的，門檻法切不出可信的邊界。
MIN_REST_SHARE = 0.08


@dataclass(frozen=True)
class ActionSegment:
    """長影片中的一次動作。

    Attributes
    ----------
    start, end:
        影格索引，``end`` 為 exclusive，已含前後留白。
    peak:
        活動量最大的影格，可當這次動作的代表格。
    peak_activity:
        該影格的活動量，正規化到整支影片的 0–1。越接近 1 越像一次完整發力。
    """

    start: int
    end: int
    peak: int
    peak_activity: float

    @property
    def num_frames(self) -> int:
        return self.end - self.start

    def seconds(self, fps: float) -> tuple[float, float]:
        return self.start / fps, self.end / fps


@dataclass(frozen=True)
class SegmentationReport:
    """切分結果與它可不可信。"""

    segments: tuple[ActionSegment, ...]
    rest_share: float
    activity_signal: str
    should_trust: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "num_segments": len(self.segments),
            "rest_share": round(self.rest_share, 4),
            "activity_signal": self.activity_signal,
            "should_trust": self.should_trust,
            "reason": self.reason,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "frames": s.num_frames,
                    "peak": s.peak,
                    "peak_activity": round(s.peak_activity, 4),
                }
                for s in self.segments
            ],
        }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """把布林陣列的 True 區段轉成 `(起, 迄)` 清單，迄為 exclusive。"""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def find_actions(
    signals: PoseSignals,
    fps: float,
    *,
    activity_signal: str = "body_speed",
    active_fraction: float = ACTIVE_FRACTION,
    merge_gap_seconds: float = MERGE_GAP_SECONDS,
    min_action_seconds: float = MIN_ACTION_SECONDS,
    margin_seconds: float = MARGIN_SECONDS,
) -> SegmentationReport:
    """在未裁切的姿態序列上找出每一次動作。

    Parameters
    ----------
    signals:
        整支長影片的 :class:`~kinetic_chain.features.PoseSignals`。
    activity_signal:
        當活動量用的訊號名稱。預設 ``body_speed``（全身關節速度均值）；
        器械類動作若身體位移小，可改用 ``wrist_speed``。
    active_fraction:
        高於「休息水位 + 此比例 × 動態範圍」才算在動作中。

    Returns
    -------
    :class:`SegmentationReport`。``should_trust`` 為 False 時仍會回傳切分結果，
    但那組邊界不應該直接採用——呼叫端要嘛換訊號、要嘛改用人工標記。

    Raises
    ------
    KineticChainError
        ``activity_signal`` 不存在，或影片短到放不下一次動作。
    """
    if fps <= 0:
        raise KineticChainError(f"fps 必須為正，收到 {fps}")
    try:
        activity = np.asarray(signals.signals[activity_signal], dtype=float)
    except KeyError as exc:
        raise KineticChainError(
            f"未知的活動量訊號 {activity_signal!r}；"
            f"可用的有 {sorted(signals.signals)}"
        ) from exc

    n = activity.size
    if n < int(round(min_action_seconds * fps)):
        raise KineticChainError(
            f"序列只有 {n} 影格，放不下一次 {min_action_seconds} 秒的動作"
        )

    rest = float(np.percentile(activity, REST_PERCENTILE))
    top = float(np.percentile(activity, 99.0))
    dynamic = top - rest
    if dynamic < 1e-9:
        return SegmentationReport(
            segments=(), rest_share=1.0, activity_signal=activity_signal,
            should_trust=False, reason="活動量幾乎沒有變化，整支影片看起來都是靜止的",
        )

    active = activity > rest + active_fraction * dynamic
    rest_share = float(1.0 - active.mean())

    runs = _runs(active)

    # 合併：同一次動作中間的減速（例如舉重的接槓）不該切成兩段
    merge_gap = int(round(merge_gap_seconds * fps))
    merged: list[list[int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    # 過短的活動是雜訊，不是一次動作
    min_frames = int(round(min_action_seconds * fps))
    kept = [(s, e) for s, e in merged if e - s >= min_frames]

    # 加留白，但不得越過鄰段：兩段之間的空隙不夠塞下兩份留白時，
    # 從空隙的中點切開。沒有這一步，相鄰的兩次動作會互相吃進對方的片段
    # ——實測 IMG_4787 就出現 8.9–10.4 秒與 10.2–12.2 秒重疊。
    margin = int(round(margin_seconds * fps))
    padded: list[list[int]] = []
    for index, (start, end) in enumerate(kept):
        lo = max(0, start - margin)
        hi = min(n, end + margin)
        if index > 0:
            previous_end = kept[index - 1][1]
            midpoint = (previous_end + start) // 2
            lo = max(lo, midpoint)
            padded[-1][1] = min(padded[-1][1], midpoint)
        padded.append([lo, hi])

    segments = []
    for (lo, hi), (start, end) in zip(padded, kept):
        peak = start + int(activity[start:end].argmax())
        segments.append(
            ActionSegment(
                start=lo,
                end=hi,
                peak=peak,
                peak_activity=float((activity[peak] - rest) / dynamic),
            )
        )

    # 貼齊影片邊界的片段代表動作延伸到畫面之外，那一段的起訖無從驗證。
    # 這同時擋掉兩種情況：影片從動作進行中開始，以及整支影片是連續反覆
    # （活動量從頭到尾沒有回到休息水位，於是被併成貫穿全片的一段）。
    unbounded = [s for s in segments if s.start == 0 or s.end == n]

    if not segments:
        reason = "找不到任何夠長的活動區間；可能是門檻太高，或這支影片沒有完整動作"
        trust = False
    elif unbounded:
        reason = (
            f"{len(unbounded)} 段貼齊影片邊界，看不到它前後的休息——"
            "可能是影片從動作中途開始／結束，也可能整段是連續反覆而被併成一段"
        )
        trust = False
    elif rest_share < MIN_REST_SHARE:
        reason = (
            f"間隔只佔 {rest_share:.0%}，低於 {MIN_REST_SHARE:.0%}——"
            "動作之間沒有明顯停頓，門檻法切不出可信的邊界"
        )
        trust = False
    else:
        reason = f"間隔佔 {rest_share:.0%}，切出 {len(segments)} 段"
        trust = True

    return SegmentationReport(
        segments=tuple(segments),
        rest_share=rest_share,
        activity_signal=activity_signal,
        should_trust=trust,
        reason=reason,
    )
