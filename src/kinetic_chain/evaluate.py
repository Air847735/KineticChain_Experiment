"""評估：模型輸出 → 順序約束解碼 → PCE 報表。

報表一律以 ``{sport}/{label_source}`` 分組。合併成單一數字會讓「模型學會了
弱標註規則」與「模型找對了真人標註的時間點」看起來像同一件事，那是兩件事。
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .data import Clip, collate
from .decode import decode
from .metrics import PCEReport, evaluate_predictions
from .model import KineticChainNet

logger = logging.getLogger(__name__)


@torch.no_grad()
def predict_clips(
    model: KineticChainNet,
    clips: Sequence[Clip],
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 16,
) -> list[dict[str, int]]:
    """對每個片段輸出 ``{事件 id: 影格索引}``。

    只解碼該片段實際有標註的事件（``clip.ordered_events``），這樣預測與真值在
    比較時涵蓋同一組事件；否則對缺標註的事件也輸出預測會讓 PCE 的分母不一致。
    """
    model.eval()
    device = torch.device(device)
    predictions: list[dict[str, int]] = []

    for start in range(0, len(clips), batch_size):
        group = list(clips[start : start + batch_size])
        batch = collate(group).to(device)
        logits = model(batch.features, batch.sport_ids, batch.frame_mask).cpu().numpy()
        for i, clip in enumerate(group):
            events = clip.ordered_events
            slots = tuple(clip.spec.slots[clip.spec.order_of(e)] for e in events)
            frames, _ = decode(logits[i], slots, valid_length=clip.num_frames)
            predictions.append(dict(zip(events, (int(f) for f in frames))))
    return predictions


def evaluate_clips(
    model: KineticChainNet,
    clips: Sequence[Clip],
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 16,
) -> dict[str, PCEReport]:
    """在片段集合上評估，回傳 ``{分組鍵: PCEReport}``（含 ``"overall"``）。"""
    if not clips:
        return {}
    predictions = predict_clips(model, clips, device=device, batch_size=batch_size)
    records = [
        (
            f"{clip.sport}/{clip.label_source}",
            clip.events,
            prediction,
            clip.ordered_events,
        )
        for clip, prediction in zip(clips, predictions)
    ]
    return evaluate_predictions(records)


def format_reports(reports: Mapping[str, PCEReport]) -> str:
    """把分組報表排版成可讀的文字。``overall`` 放最後。"""
    keys = sorted(k for k in reports if k != "overall")
    if "overall" in reports:
        keys.append("overall")
    return "\n\n".join(reports[key].format(key) for key in keys)


def sequence_violations(
    clips: Sequence[Clip], predictions: Iterable[Mapping[str, int]]
) -> int:
    """統計預測結果中違反宣告時序的片段數。

    解碼器保證不會違反（見 ``decode.constrained_argmax``），所以這個數字應該
    恆為 0。它存在的意義是**檢查那個保證真的成立**，而不是假設它成立。
    """
    violations = 0
    for clip, prediction in zip(clips, predictions):
        frames = [prediction[e] for e in clip.ordered_events if e in prediction]
        if any(b < a for a, b in zip(frames, frames[1:])):
            violations += 1
    return violations


def per_event_delta_table(
    clips: Sequence[Clip], predictions: Sequence[Mapping[str, int]]
) -> dict[str, dict[str, float]]:
    """每個事件的預測誤差分布（影格），依 ``{sport}/{event}`` 分組。"""
    buckets: dict[str, list[int]] = {}
    for clip, prediction in zip(clips, predictions):
        for event in clip.ordered_events:
            if event not in prediction:
                continue
            key = f"{clip.sport}/{event}"
            buckets.setdefault(key, []).append(
                abs(int(prediction[event]) - int(clip.events[event]))
            )
    return {
        key: {
            "n": float(len(values)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p90": float(np.percentile(values, 90)),
        }
        for key, values in sorted(buckets.items())
    }
