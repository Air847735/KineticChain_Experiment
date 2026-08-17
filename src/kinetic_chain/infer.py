"""單支影片的端到端推論。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .decode import decode
from .errors import ClipTooShortError
from .events import get_sport, sport_index
from .features import MIN_FRAMES, build, compute
from .model import KineticChainNet
from .skeleton import to_canonical

logger = logging.getLogger(__name__)

#: 姿態覆蓋率低於此值時輸出仍會產生，但標記為低信心。
LOW_COVERAGE = 0.8


@dataclass(frozen=True)
class EventPrediction:
    """單一事件的預測結果。"""

    event: str
    frame: int
    time: float
    confidence: float


@dataclass(frozen=True)
class InferenceResult:
    """一支影片的完整推論結果。"""

    sport: str
    fps: float
    num_frames: int
    events: tuple[EventPrediction, ...]
    pose_coverage: float
    low_confidence: bool
    flipped: bool

    def as_dict(self) -> dict:
        return {
            "sport": self.sport,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "pose_coverage": round(self.pose_coverage, 4),
            "low_confidence": self.low_confidence,
            "flipped": self.flipped,
            "events": [
                {
                    "event": e.event,
                    "frame": e.frame,
                    "time": round(e.time, 4),
                    "confidence": round(e.confidence, 4),
                }
                for e in self.events
            ],
        }

    def format(self) -> str:
        lines = [
            f"{self.sport}  {self.num_frames} frames @ {self.fps:.2f} fps"
            f"  (pose coverage {self.pose_coverage:.2f}"
            + ("，低信心" if self.low_confidence else "")
            + ")",
            f"  {'event':<22}{'frame':>7}{'time(s)':>10}{'conf':>8}",
        ]
        for e in self.events:
            lines.append(f"  {e.event:<22}{e.frame:>7}{e.time:>10.3f}{e.confidence:>8.3f}")
        return "\n".join(lines)


@torch.no_grad()
def predict_pose_sequence(
    model: KineticChainNet,
    pose: np.ndarray,
    fps: float,
    sport: str,
    *,
    device: torch.device | str = "cpu",
) -> InferenceResult:
    """從 canonical 關鍵點序列推論事件。

    Parameters
    ----------
    pose:
        ``(T, 13, 3)``，canonical 布局。
    """
    spec = get_sport(sport)
    if pose.shape[0] < MIN_FRAMES:
        raise ClipTooShortError(
            f"片段只有 {pose.shape[0]} 影格，少於最低要求 {MIN_FRAMES}"
        )
    if pose.shape[0] < len(spec.events):
        raise ClipTooShortError(
            f"片段只有 {pose.shape[0]} 影格，排不下 {sport!r} 宣告的 "
            f"{len(spec.events)} 個事件"
        )

    signals = compute(pose, fps, handedness_sensitive=spec.handedness_sensitive)
    features = build(pose, fps, signals=signals)

    device = torch.device(device)
    model.eval()
    logits = model(
        torch.from_numpy(features).unsqueeze(0).to(device),
        torch.tensor([sport_index(sport)], dtype=torch.long, device=device),
    )[0].cpu().numpy()

    frames, confidence = decode(logits, spec.slots)
    events = tuple(
        EventPrediction(
            event=name,
            frame=int(frame),
            time=float(frame) / fps if fps > 0 else 0.0,
            confidence=float(score),
        )
        for name, frame, score in zip(spec.events, frames, confidence)
    )

    return InferenceResult(
        sport=sport,
        fps=fps,
        num_frames=int(pose.shape[0]),
        events=events,
        pose_coverage=signals.coverage,
        low_confidence=signals.coverage < LOW_COVERAGE,
        flipped=signals.flipped,
    )


def predict_video(
    model: KineticChainNet,
    video: Path | str,
    sport: str,
    *,
    device: torch.device | str = "cpu",
    pose_device: str = "cuda",
    bbox_strategy: str = "detect",
) -> InferenceResult:
    """影片 → 事件時間點。

    ``bbox_strategy`` 預設為 ``"detect"``：一般影片不保證已裁切到單一運動員。
    已裁切的片段用 ``"whole_frame"`` 較快（跳過偵測器）。
    """
    from .pose import PoseExtractor  # 延遲匯入：核心層不依賴推論後端

    extractor = PoseExtractor(bbox_strategy=bbox_strategy, device=pose_device)
    sequence = extractor.extract_video(video, progress=True)
    pose = to_canonical(sequence.keypoints, sequence.layout)
    result = predict_pose_sequence(
        model, pose, sequence.fps, sport, device=device
    )
    if result.low_confidence:
        logger.warning(
            "%s 的姿態覆蓋率只有 %.2f，結果不可靠", Path(video).name, result.pose_coverage
        )
    return result
