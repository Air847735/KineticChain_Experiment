"""影片 → 2D 關鍵點序列。

用 RTMPose（rtmlib 的 ONNX 推論封裝）。本專案不訓練姿態模型，只需要推論，
因此不走 mmpose——`mmcv` 與 torch/CUDA 的版本綁定在同一台機器上要與其他專案
共存太麻煩，而 rtmlib 用的是 OpenMMLab 官方匯出的同一批權重。

這是**唯一**匯入 ``rtmlib`` 與 ``cv2`` 的模組（另有 :mod:`kinetic_chain.infer`
呼叫它）。核心層不得依賴推論後端，模型必須能在只有 numpy 陣列時訓練與測試。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

from .errors import PoseExtractionError

logger = logging.getLogger(__name__)

#: rtmlib 的權重快取，跨 conda 環境共用，不會重複下載。
RTMPOSE_BODY7_M = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)

BBoxStrategy = Literal["whole_frame", "detect"]


@dataclass(frozen=True)
class PoseSequence:
    """一支影片的關鍵點序列。"""

    keypoints: np.ndarray  # (T, 17, 3)，COCO-17 布局，最後一維為 x, y, confidence
    fps: float
    width: int
    height: int
    layout: str = "coco17"

    @property
    def num_frames(self) -> int:
        return int(self.keypoints.shape[0])


def _read_frames(path: Path) -> tuple[Iterator[np.ndarray], float, int, int, int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PoseExtractionError(f"無法開啟影片：{path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def frames() -> Iterator[np.ndarray]:
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame
        finally:
            capture.release()

    return frames(), fps, width, height, count


class PoseExtractor:
    """RTMPose 推論器。建構成本高（載入 ONNX），要重複使用同一個實例。

    Parameters
    ----------
    bbox_strategy:
        ``"whole_frame"``
            整張畫面當作偵測框，完全跳過人體偵測器。適用於已裁切到單一運動員的
            片段（GolfDB 的 ``videos_160`` 就是），既快又不會漏偵測。
        ``"detect"``
            先跑 YOLOX 偵測，取面積最大的框。適用於未裁切的影片。
    device:
        ``"cuda"`` 或 ``"cpu"``。
    """

    def __init__(
        self,
        *,
        bbox_strategy: BBoxStrategy = "whole_frame",
        device: str = "cuda",
        backend: str = "onnxruntime",
    ) -> None:
        self.bbox_strategy = bbox_strategy
        self.device = device
        try:
            from rtmlib import Body, RTMPose
        except ImportError as exc:  # pragma: no cover - 相依缺失
            raise PoseExtractionError(
                "需要 rtmlib 才能抽取姿態；安裝方式：pip install -e '.[pose]'"
            ) from exc

        if bbox_strategy == "whole_frame":
            self._pose = RTMPose(
                onnx_model=RTMPOSE_BODY7_M,
                model_input_size=(192, 256),
                backend=backend,
                device=device,
            )
            self._detector = None
        else:
            self._detector = Body(mode="balanced", backend=backend, device=device)
            self._pose = None

    def _bboxes(self, frame: np.ndarray) -> list[list[float]]:
        height, width = frame.shape[:2]
        return [[0.0, 0.0, float(width), float(height)]]

    def candidates(self, frame: np.ndarray) -> np.ndarray:
        """單張影像 → ``(N, 17, 3)``，畫面中偵測到的所有人。"""
        if self._pose is not None:
            keypoints, scores = self._pose(frame, bboxes=self._bboxes(frame))
        else:
            keypoints, scores = self._detector(frame)

        keypoints = np.asarray(keypoints, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if keypoints.size == 0:
            return np.zeros((0, 17, 3), dtype=np.float32)
        if keypoints.ndim == 2:      # 單人時 rtmlib 可能少一個維度
            keypoints = keypoints[None]
            scores = scores[None]
        return np.concatenate([keypoints, scores[..., None]], axis=-1).astype(np.float32)

    @staticmethod
    def _extent(person: np.ndarray) -> float:
        """關鍵點外接框的對角線長度，當作「這個人在畫面上多大」。"""
        visible = person[person[:, 2] > 0.3, :2]
        if visible.shape[0] < 2:
            return 0.0
        span = visible.max(axis=0) - visible.min(axis=0)
        return float(np.hypot(*span))

    def _select(self, people: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        """從候選中挑出目標運動員。

        單純取「信心最高」在乾淨的裁切片段上沒問題，但在真實場景會失效：健身房或
        比賽背景常有其他人，他們的姿態一樣清晰、信心一樣高。實測一支自備影片時，
        骨盆位置逐格最大跳動 810 px、身高變異係數 0.67——偵測器整段都在人之間跳。

        改成兩個準則：**畫面上最大的人**（目標運動員離鏡頭最近），加上**時間連續性**
        （與前一格選中的人位置接近）。兩者相乘，避免單一準則被短暫遮擋帶走。
        """
        if people.shape[0] == 0:
            return np.zeros((17, 3), dtype=np.float32)
        if people.shape[0] == 1:
            return people[0]

        extents = np.array([self._extent(p) for p in people])
        scale = float(extents.max()) or 1.0
        weights = extents / scale

        if previous is not None and self._extent(previous) > 0:
            anchor = np.median(previous[previous[:, 2] > 0.3, :2], axis=0)
            centres = np.array([
                np.median(p[p[:, 2] > 0.3, :2], axis=0)
                if (p[:, 2] > 0.3).any() else np.array([np.inf, np.inf])
                for p in people
            ])
            distance = np.linalg.norm(centres - anchor, axis=1)
            # 以前一格的身體尺度為單位；距離一個身長時權重降到約 1/2
            weights = weights / (1.0 + distance / max(self._extent(previous), 1.0))

        return people[int(np.argmax(weights))]

    def extract_frame(self, frame: np.ndarray) -> np.ndarray:
        """單張影像 → ``(17, 3)``。找不到人時回傳全 0（信心 0）。

        無狀態，逐格獨立選人。整段影片請用 :meth:`extract_video`，它會維持
        時間連續性。
        """
        return self._select(self.candidates(frame), None)

    def extract_video(self, path: Path | str, *, progress: bool = False) -> PoseSequence:
        """整支影片 → :class:`PoseSequence`。"""
        path = Path(path)
        if not path.is_file():
            raise PoseExtractionError(f"找不到影片：{path}")

        frames, fps, width, height, count = _read_frames(path)
        if progress:
            try:
                from tqdm import tqdm

                frames = tqdm(frames, total=count or None, desc=path.name, leave=False)
            except ImportError:
                pass

        keypoints = []
        previous: np.ndarray | None = None
        for frame in frames:
            person = self._select(self.candidates(frame), previous)
            keypoints.append(person)
            if self._extent(person) > 0:
                previous = person
        if not keypoints:
            raise PoseExtractionError(f"影片沒有任何可解碼的影格：{path}")

        return PoseSequence(
            keypoints=np.stack(keypoints),
            fps=fps,
            width=width,
            height=height,
        )


def save_sequence(sequence: PoseSequence, path: Path | str) -> None:
    """存成 ``.npz``。姿態抽取比訓練慢得多，一定要快取。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keypoints=sequence.keypoints,
        fps=np.float32(sequence.fps),
        width=np.int32(sequence.width),
        height=np.int32(sequence.height),
        layout=np.array(sequence.layout),
    )


def load_sequence(path: Path | str) -> PoseSequence:
    """讀回 :func:`save_sequence` 存下的 ``.npz``。"""
    path = Path(path)
    if not path.is_file():
        raise PoseExtractionError(f"找不到姿態快取：{path}")
    with np.load(path, allow_pickle=False) as data:
        return PoseSequence(
            keypoints=data["keypoints"].astype(np.float32),
            fps=float(data["fps"]),
            width=int(data["width"]),
            height=int(data["height"]),
            layout=str(data["layout"]),
        )
