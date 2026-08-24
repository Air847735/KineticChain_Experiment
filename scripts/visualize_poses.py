"""把偵測到的每個事件畫成骨架姿勢，用眼睛檢查那一格的動作對不對。

`scripts/visualize.py` 截的是原始影格，含可辨識的運動員，只能留在本機。這支腳本
只用關節座標畫火柴人，**不含任何影像像素**，因此可以進版控、可以放進報告。

畫的是 `features.compute` 正規化後的姿勢，也就是**模型實際看到的東西**，不是原始像素
座標。Penn Action 有 15% 的關節標為不可見，這些關節在特徵計算時被沿時間軸內插補值；
圖上以空心點標出，免得把補出來的位置當成量到的位置。

代價是看不到球棒與球，判斷「觸球」這類事件仍需回頭看原始影格。

    python scripts/visualize_poses.py --sport baseball_swing --checkpoint runs/bat/model.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kinetic_chain.data import split_clips  # noqa: E402
from kinetic_chain.evaluate import predict_clips  # noqa: E402
from kinetic_chain.events import get_sport  # noqa: E402
from kinetic_chain.metrics import tolerance  # noqa: E402
from kinetic_chain.features import MIN_CONFIDENCE  # noqa: E402
from kinetic_chain.skeleton import FLIP_PAIRS, JOINT_INDEX, NUM_JOINTS  # noqa: E402
from kinetic_chain.train import load_checkpoint  # noqa: E402

logger = logging.getLogger("visualize_poses")

#: canonical 13 點的連線。頭只連到兩肩中點，沒有頸部關節。
BONES: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)

INK = "#15181B"
MUTED = "#5C666D"
LINE = "#DCE0E3"
HIT = "#1C7038"
MISS = "#AB2E29"
LEFT = "#2F6F9F"
RIGHT = "#B0762A"
TRUNK = "#4A5560"

#: 鏡射後的關節順序，用來把可見度旗標換邊。
FLIP_ORDER = np.arange(NUM_JOINTS)
for _a, _b in FLIP_PAIRS:
    FLIP_ORDER[_a], FLIP_ORDER[_b] = _b, _a

ZH = {
    "address": "準備",
    "loading_start": "前腳離地",
    "stride_foot_contact": "前腳著地",
    "loading_peak": "最大分離",
    "pelvis_peak_rotation": "骨盆峰值",
    "torso_peak_rotation": "軀幹峰值",
    "arm_peak_velocity": "上肢峰值",
    "release_impact": "觸球／出手",
    "follow_through_mid": "隨勢中點",
    "finish": "結束",
}


def _bone_colour(first: str, second: str) -> str:
    if first.startswith("left_") and second.startswith("left_"):
        return LEFT
    if first.startswith("right_") and second.startswith("right_"):
        return RIGHT
    return TRUNK


def draw_pose(
    ax,
    pose: np.ndarray,
    measured: np.ndarray,
    frame: int,
    bounds: tuple[float, float, float, float],
) -> None:
    """畫一格的火柴人。

    Parameters
    ----------
    pose:
        ``(T, 13, 2)`` 正規化後的姿勢（``PoseSignals.pose``），y 軸向上為正。
    measured:
        ``(T, 13)`` 布林陣列，True 表示該關節在該影格是量到的、不是內插補出來的。
    """
    joints = pose[frame]
    solid = measured[frame]
    for first, second in BONES:
        i, j = JOINT_INDEX[first], JOINT_INDEX[second]
        a, b = joints[i], joints[j]
        # 兩端都量到才畫實線；有一端是補值就畫虛線，別讓補出來的骨頭看起來像量到的
        both = solid[i] and solid[j]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=_bone_colour(first, second),
                linewidth=2.0 if both else 1.2, linestyle="-" if both else (0, (2, 2)),
                alpha=1.0 if both else 0.55, solid_capstyle="round")
    ax.scatter(joints[solid, 0], joints[solid, 1], s=9, color=INK, zorder=3)
    ax.scatter(joints[~solid, 0], joints[~solid, 1], s=11, facecolor="white",
               edgecolor=MUTED, linewidth=0.9, zorder=3)

    left, right, bottom, top = bounds
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color(LINE)


def clip_bounds(pose: np.ndarray, margin: float = 0.12) -> tuple[float, float, float, float]:
    """整段共用的視野，這樣各格之間的位移看得出來。"""
    x, y = pose[:, :, 0].ravel(), pose[:, :, 1].ravel()
    width = max(x.max() - x.min(), y.max() - y.min())
    cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    half = width * (0.5 + margin)
    return cx - half, cx + half, cy - half, cy + half


def clip_figure(clip, prediction: dict[str, int], spec, path: Path) -> None:
    """一段影片：每個事件一格，標出偵測影格與與弱標註的差。"""
    events = [e for e in spec.events if e in prediction]
    tol = tolerance(clip.events, order=spec.events)
    signals = clip.signals()
    pose = signals.pose
    measured = np.asarray(clip.pose[..., 2]) >= MIN_CONFIDENCE
    if signals.flipped:
        # 特徵計算會左右鏡射，可見度旗標得跟著換邊，否則實線虛線會標到另一側
        measured = measured[:, FLIP_ORDER]
    bounds = clip_bounds(pose)

    columns = min(len(events), 5)
    rows = (len(events) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(2.05 * columns, 2.5 * rows), dpi=170)
    axes = np.atleast_1d(axes).ravel()

    for ax, event in zip(axes, events):
        frame = prediction[event]
        truth = clip.events.get(event)
        draw_pose(ax, pose, measured, frame, bounds)
        delta = None if truth is None else frame - truth
        hit = delta is not None and abs(delta) <= tol
        colour = INK if delta is None else (HIT if hit else MISS)
        suffix = "" if delta is None else f"  ({delta:+d})"
        ax.set_title(f"{ZH.get(event, event)}  f{frame}{suffix}",
                     fontsize=8.5, color=colour, pad=5)
    for ax in axes[len(events):]:
        ax.axis("off")

    fig.suptitle(
        f"{clip.clip_id}    {clip.num_frames} 影格 @ {clip.fps:.0f} fps    "
        f"容忍度 ±{tol} 格    括號為與弱標註的差    空心點與虛線為內插補值",
        fontsize=9.5, color=MUTED, x=0.012, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="baseball_swing")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/bat/model.pt"))
    parser.add_argument("--pennaction-root", type=Path, default=Path("data/raw/Penn_Action"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/poses"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # 標題含中文；沒有 CJK 字型時 matplotlib 會靜默畫成空框，所以明確檢查。
    plt.rcParams["font.family"] = ["Noto Serif CJK TC", "Noto Sans Mono CJK TC", "DejaVu Sans"]
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    if not available & set(plt.rcParams["font.family"][:2]):
        raise SystemExit("找不到中文字型，圖上的標題會變成空框；請安裝 Noto CJK")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from kinetic_chain.datasets import pennaction

    clips = pennaction.load(args.pennaction_root, sports=[args.sport])
    spec = get_sport(args.sport)
    model = load_checkpoint(args.checkpoint, device=args.device)
    _, val_clips = split_clips(clips, val_fraction=0.2, seed=0)
    predictions = predict_clips(model, val_clips, device=args.device)

    # 好的與差的都要看，否則只是在挑好看的
    scored = []
    for clip, prediction in zip(val_clips, predictions):
        tol = tolerance(clip.events, order=spec.events)
        hits = sum(
            abs(prediction[e] - clip.events[e]) <= tol
            for e in spec.events
            if e in prediction and e in clip.events
        )
        scored.append((hits, clip, prediction))
    scored.sort(key=lambda item: -item[0])
    chosen = scored[: max(args.count - 1, 1)] + scored[-1:]

    for hits, clip, prediction in chosen:
        name = clip.clip_id.replace("/", "_")
        path = args.output_dir / f"poses_{name}.png"
        clip_figure(clip, prediction, spec, path)
        logger.info("%s 命中 %d/%d → %s", clip.clip_id, hits, len(spec.events), path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
