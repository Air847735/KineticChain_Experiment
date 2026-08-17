"""把動力鏈畫成圖：三條速度曲線的峰值時序。

分數說不出「力量傳遞得順不順」。真正該看的是骨盆、軀幹、上肢三條速度曲線在時間上
怎麼排列——近端先達峰、遠端後達峰，才是動力鏈成立的樣子。

產出三張圖（都不含人物影像，可以進版控）：

``chain_trace_*.png``   單次動作的三條曲線與峰值
``chain_timeline.png``  全體片段的事件時間分布（正規化到投球期）
``chain_separation.png`` 分離時間分布，附上取樣解析度的界線
``projection_artifact.png`` 髖線投影長度與骨盆角速度的關係——為什麼骨盆峰值不可信

    python scripts/visualize_chain.py --checkpoint runs/pitch/model.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kinetic_chain.analysis import (  # noqa: E402
    CHAIN_LINKS,
    EXTENSION_CHAIN_LINKS,
    unconstrained_sequence,
)
from kinetic_chain.data import split_clips  # noqa: E402
from kinetic_chain.evaluate import predict_clips  # noqa: E402
from kinetic_chain.events import get_sport  # noqa: E402
from kinetic_chain.train import load_checkpoint  # noqa: E402

logger = logging.getLogger("visualize_chain")

INK = "#15181B"
MUTED = "#5C666D"
LINE = "#DCE0E3"
ROTATION_LABELS = {
    "pelvis_peak_rotation": ("Pelvis", "#2F6F9F"),
    "torso_peak_rotation": ("Torso", "#B0762A"),
    "arm_peak_velocity": ("Arm / wrist", "#A63A46"),
}
EXTENSION_LABELS = {
    "hip_extension_peak": ("Hip extension", "#2F6F9F"),
    "knee_extension_peak": ("Knee extension", "#B0762A"),
    "arm_peak_velocity": ("Arm / wrist", "#A63A46"),
}
CHAINS = {
    "rotation": (CHAIN_LINKS, ROTATION_LABELS),
    "extension": (EXTENSION_CHAIN_LINKS, EXTENSION_LABELS),
}
LINKS = ROTATION_LABELS
SIGNAL_OF = dict(CHAIN_LINKS)


def _style(ax) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(True, color=LINE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def trace_figure(clip, events: dict[str, int], path: Path,
                 chain: str = "rotation",
                 window_events: tuple[str, str] = ("stride_foot_contact", "release_impact"),
                 window_label: str = "Acceleration phase") -> None:
    """單次動作：三條速度曲線疊在一起，峰值標出來。"""
    links, labels = CHAINS[chain]
    signal_of = dict(links)
    signals = clip.signals().signals
    n = clip.num_frames
    frames = np.arange(n)

    fig, ax = plt.subplots(figsize=(9.5, 4.0), dpi=170)
    _style(ax)

    lo = events.get(window_events[0])
    hi = events.get(window_events[1])
    if lo is not None and hi is not None and hi > lo:
        ax.axvspan(lo, hi, color="#F0D9A8", alpha=0.35, zorder=0, label=window_label)

    peaks = unconstrained_sequence(
        clip.signals(), links=links,
        window=(lo, hi + 1) if lo is not None and hi is not None and hi > lo else None,
    )
    for event, (label, colour) in labels.items():
        values = np.asarray(signals[signal_of[event]], dtype=float)[:n]
        peak = values.max()
        if peak > 0:
            values = values / peak
        ax.plot(frames, values, color=colour, linewidth=1.9, label=label, zorder=3)
        at = peaks[event]
        ax.plot([at], [values[at]], "o", color=colour, markersize=6,
                markeredgecolor="white", markeredgewidth=1.2, zorder=4)
        ax.annotate(f"f{at}", (at, values[at]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5, color=colour)

    order = " > ".join(labels[e][0] for e in sorted(peaks, key=lambda e: peaks[e]))
    expected = tuple(name for name, _ in links)
    ok = tuple(sorted(peaks, key=lambda e: peaks[e])) == expected

    ax.set_xlabel("frame", color=MUTED, fontsize=9.5)
    ax.set_ylabel("speed (normalised to own peak)", color=MUTED, fontsize=9.5)
    ax.set_ylim(-0.04, 1.22)
    ax.set_xlim(0, n - 1)
    ax.set_title(
        f"{clip.clip_id}    peak order: {order}    "
        + ("proximal-to-distal OK" if ok else "OUT OF ORDER"),
        color=INK if ok else "#AB2E29", fontsize=10.5, loc="left", pad=10,
    )
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9, ncols=4)
    for text in legend.get_texts():
        text.set_color(MUTED)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def timeline_figure(analyses, path: Path) -> None:
    """全體片段：每個事件在正規化時間軸上的分布。"""
    order = [
        "address", "loading_start", "loading_peak", "stride_foot_contact",
        "pelvis_peak_rotation", "torso_peak_rotation", "arm_peak_velocity",
        "release_impact", "follow_through_mid", "finish",
    ]
    data = [
        [a.timeline[e] for a in analyses if e in a.timeline]
        for e in order
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=170)
    _style(ax)
    positions = list(range(len(order)))[::-1]
    box = ax.boxplot(
        data, positions=positions, orientation="horizontal", widths=0.6, showfliers=False,
        patch_artist=True, medianprops=dict(color=INK, linewidth=1.6),
        whiskerprops=dict(color=MUTED, linewidth=1.0),
        capprops=dict(color=MUTED, linewidth=1.0),
    )
    for patch, event in zip(box["boxes"], order):
        colour = LINKS[event][1] if event in LINKS else "#8FA0A8"
        patch.set_facecolor(colour)
        patch.set_alpha(0.35)
        patch.set_edgecolor(colour)

    ax.axvline(0, color=MUTED, linewidth=1.0, linestyle="--")
    ax.axvline(100, color=MUTED, linewidth=1.0, linestyle="--")
    ax.set_yticks(positions)
    ax.set_yticklabels([e.replace("_", " ") for e in order], fontsize=9)
    ax.set_xlabel("% of throw  (0 = first event, 100 = release)", color=MUTED, fontsize=9.5)
    ax.set_title(f"Event timing across {len(analyses)} pitches", color=INK,
                 fontsize=10.5, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def separation_figure(clips, path: Path) -> None:
    """分離時間分布，並畫出取樣解析度的界線。"""
    pelvis_torso, torso_arm = [], []
    for clip in clips:
        lo = clip.events.get("stride_foot_contact", 0)
        hi = clip.events.get("release_impact", clip.num_frames - 1) + 1
        peaks = unconstrained_sequence(clip.signals(), window=(lo, hi))
        pelvis_torso.append(peaks["torso_peak_rotation"] - peaks["pelvis_peak_rotation"])
        torso_arm.append(peaks["arm_peak_velocity"] - peaks["torso_peak_rotation"])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=170, sharey=True)
    pairs = (
        (axes[0], np.array(pelvis_torso), "Pelvis → Torso", "#2F6F9F"),
        (axes[1], np.array(torso_arm), "Torso → Arm", "#B0762A"),
    )
    for ax, values, title, colour in pairs:
        _style(ax)
        bins = np.arange(values.min() - 0.5, values.max() + 1.5, 1.0)
        ax.hist(values, bins=bins, color=colour, alpha=0.55, edgecolor=colour, linewidth=1.0)
        ax.axvspan(-1, 1, color="#AB2E29", alpha=0.10, zorder=0)
        ax.axvline(0, color="#AB2E29", linewidth=1.2, linestyle="--", zorder=2)
        reversed_pct = float(np.mean(values < 0) * 100)
        ax.set_title(
            f"{title}   median {np.median(values):+.0f} f ({np.median(values) * 33.3:+.0f} ms)"
            f"   reversed {reversed_pct:.0f}%",
            color=INK, fontsize=9.5, loc="left", pad=8,
        )
        ax.set_xlabel("separation (frames @ 30 fps)", color=MUTED, fontsize=9)
    axes[0].set_ylabel("pitches", color=MUTED, fontsize=9)
    fig.suptitle(
        "Red band = ±1 frame. Literature puts pelvis→torso at 20–50 ms = 0.6–1.5 frames here.",
        color=MUTED, fontsize=9, y=1.0, x=0.012, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def projection_artifact_figure(clips, path: Path) -> None:
    """髖線在投影下縮短時，方向角的微分會爆走。

    骨盆／軀幹的「角速度」是由髖線／肩線的方向角微分而來。這條向量在 2D 投影下
    會隨著身體轉向鏡頭而縮短；長度接近零時，``atan2`` 對關鍵點雜訊極度敏感，
    一個像素的抖動就能讓角度跳幾十度，微分後產生尖刺。

    此圖檢驗這件事：把所有片段逐格攤開，看髖線相對長度與骨盆角速度的關係。
    若兩者無關，各組的角速度應該差不多。
    """
    from kinetic_chain.skeleton import JOINT_INDEX

    left, right = JOINT_INDEX["left_hip"], JOINT_INDEX["right_hip"]
    lengths, speeds = [], []
    for clip in clips:
        signals = clip.signals()
        pose = signals.pose
        hip = np.linalg.norm(pose[:, right, :2] - pose[:, left, :2], axis=1)
        speed = np.asarray(signals.signals["pelvis_angular_speed"], dtype=float)
        if np.median(hip) <= 0 or speed.max() <= 0:
            continue
        lengths.append(hip / np.median(hip))
        speeds.append(speed / speed.max())
    lengths = np.concatenate(lengths)
    speeds = np.concatenate(speeds)

    edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4])
    centres, medians, share, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (lengths >= lo) & (lengths < hi)
        if mask.sum() < 20:
            continue
        centres.append((lo + hi) / 2)
        medians.append(float(np.median(speeds[mask])))
        share.append(float(np.mean(speeds[mask] > 0.5) * 100))
        counts.append(int(mask.sum()))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=170)
    _style(axes[0])
    _style(axes[1])

    axes[0].bar(centres, medians, width=0.17, color="#2F6F9F", alpha=0.75,
                edgecolor="#2F6F9F")
    for x, y, n in zip(centres, medians, counts):
        axes[0].annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 4),
                         ha="center", fontsize=7.5, color=MUTED)
    axes[0].set_ylabel("median pelvis angular speed\n(relative to clip peak)",
                       color=MUTED, fontsize=9)

    axes[1].bar(centres, share, width=0.17, color="#A63A46", alpha=0.75,
                edgecolor="#A63A46")
    axes[1].set_ylabel("% of frames above 0.5 × clip peak", color=MUTED, fontsize=9)

    r = float(np.corrcoef(lengths, speeds)[0, 1])
    for ax in axes:
        ax.set_xlabel("projected hip-line length (relative to clip median)",
                      color=MUTED, fontsize=9)
    fig.suptitle(
        f"Pelvis 'rotation speed' spikes exactly when the hip line collapses in projection"
        f"   (per-frame r = {r:+.3f}, {len(lengths)} frames)",
        color=INK, fontsize=9.5, y=1.0, x=0.012, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/pitch/model.pt"))
    parser.add_argument("--pennaction-root", type=Path, default=Path("data/raw/Penn_Action"))
    parser.add_argument("--sport", default="baseball_pitch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--traces", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/chain"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from kinetic_chain.analysis import analyse
    from kinetic_chain.datasets import pennaction

    clips = pennaction.load(args.pennaction_root, sports=[args.sport])
    spec = get_sport(args.sport)
    model = load_checkpoint(args.checkpoint, device=args.device)
    _, val_clips = split_clips(clips, val_fraction=0.2, seed=0)
    predictions = predict_clips(model, val_clips, device=args.device)

    detected = [analyse(c.clip_id, spec, p, c.fps) for c, p in zip(val_clips, predictions)]
    timeline_figure(detected, args.output_dir / "chain_timeline.png")
    separation_figure(clips, args.output_dir / "chain_separation.png")
    projection_artifact_figure(clips, args.output_dir / "projection_artifact.png")
    logger.info("時間分布與分離時間圖 → %s", args.output_dir)

    # 挑順序成立與不成立的各幾段，不只挑好看的
    expected = tuple(name for name, _ in CHAIN_LINKS)
    ok_clips, bad_clips = [], []
    for clip in val_clips:
        peaks = unconstrained_sequence(clip.signals())
        (ok_clips if tuple(sorted(peaks, key=lambda e: peaks[e])) == expected else bad_clips).append(clip)

    chosen = ok_clips[: max(args.traces - 1, 1)] + bad_clips[:1]
    for clip, prediction in zip(val_clips, predictions):
        if clip in chosen:
            name = clip.clip_id.replace("/", "_")
            trace_figure(clip, prediction, args.output_dir / f"chain_trace_{name}.png")
            logger.info("曲線圖 %s", name)
    logger.info("順序成立 %d 段 / 不成立 %d 段（驗證集 %d 段）",
                len(ok_clips), len(bad_clips), len(val_clips))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
