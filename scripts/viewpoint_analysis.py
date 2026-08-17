"""機位對旋轉量測的影響。

骨盆／軀幹的「角速度」是由連線方向角微分而來，這條連線在 2D 投影下會隨身體轉向
鏡頭而縮短。運動員自己就會轉九十度以上，所以**固定機位不會讓這件事消失**——
只會決定它發生在動作的哪一刻。

兩個資料集互相印證：

GolfDB
    有官方視角標註（down-the-line / face-on / other），1391 段，是現成的對照實驗。
Penn Action
    沒有視角標註。以骨盆的水平位移當代理量——側面拍時跨步是橫向的，位移大。
    這是**推論不是標註**，其效度由「位移 vs 塌陷位置」的相關性支撐。

    python scripts/viewpoint_analysis.py --output-dir docs/figures
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kinetic_chain.analysis import (  # noqa: E402
    CHAIN_LINKS,
    projected_length,
    projection_quality,
    unconstrained_sequence,
)
from kinetic_chain.skeleton import JOINT_INDEX  # noqa: E402

logger = logging.getLogger("viewpoint")

INK = "#15181B"
MUTED = "#5C666D"
LINE = "#DCE0E3"
EXPECTED = tuple(name for name, _ in CHAIN_LINKS)
RANDOM_RATE = 1 / 6  # 三個環節的排列，隨機猜對的機率


def _style(ax) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(True, color=LINE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def pelvis_travel(clip) -> float:
    """骨盆在畫面上的水平位移，以身體尺度為單位。側面機位的跨步是橫向的，位移大。"""
    left, right = JOINT_INDEX["left_hip"], JOINT_INDEX["right_hip"]
    raw = clip.pose[..., :2]
    pelvis = 0.5 * (raw[:, left] + raw[:, right])
    raw_width = np.median(np.linalg.norm(raw[:, left] - raw[:, right], axis=1))
    normalised_width = np.median(projected_length(clip.signals(), "pelvis"))
    if normalised_width < 1e-6:
        return float("nan")
    body = raw_width / normalised_width
    return float((pelvis[:, 0].max() - pelvis[:, 0].min()) / max(body, 1e-6))


def measure(clip, window: tuple[int, int]) -> dict | None:
    """一段片段在給定區間內的投影品質與順序判定。"""
    lo, hi = window
    if hi <= lo:
        return None
    signals = clip.signals()
    quality = projection_quality(signals, "pelvis", window=(lo, hi + 1))
    peaks = unconstrained_sequence(signals, window=(lo, hi + 1))
    order = tuple(sorted(peaks, key=lambda e: peaks[e]))
    return {
        "min_relative": quality.min_relative,
        "collapse_position": quality.collapse_position,
        "usable": quality.usable,
        "order_ok": order == EXPECTED,
        "peak_at_collapse": abs(peaks["pelvis_peak_rotation"] - quality.collapse_frame) <= 2,
        "profile": _profile(projected_length(signals, "pelvis"), lo, hi),
    }


def _profile(lengths: np.ndarray, lo: int, hi: int, bins: int = 21) -> np.ndarray:
    """把區間內的投影長度重取樣成固定長度，才能跨片段平均。"""
    reference = float(lengths.max())
    if reference <= 0 or hi <= lo:
        return np.full(bins, np.nan)
    segment = lengths[lo : hi + 1] / reference
    return np.interp(
        np.linspace(0, 1, bins), np.linspace(0, 1, segment.size), segment
    )


def summarise(groups: dict[str, list[dict]]) -> dict[str, dict]:
    out = {}
    for name, items in groups.items():
        if not items:
            continue
        out[name] = {
            "n": len(items),
            "median_min_relative": float(np.median([i["min_relative"] for i in items])),
            "median_collapse_position": float(
                np.median([i["collapse_position"] for i in items])
            ),
            "peak_at_collapse_rate": float(np.mean([i["peak_at_collapse"] for i in items])),
            "order_ok_rate": float(np.mean([i["order_ok"] for i in items])),
            "usable_rate": float(np.mean([i["usable"] for i in items])),
        }
    return out


def profile_figure(panels: list[tuple[str, str, dict[str, list[dict]]]], path: Path) -> None:
    """投影長度沿著動作進程的變化，一條線一個機位分組。

    這張圖直接回答「塌陷發生在動作的哪一刻」——比任何單一數字都清楚。
    """
    colours = ["#2F6F9F", "#B0762A", "#A63A46", "#4F7A4F"]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 3.8), dpi=170)
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, xlabel, groups) in zip(axes, panels):
        _style(ax)
        x = np.linspace(0, 1, 21)
        for colour, (name, items) in zip(colours, groups.items()):
            if not items:
                continue
            profiles = np.vstack([i["profile"] for i in items])
            median = np.nanmedian(profiles, axis=0)
            ax.plot(x, median, color=colour, linewidth=2.0,
                    label=f"{name}  (n={len(items)})")
        ax.axhline(0.5, color="#AB2E29", linewidth=1.1, linestyle="--")
        ax.annotate("unusable below here", (0.02, 0.46), fontsize=8, color="#AB2E29")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9.5)
        ax.set_ylabel("projected hip-line length\n(relative to clip max)",
                      color=MUTED, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=10)
        legend = ax.legend(loc="lower left", frameon=False, fontsize=8.5)
        for text in legend.get_texts():
            text.set_color(MUTED)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golfdb-annotations", type=Path, default=Path("data/raw/golfDB.pkl"))
    parser.add_argument("--golfdb-cache", type=Path, default=Path("data/cache/golfdb_pose"))
    parser.add_argument("--pennaction-root", type=Path, default=Path("data/raw/Penn_Action"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--json", type=Path, default=Path("runs/viewpoint_analysis.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import pandas as pd

    from kinetic_chain.datasets import golfdb, pennaction

    results: dict = {}

    # ---------------------------------------------------------------- GolfDB
    meta = pd.read_pickle(args.golfdb_annotations).set_index("id")
    golf_groups: dict[str, list[dict]] = {"face-on": [], "other": [], "down-the-line": []}
    for clip in golfdb.load(args.golfdb_annotations, args.golfdb_cache):
        top = clip.events.get("loading_peak")
        impact = clip.events.get("release_impact")
        if top is None or impact is None:
            continue
        record = measure(clip, (top, impact))
        if record is None:
            continue
        view = str(meta.loc[int(clip.clip_id.split("/")[-1]), "view"])
        golf_groups.setdefault(view, []).append(record)
    results["golfdb_by_labelled_view"] = summarise(golf_groups)

    # ----------------------------------------------------------- Penn Action
    clips = pennaction.load(args.pennaction_root, sports=["baseball_pitch"])
    travels = {c.clip_id: pelvis_travel(c) for c in clips}
    finite = np.array([v for v in travels.values() if np.isfinite(v)])
    low, high = np.percentile(finite, [33, 66])

    pitch_groups: dict[str, list[dict]] = {
        "front / back (low travel)": [],
        "intermediate": [],
        "side (high travel)": [],
    }
    for clip in clips:
        contact = clip.events.get("stride_foot_contact", 0)
        release = clip.events.get("release_impact")
        if release is None:
            continue
        record = measure(clip, (contact, release))
        if record is None:
            continue
        travel = travels[clip.clip_id]
        name = (
            "front / back (low travel)" if travel < low
            else "side (high travel)" if travel >= high
            else "intermediate"
        )
        record["travel"] = travel
        pitch_groups[name].append(record)
    results["pennaction_pitch_by_travel"] = summarise(pitch_groups)

    every = [r for items in pitch_groups.values() for r in items]
    results["pennaction_travel_vs_collapse_r"] = float(
        np.corrcoef([r["travel"] for r in every], [r["collapse_position"] for r in every])[0, 1]
    )
    results["random_baseline"] = RANDOM_RATE
    results["travel_terciles"] = {"low": float(low), "high": float(high)}

    profile_figure(
        [
            ("GolfDB, labelled view (downswing)", "top of backswing → impact", golf_groups),
            ("Penn Action pitch, inferred view", "foot contact → release", pitch_groups),
        ],
        args.output_dir / "viewpoint_profile.png",
    )

    for key in ("golfdb_by_labelled_view", "pennaction_pitch_by_travel"):
        logger.info("=== %s ===", key)
        for name, stats in results[key].items():
            logger.info(
                "  %-28s n=%3d  最短 %.2f  塌陷位置 %.2f  峰值落在塌陷 %.0f%%  順序成立 %.0f%%",
                name, stats["n"], stats["median_min_relative"],
                stats["median_collapse_position"],
                stats["peak_at_collapse_rate"] * 100, stats["order_ok_rate"] * 100,
            )
    logger.info("位移 vs 塌陷位置 r = %+.3f", results["pennaction_travel_vs_collapse_r"])

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("→ %s", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
