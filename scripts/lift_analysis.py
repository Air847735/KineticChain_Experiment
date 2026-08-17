"""舉重（挺舉）的動力鏈分析。

跟投球的關鍵差別有兩個，都不是實作細節而是力學本質：

**動力鏈的型態不同。** 投球是旋轉型（骨盆→軀幹→上肢的方向角旋轉），舉重是伸展型
（髖→膝的三點夾角伸展）。前者用跨身體連線的方向角，在 2D 投影下會退化；
後者用三點夾角，兩條邊都是長節段，不會塌。

**需要的機位相反。** 旋轉發生在水平面，正面／45° 機位才看得到；伸展發生在矢狀面，
**側面機位才看得到**。正面拍的舉重片段，髖角範圍只剩 169–180°——前後彎曲被完全壓扁。

    python scripts/lift_analysis.py --checkpoint runs/lift/model.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from kinetic_chain.analysis import (
    EXTENSION_CHAIN_LINKS,
    analyse,
    unconstrained_sequence,
)
from kinetic_chain.data import split_clips
from kinetic_chain.evaluate import evaluate_clips, predict_clips
from kinetic_chain.events import get_sport
from kinetic_chain.train import load_checkpoint

logger = logging.getLogger("lift")
SPORT = "clean_and_jerk"
EXPECTED = tuple(name for name, _ in EXTENSION_CHAIN_LINKS)

#: 髖角最小值高於此值時，矢狀面沒有被拍到——彎曲被投影壓扁，伸展量測無效。
#: 實測：明顯正面的片段髖角範圍只有 169–180°；側面片段可低到 60° 以下。
SAGITTAL_VISIBLE_DEG = 120.0


def sagittal_visibility(clip) -> float:
    """該片段的髖角最小值（度）。越小代表越看得見前後彎曲，即越接近側面機位。"""
    return float(np.degrees(clip.signals().signals["hip_angle"].min()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/lift/model.pt"))
    parser.add_argument("--pennaction-root", type=Path, default=Path("data/raw/Penn_Action"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/lift_analysis.json"))
    parser.add_argument("--figure", type=Path, default=Path("docs/figures/chain_comparison.png"))
    parser.add_argument(
        "--rotation-json", type=Path, default=Path("runs/viewpoint_analysis.json"),
        help="旋轉型動力鏈的對照數字，由 scripts/viewpoint_analysis.py 產生",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from kinetic_chain.datasets import pennaction

    clips = pennaction.load(args.pennaction_root, sports=[SPORT])
    spec = get_sport(SPORT)
    results: dict = {"sport": SPORT, "clips": len(clips)}

    # ------------------------------------------------------------------ 機位
    visibility = np.array([sagittal_visibility(c) for c in clips])
    side = visibility < SAGITTAL_VISIBLE_DEG
    results["viewpoint"] = {
        "threshold_deg": SAGITTAL_VISIBLE_DEG,
        "median_hip_min_deg": float(np.median(visibility)),
        "sagittal_visible": int(side.sum()),
        "total": len(clips),
    }
    logger.info(
        "矢狀面可見（髖角最小 <%.0f°）：%d / %d 段，中位 %.0f°",
        SAGITTAL_VISIBLE_DEG, side.sum(), len(clips), np.median(visibility),
    )

    # ---------------------------------------------------------------- 偵測表現
    _, val_clips = split_clips(clips, val_fraction=0.2, seed=args.seed)
    model = load_checkpoint(args.checkpoint, device=args.device)
    reports = evaluate_clips(model, val_clips, device=args.device)
    key = f"{SPORT}/weak"
    results["detection"] = {
        "pce_vs_weak_labels": reports[key].pce,
        "val_clips": reports[key].num_clips,
        "mean_tolerance_frames": reports[key].mean_tolerance,
        "per_event": {
            name: {"pce": s.pce, "median_delta": s.median_delta}
            for name, s in reports[key].per_event.items()
        },
    }
    logger.info("模型 vs 弱標註 PCE %.4f（%d 段）", reports[key].pce, reports[key].num_clips)

    # ---------------------------------------------------------------- 時序指標
    predictions = predict_clips(model, val_clips, device=args.device)
    detected = [analyse(c.clip_id, spec, p, c.fps) for c, p in zip(val_clips, predictions)]
    results["timeline"] = {
        event: {
            "median_percent": float(np.median([a.timeline[event] for a in detected
                                               if event in a.timeline])),
            "n": int(sum(event in a.timeline for a in detected)),
        }
        for event in spec.events
    }

    # ------------------------------------------------- 伸展型動力鏈（無約束量測）
    for scope, only_side in (("all", False), ("sagittal_visible", True)):
        orders: dict[str, int] = {}
        widths = {name: [] for name, _ in EXTENSION_CHAIN_LINKS}
        subset = [c for c, v in zip(clips, visibility)
                  if not only_side or v < SAGITTAL_VISIBLE_DEG]
        for clip in subset:
            signals = clip.signals()
            lo = clip.events.get("clean_liftoff", 0)
            hi = clip.events.get("clean_catch", clip.num_frames - 1) + 1
            peaks = unconstrained_sequence(signals, window=(lo, hi),
                                           links=EXTENSION_CHAIN_LINKS)
            key_order = " → ".join(
                e.replace("_extension_peak", "").replace("_peak_velocity", "")
                for e in sorted(peaks, key=lambda e: peaks[e])
            )
            orders[key_order] = orders.get(key_order, 0) + 1
            for name, signal in EXTENSION_CHAIN_LINKS:
                values = np.asarray(signals.signals[signal], dtype=float)
                widths[name].append(_fwhm(values))
        total = max(len(subset), 1)
        results[f"extension_chain_{scope}"] = {
            "n": len(subset),
            "proximal_to_distal_rate": orders.get("hip → knee → arm", 0) / total,
            "orders": {k: v / total for k, v in sorted(orders.items(), key=lambda kv: -kv[1])},
            "median_peak_width_frames": {k: float(np.median(v)) for k, v in widths.items()},
        }
        logger.info(
            "伸展序列（%s，n=%d）：髖→膝→上肢成立 %.1f%%",
            scope, len(subset),
            results[f"extension_chain_{scope}"]["proximal_to_distal_rate"] * 100,
        )

    results["random_baseline"] = 1 / 6

    if args.figure is not None:
        _comparison_figure(results, args.figure)
        logger.info("→ %s", args.figure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("→ %s", args.output)
    return 0


def _comparison_figure(results: dict, path: Path) -> None:
    """旋轉型 vs 伸展型：同一套流程、同一個指標，兩種動力鏈的可測性差多少。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rotation = None
    source = Path("runs/viewpoint_analysis.json")
    if source.is_file():
        data = json.loads(source.read_text(encoding="utf-8"))
        rotation = data.get("golfdb_by_labelled_view", {}).get("face-on", {})

    bars = [
        ("Rotation chain\n(golf, face-on:\nbest available view)",
         rotation.get("order_ok_rate", float("nan")) if rotation else float("nan"),
         rotation.get("n", 0) if rotation else 0, "#A63A46"),
        ("Extension chain\n(weightlifting,\nsagittal visible)",
         results["extension_chain_sagittal_visible"]["proximal_to_distal_rate"],
         results["extension_chain_sagittal_visible"]["n"], "#2F6F9F"),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=170)
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#DCE0E3")
    ax.tick_params(colors="#5C666D", labelsize=9, length=3)
    ax.grid(True, axis="y", color="#DCE0E3", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    positions = np.arange(len(bars))
    ax.bar(positions, [b[1] * 100 for b in bars], width=0.5,
           color=[b[3] for b in bars], alpha=0.85)
    for x, (_, rate, n, _c) in zip(positions, bars):
        if np.isfinite(rate):
            ax.annotate(f"{rate * 100:.0f}%\nn={n}", (x, rate * 100),
                        textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=10, color="#15181B")
    ax.axhline(100 / 6, color="#8A5C0F", linewidth=1.4, linestyle="--")
    ax.annotate("random baseline 16.7%", (-0.42, 100 / 6 + 1.6),
                ha="left", fontsize=9, color="#8A5C0F")
    ax.set_xticks(positions)
    ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.set_ylabel("proximal-to-distal order holds (%)", color="#5C666D", fontsize=9.5)
    ax.set_ylim(0, 60)
    ax.set_title("Same pipeline, same metric, two kinds of kinetic chain",
                 color="#15181B", fontsize=10.5, loc="left", pad=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def _fwhm(values: np.ndarray) -> int:
    """峰的半高全寬。真實的節段運動應有十幾格寬，單格尖刺是雜訊。"""
    at = int(values.argmax())
    threshold = values[at] / 2
    lo = at
    while lo > 0 and values[lo - 1] > threshold:
        lo -= 1
    hi = at
    while hi < values.size - 1 and values[hi + 1] > threshold:
        hi += 1
    return hi - lo + 1


if __name__ == "__main__":
    raise SystemExit(main())
