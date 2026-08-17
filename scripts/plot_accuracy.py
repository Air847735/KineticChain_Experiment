"""每個事件的誤差分布圖。

PCE 是「有沒有落在容忍度內」的通過率，容忍度依片段長度換算，看不出實際誤差幾格。
這張圖直接畫「誤差 ≤1 / ≤2 / ≤3 格的比例」，是判斷「這東西夠不夠用」最直接的依據。

輸出不含人物影像，可以進版控。

    python scripts/plot_accuracy.py --checkpoint runs/golf/model.pt \\
        --sport golf_swing --val-fold 1 --output docs/figures/golf_accuracy.png
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
from kinetic_chain.train import load_checkpoint  # noqa: E402

logger = logging.getLogger("plot_accuracy")

INK = "#15181B"
MUTED = "#5C666D"
LINE = "#DCE0E3"
BANDS = ((1, "#1C7038"), (2, "#7FA05C"), (3, "#C9A227"))


def figure(events, deltas: dict[str, np.ndarray], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 0.46 * len(events) + 1.9), dpi=170)
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(True, axis="x", color=LINE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    positions = np.arange(len(events))[::-1]
    for tol, colour in reversed(BANDS):
        share = [float(np.mean(deltas[e] <= tol) * 100) for e in events]
        ax.barh(positions, share, height=0.62, color=colour, alpha=0.9,
                label=f"within {tol} frame" + ("s" if tol > 1 else ""), zorder=3 - tol)

    for y, event in zip(positions, events):
        within = float(np.mean(deltas[event] <= 1) * 100)
        ax.annotate(f"{within:.0f}%", (within, y), textcoords="offset points",
                    xytext=(5, 0), va="center", fontsize=8.5, color=INK)

    ax.set_yticks(positions)
    ax.set_yticklabels([e.replace("golf_", "").replace("_", " ") for e in events],
                       fontsize=9)
    ax.set_xlim(0, 108)
    ax.set_xlabel("% of clips within tolerance", color=MUTED, fontsize=9.5)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=10)
    legend = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(MUTED)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sport", default="golf_swing")
    parser.add_argument("--val-fold", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--golfdb-annotations", type=Path, default=Path("data/raw/golfDB.pkl"))
    parser.add_argument("--golfdb-cache", type=Path, default=Path("data/cache/golfdb_pose"))
    parser.add_argument("--pennaction-root", type=Path, default=Path("data/raw/Penn_Action"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.sport == "golf_swing":
        from kinetic_chain.datasets import golfdb

        clips = golfdb.load(args.golfdb_annotations, args.golfdb_cache)
        source = "GolfDB, human labels"
    else:
        from kinetic_chain.datasets import pennaction

        clips = pennaction.load(args.pennaction_root, sports=[args.sport])
        source = "Penn Action, weak labels"

    _, val_clips = split_clips(
        clips, val_fraction=args.val_fraction, val_fold=args.val_fold
    )
    model = load_checkpoint(args.checkpoint, device=args.device)
    predictions = predict_clips(model, val_clips, device=args.device)

    events = [e for e in get_sport(args.sport).events
              if any(e in c.events for c in val_clips)]
    deltas = {
        e: np.array([abs(p[e] - c.events[e])
                     for c, p in zip(val_clips, predictions) if e in c.events and e in p])
        for e in events
    }
    events = [e for e in events if deltas[e].size]

    figure(
        events, deltas,
        f"{args.sport} - {source} - {len(val_clips)} clips",
        args.output,
    )
    logger.info("→ %s", args.output)
    for e in events:
        logger.info("  %-22s <=1 frame %5.1f%%  median %.0f",
                    e, np.mean(deltas[e] <= 1) * 100, np.median(deltas[e]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
