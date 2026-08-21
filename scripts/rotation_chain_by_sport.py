"""六個旋轉型運動的近端到遠端序列成立率，只看投影幾何乾淨的片段。

回答一個問題：**30 fps 的 2D 姿態到底量不量得到近端到遠端的時序？**

先前只根據高爾夫回答過一次，答案是「量不到」。那是錯的——高爾夫的下桿窗長是所有
運動裡最短的，它落在隨機水準不代表其他運動也是。本腳本把六個運動放在同一個協定下
並排比較。

協定（三個選擇都會改變結果，所以寫死在這裡而不是當參數）：

1. **不套任何順序假設**：直接取 `pelvis_angular_speed` / `torso_angular_speed` /
   `wrist_speed` 的原始峰值時間排序。弱標註為了讓標註可用，把骨盆與軀幹的搜尋範圍
   限制在上肢峰值之前，用它來「驗證」順序是循環論證。
2. **加速期窗**：起點取 `stride_foot_contact`，該運動沒有宣告時退回 `loading_peak`；
   終點取 `release_impact`。這兩端都不是由順序約束推出來的（著地看踝高度、
   觸球看腕速減速），所以不構成循環。窗外的隨勢動作會蓋掉真正的峰值。
3. **幾何乾淨**：髖線在**該窗內**的最短投影長度 ≥ ``USABLE_PROJECTION``。
   連線縮到趨近零時 ``atan2`` 是病態的，量到的「角速度」是假影。
   判定範圍與量測範圍必須是同一段，否則會把窗外塌陷的片段誤判為不可用。

隨機基準是 1/3! = 16.7%。

    python scripts/rotation_chain_by_sport.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kinetic_chain.analysis import (  # noqa: E402
    CHAIN_LINKS,
    USABLE_PROJECTION,
    projection_quality,
    unconstrained_sequence,
)
from kinetic_chain.data import Clip  # noqa: E402

logger = logging.getLogger("rotation_by_sport")

#: Penn Action 上的五個旋轉型運動；高爾夫另外從 GolfDB 讀。
PENN_SPORTS = (
    "baseball_swing",
    "baseball_pitch",
    "tennis_serve",
    "tennis_forehand",
    "bowling",
)
RANDOM_BASELINE = 100.0 / 6.0
INK = "#15181B"
MUTED = "#5C666D"
LINE = "#DCE0E3"
ACCENT = "#2F6F9F"
BAD = "#A63A46"


def acceleration_window(clip: Clip) -> tuple[int, int]:
    """加速期 `(起, 迄)`，迄為 exclusive。

    退回整段的情形只發生在兩端事件缺失或倒置時；正常資料不會走到。
    """
    events = clip.events
    lo = events.get("stride_foot_contact", events.get("loading_peak", 0))
    hi = events.get("release_impact", clip.num_frames - 1) + 1
    if hi <= lo + 1:
        return 0, clip.num_frames
    return lo, hi


def measure(clips: list[Clip], sport: str) -> dict:
    """回傳該運動的成立率與窗長；不可用的片段直接排除，不以估計值補。"""
    expected = tuple(name for name, _ in CHAIN_LINKS)
    ok: list[bool] = []
    windows: list[int] = []
    shortest: list[float] = []
    collapse_at: list[float] = []
    for clip in clips:
        signals = clip.signals()
        window = acceleration_window(clip)
        quality = projection_quality(signals, "pelvis", window=window)
        # 投影統計對**全部**片段計算，否則被過濾掉的正是幾何最差的那些，
        # 中位數會被自己的篩選條件抬上去，看不出各運動的機位差異。
        shortest.append(quality.min_relative)
        # 0 = 窗的起點，1 = 窗的終點。塌在動作中段比塌在兩端嚴重得多。
        collapse_at.append(quality.collapse_position)
        if quality.min_relative < USABLE_PROJECTION:
            continue
        peaks = unconstrained_sequence(signals, window=window)
        observed = tuple(sorted(peaks, key=lambda e: peaks[e]))
        ok.append(observed == expected)
        windows.append(window[1] - window[0] - 1)
    if not ok:
        raise SystemExit(f"{sport}：沒有任何幾何乾淨的片段")
    return {
        "sport": sport,
        "n_total": len(clips),
        "n": len(ok),
        "clean_fraction": round(len(ok) / len(clips), 3),
        "rate": round(100.0 * float(np.mean(ok)), 1),
        "window_frames": float(np.median(windows)),
        "min_projection_median": round(float(np.median(shortest)), 2),
        "collapse_position_median": round(float(np.median(collapse_at)), 2),
    }


def figure(rows: list[dict], path: Path) -> None:
    """窗長 vs 成立率。每個點是一個運動，點的大小是樣本數。"""
    windows = np.array([r["window_frames"] for r in rows], dtype=float)
    rates = np.array([r["rate"] for r in rows], dtype=float)
    sizes = np.array([r["n"] for r in rows], dtype=float)
    r = float(np.corrcoef(windows, rates)[0, 1])

    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=170)
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(True, color=LINE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    ax.axhline(
        RANDOM_BASELINE, color=BAD, linewidth=1.2, linestyle="--", zorder=1,
        label=f"chance = {RANDOM_BASELINE:.1f}% (1/3!)",
    )
    slope, intercept = np.polyfit(windows, rates, 1)
    span = np.linspace(windows.min() - 2, windows.max() + 3, 50)
    ax.plot(span, slope * span + intercept, color=LINE, linewidth=1.4, zorder=1)

    # 三個中段的點擠在一起，標籤上下交錯才不會疊字
    order = np.argsort(windows)
    levels = (16, -26, 32, -42)
    offsets = {int(index): levels[rank % len(levels)] for rank, index in enumerate(order)}

    for index, (row, x, y, s) in enumerate(zip(rows, windows, rates, sizes)):
        colour = BAD if y < RANDOM_BASELINE * 1.3 else ACCENT
        ax.scatter([x], [y], s=28 + 0.55 * s, color=colour, alpha=0.85,
                   edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(
            f"{row['sport']}  n={row['n']}",
            (x, y), textcoords="offset points", xytext=(0, offsets[index]),
            ha="center", fontsize=8.8, color=MUTED,
        )

    ax.set_xlabel("acceleration window (frames @ 30 fps, median)", color=MUTED, fontsize=9.5)
    ax.set_ylabel("proximal-to-distal rate (%)", color=MUTED, fontsize=9.5)
    ax.set_ylim(0, 68)
    ax.set_xlim(windows.min() - 3, windows.max() + 5)
    ax.set_title(
        f"Faster actions are harder to resolve at 30 fps    r = {r:+.2f} (n=6, not significant)",
        color=INK, fontsize=10.5, loc="left", pad=10,
    )
    legend = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(MUTED)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pennaction-root", type=Path, default=Path("data/raw/Penn_Action"))
    parser.add_argument("--golfdb", type=Path, default=Path("data/raw/golfDB.pkl"))
    parser.add_argument("--golfdb-cache", type=Path, default=Path("data/cache/golfdb_pose"))
    parser.add_argument("--output", type=Path, default=Path("runs/rotation_chain_by_sport.json"))
    parser.add_argument("--figure", type=Path, default=Path("docs/figures/rotation_by_window.png"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from kinetic_chain.datasets import golfdb, pennaction

    rows = [
        measure(pennaction.load(args.pennaction_root, sports=[sport]), sport)
        for sport in PENN_SPORTS
    ]
    rows.append(measure(golfdb.load(args.golfdb, args.golfdb_cache), "golf_swing"))
    rows.sort(key=lambda r: -r["rate"])

    def correlate(records: list[dict]) -> dict:
        """相關係數連同 p 值一起回報。

        只有六個點，光看 r 會過度解讀——r=+0.72 在 n=6 時 p 仍大於 0.05。
        """
        x = [r["window_frames"] for r in records]
        y = [r["rate"] for r in records]
        pearson = stats.pearsonr(x, y)
        spearman = stats.spearmanr(x, y)
        return {
            "n": len(records),
            "pearson_r": round(float(pearson.statistic), 3),
            "pearson_p": round(float(pearson.pvalue), 3),
            "spearman_rho": round(float(spearman.statistic), 3),
            "spearman_p": round(float(spearman.pvalue), 3),
        }

    windows = np.array([r["window_frames"] for r in rows], dtype=float)
    rates = np.array([r["rate"] for r in rows], dtype=float)
    correlation = float(np.corrcoef(windows, rates)[0, 1])
    without_golf = [r for r in rows if r["sport"] != "golf_swing"]

    result = {
        "protocol": {
            "chain": [name for name, _ in CHAIN_LINKS],
            "window": "stride_foot_contact（無則 loading_peak）→ release_impact",
            "projection_filter": f"髖線窗內最短相對投影長度 >= {USABLE_PROJECTION}",
            "random_baseline": round(RANDOM_BASELINE, 1),
        },
        "window_rate_correlation": correlate(rows),
        "window_rate_correlation_without_golf": correlate(without_golf),
        "sports": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure(rows, args.figure)

    for row in rows:
        logger.info(
            "%-16s n=%3d/%4d (%.0f%%)  rate=%4.1f%%  window=%.0f  塌陷位置=%.2f",
            row["sport"], row["n"], row["n_total"], 100 * row["clean_fraction"],
            row["rate"], row["window_frames"], row["collapse_position_median"],
        )
    stat = result["window_rate_correlation"]
    logger.info(
        "窗長 vs 成立率 r=%+.3f (p=%.3f, n=%d)；排除高爾夫 r=%+.3f (p=%.3f)；圖 → %s",
        stat["pearson_r"], stat["pearson_p"], stat["n"],
        result["window_rate_correlation_without_golf"]["pearson_r"],
        result["window_rate_correlation_without_golf"]["pearson_p"],
        args.figure,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
