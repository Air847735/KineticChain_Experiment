"""論文風格的架構圖。

`README.md` 的 mermaid 流程圖說明得了「有哪些階段」，說明不了「膨脹卷積實際怎麼讀
這 58 維特徵」——那需要畫出通道軸與時間軸的區別、以及感受野怎麼隨層數展開。
這支腳本產生三張圖：

``arch_pipeline.png``   全流程，含每一段的張量形狀與通道／時間軸的區別
``arch_dilation.png``  膨脹卷積怎麼讀時間軸；感受野逐層展開 vs 各運動的動作長度
``arch_block.png``     單一殘差塊內部，含 FiLM 條件

三張都只有圖表，不含人物影像，可以進版控。

    python scripts/plot_architecture.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from kinetic_chain.model import ModelConfig  # noqa: E402

logger = logging.getLogger("plot_architecture")

INK = "#16181A"
MUTED = "#5E656B"
FAINT = "#9AA1A6"
LINE = "#D5D9DC"
PAPER = "#FFFFFF"
BAND = "#F2F4F5"

# 四段管線各自的顏色。確定性的三段用同一族藍灰，有參數的那段用暖色標出來。
DETERMINISTIC = "#4A6E8F"
LEARNED = "#B0762A"
DECODE = "#3F7A5A"
ACCENT = "#A63A46"


def _fonts() -> None:
    from matplotlib import font_manager

    wanted = ["Noto Serif CJK TC", "Noto Sans Mono CJK TC", "DejaVu Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    if not available & set(wanted[:2]):
        raise SystemExit("找不到中文字型，圖上的標題會變成空框；請安裝 Noto CJK")
    plt.rcParams["font.family"] = wanted
    plt.rcParams["axes.unicode_minus"] = False


def _blank(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_facecolor(PAPER)


def _box(ax, x, y, w, h, label, sub="", colour=DETERMINISTIC, alpha=0.13, fs=9.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=colour, edgecolor=colour, alpha=alpha, linewidth=1.3, zorder=2,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="none", edgecolor=colour, linewidth=1.3, zorder=3,
        )
    )
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), label, ha="center",
            va="center", fontsize=fs, color=INK, zorder=4)
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                fontsize=fs - 2.2, color=MUTED, zorder=4)


def _arrow(ax, x0, y0, x1, y1, colour=FAINT, style="-|>", lw=1.2, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11,
            color=colour, linewidth=lw, linestyle=ls, zorder=1,
            shrinkA=1, shrinkB=1,
        )
    )


# ----------------------------------------------------------------- 圖 1
def pipeline_figure(path: Path) -> None:
    """全流程。重點在於畫出「通道軸」與「時間軸」是兩件不同的事。"""
    fig = plt.figure(figsize=(11.8, 6.7), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    _blank(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.035, 0.955, "KineticChain 管線", fontsize=14.5, color=INK, va="top")
    ax.text(0.035, 0.905,
            "四段，只有第三段有學習參數。T = 影格數，隨影片長度變動；"
            "整段影片一次進模型，不逐格餵。",
            fontsize=9.2, color=MUTED, va="top")

    y, h = 0.635, 0.155
    xs = [0.035, 0.265, 0.495, 0.725]
    w = 0.20

    _box(ax, xs[0], y, w, h, "1 · 姿態抽取", "RTMPose ONNX", DETERMINISTIC)
    _box(ax, xs[1], y, w, h, "2 · 力學特徵", "正規化 · 平滑 · 差分", DETERMINISTIC)
    _box(ax, xs[2], y, w, h, "3 · 膨脹卷積網路", "554,739 參數", LEARNED)
    _box(ax, xs[3], y, w, h, "4 · 順序約束解碼", "動態規劃 · 無參數", DECODE)

    shapes = [
        (xs[0] + w / 2, "影片\nT × H × W × 3"),
        (xs[1] + w / 2, "關鍵點\nT × 13 × 3"),
        (xs[2] + w / 2, "特徵\nT × 58"),
        (xs[3] + w / 2, "logits\n19 × T"),
    ]
    for x, text in shapes:
        ax.text(x, y + h + 0.022, text.replace("\n", "  "), ha="center", va="bottom",
                fontsize=8.4, color=MUTED)
    ax.text(xs[3] + w + 0.008, y + h / 2, "事件影格\n+ 信心", ha="left", va="center",
            fontsize=8.4, color=INK, linespacing=1.5)

    for i in range(3):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    # 底下：確定性 vs 有參數
    ax.add_patch(Rectangle((xs[0], y - 0.048), xs[1] + w - xs[0], 0.024,
                           facecolor=DETERMINISTIC, alpha=0.16, edgecolor="none"))
    ax.text((xs[0] + xs[1] + w) / 2, y - 0.036, "確定性 · 可單獨單元測試",
            ha="center", va="center", fontsize=8, color=DETERMINISTIC)
    ax.add_patch(Rectangle((xs[2], y - 0.048), w, 0.024,
                           facecolor=LEARNED, alpha=0.16, edgecolor="none"))
    ax.text(xs[2] + w / 2, y - 0.036, "唯一學習的部分",
            ha="center", va="center", fontsize=8, color=LEARNED)
    ax.add_patch(Rectangle((xs[3], y - 0.048), w, 0.024,
                           facecolor=DECODE, alpha=0.16, edgecolor="none"))
    ax.text(xs[3] + w / 2, y - 0.036, "順序為硬保證",
            ha="center", va="center", fontsize=8, color=DECODE)

    # ---- 下半：58 維特徵在模型裡是「通道」，不是「一串數字」
    ax.plot([0.035, 0.965], [0.455, 0.455], color=LINE, linewidth=1)
    ax.text(0.035, 0.425, "模型看到的東西：58 是「通道」，T 是「時間」",
            fontsize=11.5, color=INK, va="top")
    ax.text(0.035, 0.383,
            "特徵矩陣轉置成 (58, T) 後餵給 Conv1d。卷積只沿時間軸滑動，"
            "在每一個時間位置上一次讀進全部 58 個通道。",
            fontsize=9, color=MUTED, va="top")

    # 特徵格點
    gx0, gy0 = 0.045, 0.105
    gw, gh = 0.40, 0.245
    n_t, n_c = 26, 12
    rng = np.random.default_rng(3)
    grid = rng.normal(size=(n_c, n_t))
    grid = (grid - grid.min()) / (grid.max() - grid.min())
    ax.imshow(grid, extent=(gx0, gx0 + gw, gy0, gy0 + gh), aspect="auto",
              cmap="Blues", alpha=0.55, zorder=2, origin="lower", vmin=-0.4, vmax=1.3)
    ax.add_patch(Rectangle((gx0, gy0), gw, gh, facecolor="none",
                           edgecolor=MUTED, linewidth=1.1, zorder=3))

    ax.text(gx0 - 0.012, gy0 + gh / 2, "58 個通道", rotation=90, ha="center",
            va="center", fontsize=9, color=INK)
    ax.text(gx0 + gw * 0.16, gy0 - 0.026, "時間 T（影格）", ha="center", va="center",
            fontsize=9, color=INK)
    for label, frac in (("關節座標 ×26", 0.80), ("關節速度 ×13", 0.52),
                        ("角度與角速度 ×9", 0.30), ("其他 ×10", 0.10)):
        ax.text(gx0 + gw + 0.012, gy0 + gh * frac, label, ha="left", va="center",
                fontsize=7.8, color=MUTED)

    # 卷積核：涵蓋全部通道、時間上只取 5 格
    kx = gx0 + gw * 0.44
    kw = gw / n_t
    for offset in range(5):
        ax.add_patch(Rectangle((kx + offset * kw * 2, gy0), kw, gh,
                               facecolor=ACCENT, alpha=0.22,
                               edgecolor=ACCENT, linewidth=1.2, zorder=4))
    ax.annotate(
        "一個卷積核：高度 = 全部 58 通道，寬度 = 5 格、間隔 d",
        xy=(kx + 4 * kw, gy0), xytext=(gx0 + gw * 0.52, gy0 - 0.062),
        fontsize=8.2, color=ACCENT, ha="center", va="center",
        arrowprops=dict(arrowstyle="-", color=ACCENT, linewidth=0.9),
    )

    # 右側：兩種卷積的差別
    bx = 0.60
    ax.text(bx, gy0 + gh + 0.055, "兩種卷積，各自只動一個軸",
            fontsize=9.6, color=INK, va="bottom")

    _box(ax, bx, gy0 + gh - 0.062, 0.36, 0.066,
         "第 0 層：1×1 卷積  58 → 128",
         "只混通道，完全不看時間。逐格把 58 個量重組成 128 個",
         DETERMINISTIC, alpha=0.10, fs=8.8)
    _box(ax, bx, gy0 + gh - 0.152, 0.36, 0.066,
         "第 1–6 層：膨脹卷積  128 → 128",
         "只沿時間滑動，每格一次讀完 128 通道 × 5 個時間點",
         LEARNED, alpha=0.10, fs=8.8)
    _box(ax, bx, gy0 + gh - 0.242, 0.36, 0.066,
         "輸出層：1×1 卷積  128 → 19",
         "逐格算出 19 個事件槽各自的分數",
         DECODE, alpha=0.10, fs=8.8)

    ax.text(bx, gy0 - 0.062,
            "第 0 層之後，通道就不再是「座標」「速度」了——\n"
            "它們是模型自己學出來的組合。",
            fontsize=8.2, color=MUTED, va="center", linespacing=1.6)

    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


# ----------------------------------------------------------------- 圖 2
def dilation_figure(path: Path) -> None:
    """膨脹卷積怎麼讀時間軸。上圖畫機制，下圖畫感受野 vs 各運動的動作長度。"""
    cfg = ModelConfig().resolved()
    kernel = cfg.kernel_size
    dilations = [2**i for i in range(cfg.num_layers)]

    half = np.cumsum([(kernel - 1) * d for d in dilations])
    spans = 2 * half + 1

    fig = plt.figure(figsize=(11.5, 7.8), dpi=200)
    ax = fig.add_axes([0.055, 0.495, 0.90, 0.395])
    ax2 = fig.add_axes([0.075, 0.075, 0.79, 0.315])

    fig.text(0.055, 0.965, "膨脹卷積怎麼讀時間軸", fontsize=14.5, color=INK, va="top")
    fig.text(0.055, 0.928,
             "窗戶永遠是 5 格，但每往上一層，格子之間的間隔加倍。"
             "參數量不變，看得到的範圍指數成長。",
             fontsize=9.2, color=MUTED, va="top")

    # ---- 上：每一層實際讀哪五格。畫完整連線會糊成一團，改畫「抽樣位置」。
    _blank(ax)
    shown = 4
    half_view = 34
    ax.set_xlim(-half_view - 11, half_view + 15)
    ax.set_ylim(-0.95, shown - 0.30)

    ticks = list(range(-half_view, half_view + 1))
    for layer in range(shown):
        d = dilations[layer]
        y_row = shown - 1 - layer
        ax.plot([-half_view, half_view], [y_row, y_row], color=BAND, linewidth=9,
                solid_capstyle="round", zorder=0)
        for i in ticks:
            ax.plot([i], [y_row], "o", markersize=2.1, color=LINE, zorder=2)

        taps = [(i - (kernel - 1) // 2) * d for i in range(kernel)]
        for tap in taps:
            ax.plot([tap, 0], [y_row, y_row + 0.30], color=ACCENT,
                    linewidth=0.85, alpha=0.55, zorder=3)
            ax.plot([tap], [y_row], "o", markersize=5.6, color=ACCENT, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.9)
        ax.plot([0], [y_row + 0.30], "s", markersize=5.2, color=INK, zorder=5,
                markeredgecolor="white", markeredgewidth=0.9)

        ax.text(-half_view - 1.8, y_row, f"第 {layer + 1} 層   d = {d}",
                ha="right", va="center", fontsize=8.8, color=INK)
        ax.text(half_view + 1.8, y_row,
                f"這一層跨 ±{2 * d}　累積 ±{half[layer]}",
                ha="left", va="center", fontsize=8.2, color=MUTED)

    ax.text(0, shown - 1 + 0.52, "輸出的那一格", ha="center", va="bottom",
            fontsize=8.6, color=INK)
    ax.annotate(
        "每層都只取 5 格，但間隔加倍",
        xy=(-2 * dilations[shown - 1], 0), xytext=(-half_view + 1, -0.70),
        fontsize=8.6, color=ACCENT, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=ACCENT, linewidth=0.9,
                        connectionstyle="angle,angleA=0,angleB=90,rad=3"),
    )
    ax.text(half_view + 1.8, -0.72,
            "第 5、6 層同理（d = 16、32），\n累積到 ±252",
            ha="left", va="center", fontsize=8.2, color=FAINT, linespacing=1.5)

    # ---- 下：感受野 vs 動作長度
    ax2.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax2.spines[side].set_color(LINE)
    ax2.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax2.grid(True, axis="y", color=LINE, linewidth=0.6, alpha=0.7)
    ax2.set_axisbelow(True)

    layers = np.arange(1, len(dilations) + 1)

    ax2.step(np.concatenate([[0], layers]), np.concatenate([[1], spans]),
             where="post", color=LEARNED, linewidth=2.0, zorder=3)
    ax2.plot(layers, spans, "o", color=LEARNED, markersize=5.5, zorder=4,
             markeredgecolor="white", markeredgewidth=1.1)
    for layer, value, d in zip(layers, spans, dilations):
        ax2.annotate(f"{value}", (layer, value), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8.2, color=LEARNED)
        ax2.annotate(f"d={d}", (layer, value), textcoords="offset points",
                     xytext=(-17, -4), ha="right", fontsize=7.6, color=FAINT)

    references = [
        ("高爾夫下桿 9 格", 9, ACCENT),
        ("打擊動作 36 格", 36, MUTED),
        ("投球動作 74 格", 74, MUTED),
        ("舉重動作 287 格", 287, MUTED),
    ]
    for label, value, colour in references:
        ax2.axhline(value, color=colour, linewidth=1.0, linestyle="--", alpha=0.65)
        ax2.text(6.12, value, label, ha="left", va="center", fontsize=8.2,
                 color=colour)

    ax2.set_yscale("log")
    ax2.set_xlim(0, 6.08)
    ax2.set_ylim(3, 1400)
    ax2.set_xticks(layers)
    ax2.set_xlabel("膨脹卷積層數", color=MUTED, fontsize=9)
    ax2.set_ylabel("雙向感受野（影格，對數軸）", color=MUTED, fontsize=9)
    ax2.set_title(
        "六層之後每一格都看得到前後 252 格（8.4 秒 @30fps）——"
        "四個運動的完整動作都在範圍內",
        color=INK, fontsize=9.6, loc="left", pad=8,
    )

    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


# ----------------------------------------------------------------- 圖 3
def block_figure(path: Path) -> None:
    """單一殘差塊內部，含 FiLM 條件與參數量。"""
    fig = plt.figure(figsize=(10.5, 4.6), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    _blank(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.035, 0.945, "單一殘差塊（共六層，唯一的差別是 d）",
            fontsize=13.5, color=INK, va="top")
    ax.text(0.035, 0.885,
            "每層 81,920 個卷積參數（128 輸出 × 128 輸入 × 5 格）。"
            "六層合計約 49 萬，佔全模型的 89%。",
            fontsize=8.8, color=MUTED, va="top")

    y, h = 0.42, 0.20
    xs = [0.045, 0.215, 0.385, 0.525, 0.655, 0.785]
    ws = [0.145, 0.145, 0.115, 0.105, 0.105, 0.14]
    boxes = [
        ("Conv1d", "128 → 128\nk=5 · dilation d", LEARNED),
        ("BatchNorm1d", "逐通道標準化", DETERMINISTIC),
        ("FiLM", "γ ⊙ h + β", ACCENT),
        ("GELU", "非線性", DETERMINISTIC),
        ("Dropout", "p = 0.1", DETERMINISTIC),
        ("＋ 殘差相加", "與輸入相加", DECODE),
    ]
    for x, w, (label, sub, colour) in zip(xs, ws, boxes):
        _box(ax, x, y, w, h, label, sub, colour, alpha=0.12, fs=9)
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i] + ws[i], y + h / 2, xs[i + 1], y + h / 2)

    ax.text(0.02, y + h / 2, "h", fontsize=11, color=INK, ha="center", va="center")
    _arrow(ax, 0.032, y + h / 2, xs[0], y + h / 2)
    ax.text(0.955, y + h / 2, "h′", fontsize=11, color=INK, ha="center", va="center")
    _arrow(ax, xs[-1] + ws[-1], y + h / 2, 0.945, y + h / 2)

    # 殘差捷徑
    skip_y = y + h + 0.10
    _arrow(ax, 0.038, y + h * 0.78, 0.038, skip_y, colour=DECODE, style="-")
    _arrow(ax, 0.038, skip_y, xs[-1] + ws[-1] / 2, skip_y, colour=DECODE, style="-")
    _arrow(ax, xs[-1] + ws[-1] / 2, skip_y, xs[-1] + ws[-1] / 2, y + h,
           colour=DECODE)
    ax.text((0.038 + xs[-1]) / 2, skip_y + 0.022, "殘差捷徑：讓六層疊得起來",
            ha="center", va="bottom", fontsize=8.4, color=DECODE)

    # FiLM 條件來源
    emb_y = y - 0.20
    _box(ax, xs[2] - 0.075, emb_y, 0.26, 0.115,
         "運動項目 embedding  32 維",
         "線性投影成 γ 與 β（各 128 維）", ACCENT, alpha=0.09, fs=8.6)
    _arrow(ax, xs[2] + 0.055, emb_y + 0.115, xs[2] + 0.055, y,
           colour=ACCENT, ls="--")

    ax.text(0.035, 0.11,
            "FiLM 讓「哪個運動」影響每一層的特徵轉換，而不是只拼接在輸入上。"
            "但單運動訓練時 embedding 只有一個有效索引，\n"
            "γ 與 β 退化成固定常數——實測聯合訓練較差，所以這部分目前等於閒置。",
            fontsize=8.4, color=MUTED, va="center", linespacing=1.7)

    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _fonts()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, fn in (
        ("arch_pipeline.png", pipeline_figure),
        ("arch_dilation.png", dilation_figure),
        ("arch_block.png", block_figure),
    ):
        path = args.output_dir / name
        fn(path)
        logger.info("%s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
