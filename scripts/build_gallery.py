"""把所有實驗產出的圖片整理成一個可比對的資料夾。

圖散在 `docs/figures/`（可進版控的曲線圖）與 `runs/` 底下十幾個目錄（含人物影像，
不進版控），要一次看完得到處翻。本腳本依「主題 → 運動 → 種類」重新命名並複製到
單一資料夾，附一份 `INDEX.md` 說明每張圖是什麼、由哪個腳本產生。

輸出目錄含可辨識個人的影像，**不進版控**（已加入 .gitignore）。

    python scripts/build_gallery.py
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger("gallery")

#: (輸出檔名, 來源路徑, 分類, 說明, 產生腳本)
#: 輸出檔名前綴為兩位數編號，讓檔案總管的字母排序等同閱讀順序。
ITEMS: tuple[tuple[str, str, str, str, str], ...] = (
    # --- 01 主結果：高爾夫，唯一有真人標註的運動 -------------------------
    ("01-golf-accuracy.png", "docs/figures/golf_accuracy.png",
     "高爾夫（真人標註）", "各事件的誤差分布。四折 PCE 0.786，勝過 SwingNet 已發表的 0.715/0.761。",
     "scripts/plot_accuracy.py"),
    ("01-golf-frames-best.png", "runs/gallery_src/frames_golf/best_83.png",
     "高爾夫（真人標註）", "最好的一段，8/8。上排模型偵測、下排真人標註。", "scripts/visualize.py --crop"),
    ("01-golf-frames-median.png", "runs/gallery_src/frames_golf/median_10.png",
     "高爾夫（真人標註）", "中位的一段，6/8。容忍度只有 1 影格。", "scripts/visualize.py --crop"),
    ("01-golf-frames-worst.png", "runs/gallery_src/frames_golf/worst_328.png",
     "高爾夫（真人標註）", "最差的一段，0/8。容忍度 1 影格，全部差 2 格以上。", "scripts/visualize.py --crop"),

    # --- 02 棒球投球 -----------------------------------------------------
    ("02-pitch-frames-best.png", "runs/gallery_src/frames_pitch/best_0018.png",
     "棒球投球（弱標註）", "最好的一段，10/10。下排是規則推導的弱標註，不是真人標註。",
     "scripts/visualize.py --crop"),
    ("02-pitch-frames-median.png", "runs/gallery_src/frames_pitch/median_0035.png",
     "棒球投球（弱標註）", "中位的一段，7/10，容忍度 1 影格。", "scripts/visualize.py --crop"),
    ("02-pitch-frames-worst.png", "runs/gallery_src/frames_pitch/worst_0147.png",
     "棒球投球（弱標註）", "最差的一段，3/10。", "scripts/visualize.py --crop"),
    ("02-pitch-poses-best.png", "runs/gallery_src/poses_pitch/poses_penn_action_0018.png",
     "棒球投球（弱標註）", "模型實際看到的骨架，10/10。空心點與虛線是內插補值。", "scripts/visualize_poses.py"),
    ("02-pitch-poses-worst.png", "runs/gallery_src/poses_pitch/poses_penn_action_0147.png",
     "棒球投球（弱標註）", "同上，3/10。", "scripts/visualize_poses.py"),
    ("02-pitch-timeline.png", "docs/figures/chain_timeline.png",
     "棒球投球（弱標註）", "事件時間分布，正規化到投球期。", "scripts/visualize_chain.py"),
    ("02-pitch-chain-trace.png", "docs/figures/chain_trace_penn_action_0045.png",
     "棒球投球（弱標註）", "骨盆／軀幹／上肢三條速度曲線與峰值時序。", "scripts/visualize_chain.py"),
    ("02-pitch-separation.png", "docs/figures/chain_separation.png",
     "棒球投球（弱標註）", "分離時間分布，附取樣解析度的界線。", "scripts/visualize_chain.py"),

    # --- 03 投影假影與機位：本專案最重要的方法論發現 ----------------------
    ("03-projection-artifact.png", "docs/figures/projection_artifact.png",
     "投影假影（方法論）", "髖線投影長度與骨盆角速度的關係。連線縮短時 atan2 病態，量到的是假影。",
     "scripts/visualize_chain.py"),
    ("03-viewpoint-profile.png", "docs/figures/viewpoint_profile.png",
     "投影假影（方法論）", "機位對旋轉量測的影響。投手側面的可用率最高，先前判反了。",
     "scripts/viewpoint_analysis.py"),

    # --- 04 棒球打者：推翻先前結論的對照組 --------------------------------
    ("04-bat-frames-best.png", "runs/visualise_bat_crop/best_0220.png",
     "棒球打者（弱標註）", "最好的一段，9/10。但前五欄打者還沒開始揮，球棒還在後面——規則標錯位置。",
     "scripts/visualize.py --crop"),
    ("04-bat-frames-worst.png", "runs/visualise_bat_crop/worst_0266.png",
     "棒球打者（弱標註）", "最差的一段，1/10。骨盆峰值差 12 格。", "scripts/visualize.py --crop"),
    ("04-bat-poses-best.png", "docs/figures/bat_poses_best.png",
     "棒球打者（弱標註）", "同一段的骨架。姿勢差異很小，看不出球棒還在後面。", "scripts/visualize_poses.py"),
    ("04-bat-poses-worst.png", "docs/figures/bat_poses_worst.png",
     "棒球打者（弱標註）", "同上，最差的一段。後半段整個左側都是補值。", "scripts/visualize_poses.py"),
    ("04-bat-timeline.png", "docs/figures/bat_timeline.png",
     "棒球打者（弱標註）", "22 段驗證片段的事件時間分布。", "scripts/visualize_chain.py"),
    ("04-bat-trace-ok.png", "docs/figures/bat_trace_in_order.png",
     "棒球打者（弱標註）", "順序成立的一段：骨盆 f22 → 軀幹 f26 → 上肢 f32。", "scripts/visualize_chain.py"),
    ("04-bat-trace-bad.png", "docs/figures/bat_trace_out_of_order.png",
     "棒球打者（弱標註）", "順序不成立：上肢與骨盆只差 1 格，33 ms 在 30 fps 下量不出來。",
     "scripts/visualize_chain.py"),
    ("04-rotation-by-window.png", "docs/figures/rotation_by_window.png",
     "棒球打者（弱標註）", "六個運動的窗長 vs 序列成立率。打者提供了缺少的對照，推翻先前的結論。",
     "scripts/rotation_chain_by_sport.py"),

    # --- 05 舉重：伸展型動力鏈 -------------------------------------------
    ("05-lift-phases.png", "runs/figures/lift_phases.png",
     "舉重（弱標註）", "舉重的分期定義示意。", "手工"),
    ("05-lift-frames-best.png", "runs/gallery_src/frames_lift/best_0729.png",
     "舉重（弱標註）", "最好的一段，9/9。注意容忍度高達 9 影格，這個分數被容忍度灌水。",
     "scripts/visualize.py --crop"),
    ("05-lift-frames-worst.png", "runs/gallery_src/frames_lift/worst_0748.png",
     "舉重（弱標註）", "最差的一段，3/9，容忍度 20 影格。", "scripts/visualize.py --crop"),
    ("05-lift-frames-median.png", "runs/gallery_src/frames_lift/median_0762.png",
     "舉重（弱標註）", "中位的一段，7/9，容忍度 13 影格。", "scripts/visualize.py --crop"),
    ("05-lift-poses-best.png", "runs/gallery_src/poses_lift/poses_penn_action_0729.png",
     "舉重（弱標註）", "模型實際看到的骨架。", "scripts/visualize_poses.py"),
    ("05-lift-poses-worst.png", "runs/gallery_src/poses_lift/poses_penn_action_0748.png",
     "舉重（弱標註）", "同上，最差的一段 3/9。", "scripts/visualize_poses.py"),
    ("05-lift-trace.png", "docs/figures/lift_trace.png",
     "舉重（弱標註）", "伸展型動力鏈：髖 → 膝 → 上肢的三點夾角速度。", "scripts/visualize_chain.py"),
    ("05-chain-comparison.png", "docs/figures/chain_comparison.png",
     "舉重（弱標註）", "旋轉型與伸展型動力鏈的對照。", "scripts/visualize_chain.py"),

    # --- 06 自備影片 -----------------------------------------------------
    ("06-own-frames.png", "runs/visualise_own.png",
     "自備影片（弱標註）", "自己拍的舉重影片，RTMPose 抽姿態後的偵測結果。", "scripts/visualize.py"),
    ("06-own-trace.png", "docs/figures/own_lift_trace.png",
     "自備影片（弱標註）", "自備影片的伸展型動力鏈曲線。", "scripts/visualize_chain.py"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("gallery"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)

    copied: list[tuple[str, str, str, str]] = []
    missing: list[str] = []
    for name, source, group, caption, script in ITEMS:
        path = Path(source)
        if not path.is_file():
            missing.append(f"{name}  <-  {source}")
            continue
        shutil.copy2(path, args.output / name)
        copied.append((name, group, caption, script))

    lines = [
        "# 實驗圖片總表",
        "",
        f"共 {len(copied)} 張。由 `scripts/build_gallery.py` 產生，**不進版控**"
        "（含可辨識個人的影像）。",
        "",
        "## 讀圖前必須知道的兩件事",
        "",
        "1. **只有高爾夫是真人標註**（GolfDB）。其餘全部是規則推導的弱標註，",
        "   影格對照圖的下排寫 `rule` 而不是 `true` 就是這個意思。",
        "   弱標註上的分數只代表模型學會了規則，不代表規則正確。",
        "2. **命中／未中的標準跨運動不可比。** 容忍度與動作長度成正比：",
        "   高爾夫平均 2.7 影格、打擊 1.0、舉重 9.7。舉重看起來分數高是容忍度寬，",
        "   固定成 2 影格後舉重反而是最差的。見 `docs/architecture.md` 的 S7。",
        "",
    ]
    current = None
    for name, group, caption, script in copied:
        if group != current:
            lines += ["", f"## {group}", ""]
            current = group
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"![{name}]({name})")
        lines.append("")
        lines.append(caption)
        lines.append("")
        lines.append(f"產生：`{script}`")
        lines.append("")

    if missing:
        lines += ["", "## 缺少的來源檔", ""]
        lines += [f"- `{item}`" for item in missing]
        lines += ["", "重跑對應的腳本即可補上。", ""]

    (args.output / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")

    logger.info("複製 %d 張到 %s", len(copied), args.output)
    for item in missing:
        logger.warning("缺少來源：%s", item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
