# 資料與實驗對照表

- Date: 2026-08-17
- 目的：每個運動項目的資料從哪來、標註是什麼型態、訓練出來的權重與實驗結果放在哪。

## 一句話

**真人事件標註只有高爾夫一個運動（GolfDB）**，其餘六個運動全部是規則推導的弱標註
（Penn Action）。這條界線決定了哪些數字可以當結論、哪些只能當自我一致性檢查。

## 運動項目 → 資料來源

| 運動項目 | 轉接模組 | 資料集 | 標註型態 | 讀入段數 |
|---|---|---|---|---|
| `golf_swing` | `datasets/golfdb.py` | GolfDB | **真人**，8 事件 | 1391 |
| `golf_swing` | `datasets/pennaction.py` | Penn Action `golf_swing` | 弱標註 | 123 |
| `baseball_pitch` | `datasets/pennaction.py` | Penn Action `baseball_pitch` | 弱標註 | 139 |
| `baseball_swing` | `datasets/pennaction.py` | Penn Action `baseball_swing` | 弱標註 | 110 |
| 自備影片 | `datasets/local_video.py` | `/srv/datasets/weight` | 弱標註（RTMPose 姿態） | 8 / 12 |
| 人工標註 | `datasets/annotations.py` | `annotations/*.csv` | **真人**（尚未實際標註） | 0 |
| `tennis_serve` | `datasets/pennaction.py` | Penn Action `tennis_serve` | 弱標註 | 136 |
| `tennis_forehand` | `datasets/pennaction.py` | Penn Action `tennis_forehand` | 弱標註 | 107 |
| `bowling` | `datasets/pennaction.py` | Penn Action `bowl` | 弱標註 | 102 |
| `clean_and_jerk` | `datasets/pennaction.py` | Penn Action `clean_and_jerk` | 弱標註 | 60 |

高爾夫是唯一同時有兩個來源的運動；`Clip.label_source` 把 `human` 與 `weak` 分開，
評估報表一律不合併。

Penn Action 合計讀入 777 段，捨棄 380 段（覆蓋率過低 63、弱標註不合格 317）。
弱標註不合格的三個原因見 `weak_labels.derive`：違反宣告時序、四個以上事件擠在同一格、
中段事件被釘在片段邊界。

## 原始資料位置（不納入版控）

```
data/raw/golfDB.pkl              GolfDB 標註，1400 筆
data/raw/videos_160/             GolfDB 影片，160×160，1400 支
data/raw/Penn_Action/labels/     Penn Action 關節標註，2326 個 .mat
data/raw/Penn_Action/frames/     Penn Action 影格（只在需要目視檢查時解壓）
data/cache/golfdb_pose/          RTMPose 抽出的關鍵點快取，1400 個 .npz，40 MB
```

Penn Action 的關節是**人工標註**，不需要跑姿態抽取；GolfDB 只有影片，關鍵點由
RTMPose 抽出後快取。所以兩者的姿態品質不是同一回事——Penn Action 的關節是真值，
GolfDB 的是估計值。

但 Penn Action 的「真值」有缺口：**15.5% 的關節標為不可見**（打擊項目的全體平均
可見率 84.5%）。這些關節在 `features.compute` 中沿時間軸內插補值，模型看到的是補過的
姿勢。`scripts/visualize_poses.py` 會把補值的關節畫成空心點，避免把補出來的位置
當成量到的位置。

## 事件詞彙的分配

| | 數量 | 說明 |
|---|---|---|
| Canonical | 10 | 跨運動共用輸出槽 |
| Sport-specific（高爾夫） | 3 | `golf_toe_up` 等，定義在球桿上 |
| Sport-specific（舉重） | 6 | `clean_liftoff` 等 |
| **輸出頭總槽數** | **19** | 所有運動共用，以遮罩選出 |

每個運動用到幾個 canonical：

| 運動 | 事件數 | canonical | 專屬 |
|---|---|---|---|
| `golf_swing` | 11 | 8 | 3 |
| `baseball_pitch` | 10 | 10 | 0 |
| `bowling` | 10 | 10 | 0 |
| `baseball_swing` | 9 | 9 | 0 |
| `tennis_forehand` | 9 | 9 | 0 |
| `tennis_serve` | 9 | 9 | 0 |
| **`clean_and_jerk`** | 9 | **3** | **6** |

舉重只有 3 個 canonical，是 canonical 詞彙適用邊界的直接證據——詳見
`docs/lift-analysis.md`。

## 訓練出來的權重

| 檔案 | 訓練內容 | 用途 |
|---|---|---|
| `runs/golf/model.pt` | `golf_swing`，GolfDB 第 1 折 | 高爾夫的主力模型 |
| `runs/pitch/model.pt` | `baseball_pitch`，Penn Action | `scripts/pitch_analysis.py` |
| `runs/lift/model.pt` | `clean_and_jerk`，Penn Action | `scripts/lift_analysis.py` |
| `runs/bat/model.pt` | `baseball_swing`，Penn Action | `docs/batting-analysis.md` |

`runs/` 不納入版控。全部由下列指令重建：

```bash
python -m kinetic_chain.cli train --sport golf_swing --no-pennaction --val-fold 1 \
    --epochs 60 --output runs/golf
python -m kinetic_chain.cli train --sport baseball_pitch --no-golfdb \
    --epochs 80 --output runs/pitch
python -m kinetic_chain.cli train --sport clean_and_jerk --no-golfdb \
    --epochs 80 --output runs/lift
python -m kinetic_chain.cli train --sport baseball_swing --no-golfdb \
    --epochs 80 --output runs/bat
```

## 實驗腳本 → 產出 → 文件

| 腳本 | 產出 | 回答什麼 | 寫在哪 |
|---|---|---|---|
| `run_experiments.py` | `runs/experiments.json` | 四折 PCE；聯合訓練 vs 單運動（S4）；微調（S6） | `docs/architecture.md` |
| `data_efficiency.py` | `runs/data_efficiency.json` | 新運動要標多少資料 | `README.md` |
| `pitch_analysis.py` | `runs/pitch_analysis.json` | 投球的分期時序與序列是否成立 | `docs/pitch-analysis.md` |
| `viewpoint_analysis.py` | `runs/viewpoint_analysis.json` | 機位對旋轉量測的影響 | `docs/pitch-analysis.md` |
| `lift_analysis.py` | `runs/lift_analysis.json` | 舉重的伸展型動力鏈 | `docs/lift-analysis.md` |
| `lift_analysis.py --source local` | `runs/lift_analysis_local.json` | 自備影片的實測 | `docs/own-video-analysis.md` |
| `rotation_chain_by_sport.py` | `runs/rotation_chain_by_sport.json`、`docs/figures/rotation_by_window.png` | 六個旋轉型運動的序列成立率，只看幾何乾淨的片段 | `docs/batting-analysis.md` |
| `bat_report.py` | `runs/bat_report.json` | 打者驗證集的逐段偵測結果 | `docs/batting-analysis.md` |
| `visualize_poses.py` | `runs/poses/*.png` | 每個事件那一格的骨架姿勢（**只有關節，不含影像，可進版控**） | `docs/batting-analysis.md` |
| `error_budget.py` | `runs/error_budget.json` | 跨運動的誤差（影格／毫秒／固定容忍度），PCE 不可跨運動比較 | `docs/architecture.md` S7 |
| `failure_modes.py` | `runs/failure_modes.json` | 大錯集中在整段還是特定事件 | `docs/architecture.md` S7 |
| `rule_type_errors.py` | `runs/rule_type_errors.json` | 誤差是否隨弱標註規則的種類而不同 | `docs/architecture.md` S7 |
| `label_vs_data.py` | `runs/label_vs_data.json` | 把高爾夫降到 88 段，分離「資料量」與「標註品質」 | `docs/architecture.md` S7 |
| `build_gallery.py` | `gallery/`（含 `INDEX.md`） | 把所有實驗圖片整理成一個可比對的資料夾（**含人物，不進版控**） | 僅本機 |
| `make_annotation_template.py` | `annotations/*.csv` | 預填的人工標註範本 | `docs/own-video-analysis.md` |
| `plot_accuracy.py` | `docs/figures/*.png` | 各事件的誤差分布 | `README.md` |
| `visualize.py` | `runs/visualise*/` | 逐格畫面對照（**含人物，不進版控**）；`--crop` 裁到運動員周圍 | 僅本機 |
| `visualize_chain.py` | `docs/figures/*.png` | 動力鏈曲線、投影假影 | `docs/pitch-analysis.md` |

## 哪些數字可以當結論

| 數字 | 依據 | 可信度 |
|---|---|---|
| 高爾夫四折 PCE 0.786 | GolfDB 真人標註，1391 段 | **可當結論**，且與 SwingNet 同協定可比 |
| 各運動 vs 弱標註的 PCE | 規則推導的標註 | 只代表模型學會規則的程度 |
| **跨運動比較 PCE** | 容忍度隨動作長度變動（1.0 到 9.7 影格） | **不可比**，必須改用固定容忍度或毫秒，見 `docs/architecture.md` S7 |
| 旋轉型序列成立率 | 受投影假影汙染 | **不可當結論**，見 `docs/pitch-analysis.md` |
| 伸展型序列成立率 42%（舉重） | 三點夾角，不受該假影影響 | 可當觀察，但無真值可驗證 |
| 旋轉型序列成立率（幾何乾淨的片段） | 已排除投影假影 | 可當觀察；跨六個運動一致，見 `docs/batting-analysis.md` |
| 窗長與成立率的相關 `r = +0.72` | 只有六個點，p = 0.108 | **不可當結論**，只是目前最合理的解釋 |
| 各階段時間佔比 | 由偵測事件換算 | 相對比例可用，絕對解析度受 30 fps 限制 |
| 打者 `arm_peak_velocity` PCE 0.818 | 該格手腕有 19% 是內插補值 | 可當觀察，但不是全部量到的 |
| 打者 `pelvis_peak_rotation` / `torso_peak_rotation` 的偵測值 | 弱標註只有 33% / 37% 落在真峰值上 | **不可用**，弱標註已知標錯，見 `docs/batting-analysis.md` |
