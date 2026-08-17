# Project Overview

- Project: `KineticChain`
- Purpose: 一套流程，從影片中偵測指定運動項目的動力鏈關鍵時間點；每個運動各訓練一份權重。
- Project type: library + CLI（核心為可匯入的 Python 套件，CLI 為薄封裝）
- Primary language / runtime: Python `>=3.10`（見 `pyproject.toml`），開發/測試用 conda 環境 `kinetic-chain`（Python 3.12）
- Requirements source: `docs/spec.md`
- Design source: `docs/architecture.md`

# Architecture Map

- `src/kinetic_chain/events.py`：全域事件詞彙與 `SportSpec` 註冊表；新增運動項目的唯一入口
- `src/kinetic_chain/skeleton.py`：關鍵點布局（COCO-17 / Penn Action-13）與跨布局對映
- `src/kinetic_chain/pose.py`：RTMPose（rtmlib/ONNX）影片 → `(T, J, 3)` 關鍵點序列
- `src/kinetic_chain/features.py`：關鍵點 → 尺度與位置不變的力學特徵 `(T, F)`
- `src/kinetic_chain/weak_labels.py`：由力學訊號程式化推導 canonical 事件（**弱標註**）
- `src/kinetic_chain/model.py`：`KineticChainNet`，dilated TCN + FiLM 運動項目條件 + 共用事件輸出頭
- `src/kinetic_chain/decode.py`：順序約束 Viterbi 解碼，無參數
- `src/kinetic_chain/metrics.py`：PCE 與容忍度
- `src/kinetic_chain/data.py`：`Clip` 記錄、批次組裝、遮罩
- `src/kinetic_chain/datasets/`：外部資料集 → `Clip` 的轉接層
- `src/kinetic_chain/{train,evaluate,infer,cli}.py`：訓練、評估、推論、命令列
- `src/kinetic_chain/errors.py`：例外階層，全部繼承 `KineticChainError`
- Data / external boundary：GolfDB 標註（`data/raw/golfDB.pkl`）、Penn Action（`data/raw/`）、
  RTMPose ONNX 權重（首次使用自動下載至 `~/.cache/rtmlib`）。無網路服務相依。
- Detailed design and verification：`docs/architecture.md`

# Commands and Verification

- Install / setup: `conda create -y -n kinetic-chain python=3.12 && conda activate kinetic-chain && pip install -e ".[dev]"`；資料取得見 `README.md`
- Format / lint: none confirmed（尚未設定）
- Type / static check: none confirmed
- Unit / integration test: `pytest`（不需 GPU、不需資料集）
- Train（單運動，建議用法）: `python -m kinetic_chain.cli train --sport <id> --val-fold 1 --output runs/<id>`
- Evaluate: `python -m kinetic_chain.cli eval --checkpoint <path>`
- Infer: `python -m kinetic_chain.cli infer --video <path> --sport <id>`
- Benchmark: GolfDB 上的 PCE，對照 SwingNet 已發表數字。實際結果記於 `docs/architecture.md`。

不得宣稱未實際執行的檢查已通過。

# Project-specific Rules

- **新增運動項目不得修改 `model.py`、`train.py` 或 `decode.py`。** 只能修改 `events.py` 的註冊表
  與新增 `datasets/` 下的轉接模組。若某個運動需要動到模型，代表抽象設計有問題，先修設計。
- **預設是一個運動一個模型。** 跨運動聯合訓練（`--sport` 給多個）與跨運動微調
  （`--init-from`）都實測過，都沒有比單運動從頭訓練好（見 `docs/architecture.md` 的
  S4、S6）。這兩條路徑保留可執行，但不得在未重新驗證的情況下當成建議用法。
- **弱標註與真人標註不得混報。** `Clip.label_source` 是必填欄位；任何評估報表必須分開統計
  `human` 與 `weak`，不得合併成單一數字。弱標註上的分數只代表模型學會了規則，
  不代表規則正確。
- 事件順序（`SportSpec.events` 的排列）是力學事實，由解碼器強制保證，不得改成靠 loss 誘導。
- 核心層（`events`/`features`/`model`/`decode`/`metrics`）不得匯入 `rtmlib`、`cv2` 或任何
  推論後端；姿態抽取只在 `pose.py` 與 `infer.py` 中發生。模型必須能在只有 numpy 陣列的
  情況下訓練與測試。
- Canonical 事件的定義一旦寫入 `events.py` 就是跨運動契約，變更前須確認所有已註冊運動
  的語意是否仍然一致。
- 不得將輸入影片、資料集媒體檔或可識別個人的媒體寫入 repository。`data/` 不納入版控。
- 輸出定位為動作分析的時間點標記，不是醫療診斷，也不是動作品質評分。

# Required Rules

- 修改前確認需求、受影響模組、介面、資料、相容性、測試與輸出。
- 修改範圍限於需求與必要連帶調整，不做無關重構或格式大洗版。
- 遵循現有架構、命名、錯誤處理、logging、測試與套件管理方式。
- 不覆蓋或還原使用者既有且與本次無關的變更。
- 不提交密碼、token、私鑰、個資、正式資料或其他機密。
- 新增依賴前確認必要性、相容性、授權與維護風險。
- 改變公開 API、資料格式、schema 或事件詞彙前，核對 `docs/spec.md`，並在
  `docs/architecture.md` 說明相容性與 migration（如適用）。
- 修正缺陷時，在可行範圍內新增能重現問題並防止回歸的測試。
- 完成前執行與風險相稱的檢查，清楚列出未執行項目及原因。
- 缺少自動化測試或外部驗證不阻止保存範圍明確的 commit，但不得因此宣稱功能已完整驗證。
- Python 原始碼、測試、相依套件或 Python 設定的實作、修改、除錯與 review，使用
  `python-code-maintenance` skill；使用前先核對已確認的 Python 版本、工具與相容性限制。

# Documentation Maintenance

- 使用者確認的研究問題、目標、範圍、限制、輸入輸出與成功標準更新至 `docs/spec.md`。
- 穩定的模組責任、資料流、介面、資料模型、演算法、正確性、複雜度、測試與實驗設計更新至
  `docs/architecture.md`。
- 重要設計選擇記錄在 `docs/architecture.md` 的「Design Decisions and Trade-offs」。
- 實驗結果（含失敗的實驗）記錄在「Verification and Experiments」，不只留在聊天紀錄。
- 專案入口、安裝、執行、驗證指令或結果摘要改變時更新 `README.md`。
- 只有存在未完成工作且需要交接時才更新 `HANDOFF.md`。

# Read Documents on Demand

- 需求、成功標準或範圍改變：讀取並更新 `docs/spec.md`。
- 模組、資料流、公開介面、schema、演算法、複雜度、測試或實驗方法改變：讀取並更新
  `docs/architecture.md`。
- 涉及既有設計理由或替代方案：查詢 `docs/architecture.md` 的設計決策與 Git 歷史。
- 接手未完成工作：讀取 `HANDOFF.md`，並完成下列稽核。

# Handoff Audit

接手工作時先不要修改檔案：

1. 讀取 `HANDOFF.md`；若涉及需求或成功標準，再讀取 `docs/spec.md`。
2. 檢查 `git status`、`git diff` 與最近 commits；不是 Git repository 時說明缺口。
3. 核對 handoff 與實際程式碼、設定、測試及驗證輸出。
4. 先回報矛盾、未完成項目、驗證狀態與預計修改範圍，再開始實作。

`HANDOFF.md` 是交接摘要；Git、程式碼、設定與實際測試結果優先。

# Git Commit Rules

- 每完成一個可獨立理解、可獨立檢視且不破壞 repository 基本一致性的子步驟，就建立一個 commit。
- Commit 前檢查 `git status` 與 `git diff`，只納入本次子步驟的變更。
- 無法執行的檢查在 commit 的 `Verification` 記為 `not run` 並說明原因。
- Commit 是版本檢查點，不代表已通過完整驗收或正式環境驗證。
- 不得 amend、rebase 或 force push 已存在的 commits，除非使用者明確要求。
- 不得自行執行 `git init`、建立遠端 repository 或 `git push`，除非使用者明確要求。

小型文件或格式修改可使用：

```text
type(scope): short summary
```

一般功能、修正、跨檔案或行為變更使用：

```text
type(scope): short summary

Why:
- modification reason

Changes:
- important behavior or structure change

Verification:
- checks actually executed
- checks not executed and reason

Risks:
- remaining risk; omit when none
```
