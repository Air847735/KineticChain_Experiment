# Handoff

- Status: `no-handoff`
- Updated: 2026-08-16

首次完整實作已完成，86 個測試通過，四折對照實驗已執行。後續方向記於
`docs/spec.md` 的 Open Questions 與 `docs/architecture.md` 的 Known Gaps。

需要注意的環境細節（不重裝環境的話不會遇到）：`rtmlib` 與 `onnxruntime-gpu` 的版本
衝突會讓姿態抽取**靜默**退回 CPU，處理方式見 `README.md` 的 Setup。
