"""外部資料集 → :class:`kinetic_chain.data.Clip` 的轉接層。

每個模組只負責把一個資料集的原始格式讀成統一的 ``Clip``，不做特徵、不碰模型。
新增資料集時只新增這裡的模組，訓練與評估程式不動。
"""

from __future__ import annotations

from . import golfdb, local_video, pennaction

__all__ = ["golfdb", "local_video", "pennaction"]
