"""例外階層。所有本專案拋出的例外都繼承 :class:`KineticChainError`。"""

from __future__ import annotations


class KineticChainError(Exception):
    """本專案所有例外的基底。"""


class UnknownSportError(KineticChainError):
    """要求的運動項目 id 不在事件註冊表中。不做模糊比對。"""


class UnknownEventError(KineticChainError):
    """引用了不存在於全域事件詞彙的事件 id。"""


class SportSpecError(KineticChainError):
    """運動項目定義本身不合法（事件重複、順序矛盾、布局未知等）。"""


class ClipTooShortError(KineticChainError):
    """片段影格數不足以容納該運動項目宣告的事件數。"""


class PoseExtractionError(KineticChainError):
    """姿態抽取失敗（影片無法解碼、後端不可用等）。"""


class DatasetError(KineticChainError):
    """外部資料集缺失、格式不符或內容不一致。"""


class WeakLabelError(KineticChainError):
    """弱標註推導失敗，例如推導結果違反宣告的事件順序。"""
