"""Penn Action 轉接層（**弱標註**）。

Penn Action 提供 2326 段裁切好的單人運動片段，每一影格都有 13 個關節的人工標註
——但**沒有事件標註**。因此關鍵點是真值，事件時間點是由
:mod:`kinetic_chain.weak_labels` 的運動學規則推導出來的弱標註。

之所以還是用它：它是少數同時具備「多種運動」「單人裁切片段」「逐影格關節標註」
的公開資料集，而且關節標註是人工的，不必先跑一次姿態抽取，讓多運動的部分能在
不引入姿態誤差的條件下驗證。

資料取得
--------
``https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz``（約 3.2 GB）
解壓後結構為 ``Penn_Action/{frames,labels}/{seq_id}``。本模組只讀 ``labels/*.mat``。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from ..data import Clip
from ..errors import DatasetError, KineticChainError
from ..events import get_sport
from ..features import compute
from ..skeleton import to_canonical
from ..weak_labels import derive

logger = logging.getLogger(__name__)

DATASET_NAME = "penn_action"

#: Penn Action 標註 30 fps 的 YouTube 片段，資料集本身不記錄影格率。
#: 用於把速度換算成每秒並決定平滑窗長；不同的假設會等比縮放所有速度訊號，
#: 不影響極值位置，因此對事件推導不敏感。
ASSUMED_FPS = 30.0

#: Penn Action 標註的關鍵點布局。布局屬於資料來源，不屬於運動項目——
#: 同一個運動（如高爾夫）可以同時來自 Penn Action 與 GolfDB，布局不同。
LAYOUT = "penn13"

#: Penn Action 的 ``action`` 欄位 → 本專案的運動項目 id。
#: 未列出的動作（伏地挺身、深蹲、彈吉他等）沒有單次彈道式的動力鏈事件，不納入。
ACTION_TO_SPORT: dict[str, str] = {
    "baseball_pitch": "baseball_pitch",
    "baseball_swing": "baseball_swing",
    "golf_swing": "golf_swing",
    "tennis_serve": "tennis_serve",
    "tennis_forehand": "tennis_forehand",
    "bowl": "bowling",
}


def _as_scalar_string(value: object) -> str:
    """把 ``scipy.io.loadmat`` 讀出的字串欄位攤平成 ``str``。"""
    array = np.asarray(value).ravel()
    if array.size == 0:
        return ""
    item = array[0]
    while isinstance(item, np.ndarray):
        flat = item.ravel()
        if flat.size == 0:
            return ""
        item = flat[0]
    return str(item).strip()


def _load_label(path: Path) -> dict[str, object]:
    from scipy.io import loadmat  # 延遲匯入：核心層不該因為資料集而多帶相依

    try:
        return loadmat(str(path))
    except Exception as exc:  # noqa: BLE001 - loadmat 的失敗型別不穩定
        raise DatasetError(f"無法讀取 Penn Action 標註 {path}：{exc}") from exc


def _pose_from_label(mat: dict[str, object]) -> np.ndarray:
    """把 ``x``/``y``/``visibility`` 組成 ``(T, 13, 3)``。"""
    try:
        x = np.asarray(mat["x"], dtype=np.float32)
        y = np.asarray(mat["y"], dtype=np.float32)
        visible = np.asarray(mat["visibility"], dtype=np.float32)
    except KeyError as exc:
        raise DatasetError(f"Penn Action 標註缺少欄位 {exc}") from exc

    if not (x.shape == y.shape == visible.shape):
        raise DatasetError(
            f"Penn Action 的 x/y/visibility 形狀不一致：{x.shape}, {y.shape}, {visible.shape}"
        )
    if x.ndim != 2 or x.shape[1] != 13:
        raise DatasetError(f"Penn Action 的關節數應為 13，收到 {x.shape}")

    return np.stack([x, y, visible], axis=-1)


def iter_label_paths(root: Path) -> Iterator[Path]:
    """列出 ``labels/*.mat``。接受解壓後的根目錄或 ``labels`` 目錄本身。"""
    root = Path(root)
    labels = root / "labels" if (root / "labels").is_dir() else root
    if not labels.is_dir():
        raise DatasetError(f"找不到 Penn Action 的 labels 目錄：{root}")
    return iter(sorted(labels.glob("*.mat")))


def load(
    root: Path | str,
    *,
    sports: Sequence[str] | None = None,
    min_coverage: float = 0.8,
    limit: int | None = None,
) -> list[Clip]:
    """讀取 Penn Action 並產生帶弱標註的 :class:`Clip`。

    Parameters
    ----------
    root:
        解壓後的 ``Penn_Action`` 目錄。
    sports:
        只保留這些運動項目 id。``None`` 表示全部支援的項目。
    min_coverage:
        關鍵點可見率低於此值的片段捨棄。
    limit:
        只讀前 N 段，供快速冒煙測試。

    Notes
    -----
    弱標註推導失敗（違反宣告的事件時序、依賴解不出來）的片段會被**捨棄**並記錄，
    不會以「盡量湊」的方式留下。推導不出合序的事件通常代表片段本身不完整或
    關節標註品質差，硬留下只會把雜訊餵進訓練。
    """
    wanted = set(sports) if sports else set(ACTION_TO_SPORT.values())
    clips: list[Clip] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for path in iter_label_paths(Path(root)):
        if limit is not None and len(clips) >= limit:
            break
        mat = _load_label(path)
        action = _as_scalar_string(mat.get("action", ""))
        sport = ACTION_TO_SPORT.get(action)
        if sport is None or sport not in wanted:
            skip(f"action={action or 'unknown'}")
            continue

        spec = get_sport(sport)
        pose = to_canonical(_pose_from_label(mat), LAYOUT)

        try:
            signals = compute(
                pose, ASSUMED_FPS, handedness_sensitive=spec.handedness_sensitive
            )
        except KineticChainError as exc:
            logger.debug("%s: 特徵計算失敗 %s", path.stem, exc)
            skip("features_failed")
            continue

        if signals.coverage < min_coverage:
            skip("low_coverage")
            continue

        try:
            events = derive(signals, spec)
        except KineticChainError as exc:
            logger.debug("%s: 弱標註失敗 %s", path.stem, exc)
            skip("weak_label_invalid")
            continue

        clip = Clip(
            clip_id=f"{DATASET_NAME}/{path.stem}",
            sport=sport,
            pose=pose,
            fps=ASSUMED_FPS,
            events=events,
            label_source="weak",
            dataset=DATASET_NAME,
            coverage=signals.coverage,
        )
        clip._signals = signals  # 已經算過，不重算
        clips.append(clip)

    logger.info(
        "Penn Action: 讀入 %d 段；捨棄 %s",
        len(clips),
        {k: v for k, v in sorted(skipped.items()) if not k.startswith("action=")},
    )
    return clips
