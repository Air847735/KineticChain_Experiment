"""全域事件詞彙與運動項目註冊表。

這是新增運動項目的**唯一**入口。模型、訓練迴圈與解碼器都不認識任何特定運動，
只認識這裡宣告的事件槽與順序。

事件 id 分兩類：

``CANONICAL_EVENTS``
    力學定義跨運動相同的節點。多個運動共用同一個輸出槽，因此彼此的資料會互相
    貢獻梯度——這是「一個模型多運動」的實質內容。

``SPORT_SPECIFIC_EVENTS``
    只在單一運動中有定義的節點（如高爾夫的桿頭朝上）。共用不了，也不該共用。

兩者的聯集就是模型輸出頭的大小。任何運動的事件集合都是這個聯集的子集，
推論時以遮罩選出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .errors import SportSpecError, UnknownEventError, UnknownSportError

# --------------------------------------------------------------------------
# 事件詞彙
# --------------------------------------------------------------------------

#: 跨運動共用的動力鏈節點。
#:
#: 這裡的排列是投擲類的典型時序，**不是跨運動的不變式**。每個運動的權威順序是
#: 自己 ``SportSpec.events`` 的排列。實際會分歧：投擲類先舉腿到頂再前腳著地
#: （``loading_peak`` → ``stride_foot_contact``），擊球類則相反——前腳先落地
#: 建立支撐，軀幹才拉到最大分離。
#:
#: 真正跨運動成立的只有 :data:`UNIVERSAL_ORDER` 那幾條。
CANONICAL_EVENTS: tuple[str, ...] = (
    "address",              # 動作開始前的靜止準備
    "loading_start",        # 反向動作（蓄力）開始
    "loading_peak",         # 反向動作最大位置
    "stride_foot_contact",  # 前腳著地，地面反作用力進入動力鏈
    "pelvis_peak_rotation", # 骨盆連線方向角的角速度峰值
    "torso_peak_rotation",  # 肩線方向角的角速度峰值
    "arm_peak_velocity",    # 遠端上肢（腕）線速度峰值
    "release_impact",       # 出手／擊球／觸擊瞬間
    "follow_through_mid",   # 隨勢動作中點
    "finish",               # 動作結束的靜止姿勢
)

#: 每個運動的事件排列都必須滿足的順序約束，寫成 ``(先, 後)`` 的配對。
#:
#: 只列真正跨運動成立的：動作從準備開始、以結束姿勢收尾，中間力量由骨盆經軀幹
#: 傳到遠端上肢再釋放。其餘節點（蓄力頂點、前腳著地）的相對位置因運動而異，
#: 不放在這裡——放進來就會變成用一個運動的習慣去約束另一個運動。
UNIVERSAL_ORDER: tuple[tuple[str, str], ...] = (
    ("address", "loading_start"),
    ("loading_start", "loading_peak"),
    ("pelvis_peak_rotation", "torso_peak_rotation"),
    ("torso_peak_rotation", "arm_peak_velocity"),
    ("arm_peak_velocity", "release_impact"),
    ("release_impact", "follow_through_mid"),
    ("follow_through_mid", "finish"),
)

#: 單一運動專屬的節點。命名一律以運動 id 為前綴，避免誤以為可跨運動共用。
#:
#: 舉重的節點幾乎全在這裡，而不是 canonical——這是實測結果，不是偷懶：
#: canonical 詞彙是從投擲／擊球這類**旋轉型**動作歸納出來的，換到**伸展型**的
#: 舉重就只有 ``address`` / ``arm_peak_velocity`` / ``finish`` 三個對得上。
#: 詳見 `docs/architecture.md` 的「Canonical 詞彙的適用邊界」。
SPORT_SPECIFIC_EVENTS: tuple[str, ...] = (
    "golf_toe_up",          # 上桿至桿身水平（桿頭朝上）
    "golf_mid_backswing",   # 上桿至前臂水平
    "golf_mid_downswing",   # 下桿至前臂水平
    "clean_liftoff",        # 槓鈴離地
    "clean_knee_pass",      # 槓鈴通過膝蓋（第一拉結束）
    "clean_catch",          # 接槓於肩，蹲至最低
    "clean_recovery",       # 由蹲站起
    "clean_jerk_dip",       # 上挺前的預蹲最低點
    "clean_overhead",       # 槓鈴到達最高點（挺舉鎖定）
)

#: 模型輸出頭涵蓋的完整事件槽，順序即輸出索引。
ALL_EVENTS: tuple[str, ...] = CANONICAL_EVENTS + SPORT_SPECIFIC_EVENTS

NUM_EVENT_SLOTS = len(ALL_EVENTS)

EVENT_INDEX: Mapping[str, int] = MappingProxyType(
    {name: i for i, name in enumerate(ALL_EVENTS)}
)


def event_index(event: str) -> int:
    """事件 id → 輸出槽索引。"""
    try:
        return EVENT_INDEX[event]
    except KeyError as exc:
        raise UnknownEventError(
            f"未知的事件 id {event!r}；已註冊的事件為 {ALL_EVENTS}"
        ) from exc


# --------------------------------------------------------------------------
# 弱標註規則
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WeakRule:
    """由力學訊號推導單一事件的宣告式規則。

    規則的實作在 :mod:`kinetic_chain.weak_labels`；這裡只宣告要用哪一條規則、
    吃哪一個訊號。加新運動時寫規則，不寫程式。

    Attributes
    ----------
    event:
        要推導的事件 id。
    rule:
        規則名稱，須存在於 ``weak_labels.RULES``。
    params:
        規則參數。內容由規則自行解讀。
    """

    event: str
    rule: str
    params: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        event_index(self.event)  # 提早失敗：事件 id 打錯時在載入模組時就爆
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


# --------------------------------------------------------------------------
# 運動項目
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SportSpec:
    """一個運動項目的完整宣告。

    Attributes
    ----------
    sport_id:
        運動項目 id，也是模型 sport embedding 的查表鍵。
    display_name:
        人看的名稱。
    events:
        該運動可能標註的事件，**依時序排列**。單一片段不必提供全部事件；
        實際有哪些由 ``Clip.events`` 決定，缺的槽在訓練與解碼時被遮罩掉。
    handedness_sensitive:
        是否需要左右鏡射正規化，把左打／左投統一成同一個方向。
    weak_rules:
        弱標註規則。只有沒有真人標註的資料集才會用到。
    """

    sport_id: str
    display_name: str
    events: tuple[str, ...]
    handedness_sensitive: bool = True
    weak_rules: tuple[WeakRule, ...] = ()

    def __post_init__(self) -> None:
        if not self.events:
            raise SportSpecError(f"{self.sport_id!r} 沒有宣告任何事件")
        if len(set(self.events)) != len(self.events):
            raise SportSpecError(f"{self.sport_id!r} 的事件有重複：{self.events}")
        for name in self.events:
            event_index(name)
        position = {event: i for i, event in enumerate(self.events)}
        for earlier, later in UNIVERSAL_ORDER:
            if earlier in position and later in position and position[earlier] > position[later]:
                raise SportSpecError(
                    f"{self.sport_id!r} 把 {later!r} 排在 {earlier!r} 之前，"
                    "違反跨運動的順序約束 UNIVERSAL_ORDER"
                )
        declared = set(self.events)
        for rule in self.weak_rules:
            if rule.event not in declared:
                raise SportSpecError(
                    f"{self.sport_id!r} 的弱標註規則產生了未宣告的事件 {rule.event!r}"
                )

    @property
    def slots(self) -> tuple[int, ...]:
        """該運動的事件對應到的輸出槽索引，順序同 ``events``。"""
        return tuple(event_index(name) for name in self.events)

    def order_of(self, event: str) -> int:
        """事件在該運動時序中的位置。"""
        try:
            return self.events.index(event)
        except ValueError as exc:
            raise UnknownEventError(
                f"{event!r} 不在 {self.sport_id!r} 宣告的事件中"
            ) from exc


def _peak(event: str, signal: str, **params: object) -> WeakRule:
    return WeakRule(event, "signal_peak", {"signal": signal, **params})


#: 舉重（挺舉）的分期依 IWF 與生物力學文獻：
#: 起始 → 第一拉（離地至膝）→ 過渡（雙膝彎曲）→ 第二拉（三重伸展）→
#: 接槓 → 站起 → 上挺預蹲 → 上挺發力 → 過頭鎖定。
#: 這裡量得到的是其中有 2D 姿態對應的節點；槓鈴本身沒有關鍵點，以手腕代理。
_LIFT_RULES: tuple[WeakRule, ...] = (
    WeakRule("address", "rest_start"),
    # 過頭最高點：全片段手腕最高，是整個動作最不會誤判的錨點，先定它
    WeakRule("clean_overhead", "signal_extreme", {"signal": "wrist_height", "mode": "max"}),
    # 接槓：骨盆下沉最多的一刻。踝高是相對骨盆的，蹲得越低這個值越大
    WeakRule(
        "clean_catch",
        "signal_extreme",
        {"signal": "lead_ankle_height", "mode": "max", "before": "clean_overhead"},
    ),
    WeakRule(
        "arm_peak_velocity",
        "signal_peak",
        {"signal": "wrist_speed", "after": "clean_catch", "before": "clean_overhead"},
    ),
    WeakRule(
        "clean_recovery",
        "signal_extreme",
        {"signal": "lead_ankle_height", "mode": "min",
         "after": "clean_catch", "before": "arm_peak_velocity"},
    ),
    WeakRule(
        "clean_jerk_dip",
        "signal_extreme",
        {"signal": "lead_ankle_height", "mode": "max",
         "after": "clean_recovery", "before": "arm_peak_velocity"},
    ),
    WeakRule(
        "clean_liftoff",
        "signal_onset",
        {"signal": "wrist_height", "after": "address", "before": "clean_catch"},
    ),
    WeakRule(
        "clean_knee_pass",
        "signal_crossing",
        {"signal": "wrist_height", "reference": "lead_knee_height",
         "after": "clean_liftoff", "before": "clean_catch"},
    ),
    WeakRule("finish", "rest_end"),
)

#: 投擲／擊球類共通的近端到遠端規則。
#:
#: 求解順序刻意是**由遠端往近端**，與力學上的傳遞方向相反：手腕速度峰值是三個
#: 訊號中最穩健的（2D 投影下腕點位置明確，而髖線與肩線在側面視角會塌成一點，
#: 方向角變得極不穩定），先定出它，再往回在加速階段內找軀幹與骨盆的峰值。
#:
#: **這使近端到遠端的順序由建構方式保證，而不是由資料驗證出來的。** 隨勢階段的
#: 骨盆／軀幹旋轉常常大於加速階段，若取全片段極值會落在擊球之後；限制在加速階段
#: 內搜尋才符合「投擲動作的骨盆峰值」這個定義。代價是這批弱標註**不能**用來檢驗
#: 近端到遠端假說是否成立——那是被預設進去的，不是被測出來的。
_CHAIN_RULES: tuple[WeakRule, ...] = (
    _peak("arm_peak_velocity", "wrist_speed"),
    _peak("torso_peak_rotation", "torso_angular_speed", before="arm_peak_velocity"),
    _peak("pelvis_peak_rotation", "pelvis_angular_speed", before="torso_peak_rotation"),
    WeakRule(
        "release_impact",
        "post_peak_decel",
        {"signal": "wrist_speed", "after": "arm_peak_velocity"},
    ),
)

_SPORTS: dict[str, SportSpec] = {}


def register_sport(spec: SportSpec, *, replace: bool = False) -> SportSpec:
    """把運動項目加入註冊表。"""
    if spec.sport_id in _SPORTS and not replace:
        raise SportSpecError(f"運動項目 {spec.sport_id!r} 已註冊")
    _SPORTS[spec.sport_id] = spec
    return spec


def get_sport(sport_id: str) -> SportSpec:
    """取得運動項目定義。找不到就拋錯，不做模糊比對。"""
    try:
        return _SPORTS[sport_id]
    except KeyError as exc:
        raise UnknownSportError(
            f"未註冊的運動項目 {sport_id!r}；已註冊的有 {sorted(_SPORTS)}"
        ) from exc


def registered_sports() -> tuple[str, ...]:
    """已註冊的運動項目 id，排序後回傳（作為 embedding 索引須穩定）。"""
    return tuple(sorted(_SPORTS))


def sport_index(sport_id: str) -> int:
    """運動項目 id → embedding 索引。"""
    sports = registered_sports()
    try:
        return sports.index(sport_id)
    except ValueError as exc:
        raise UnknownSportError(
            f"未註冊的運動項目 {sport_id!r}；已註冊的有 {list(sports)}"
        ) from exc


def event_mask(sport_id: str, present: Mapping[str, int] | None = None) -> np.ndarray:
    """回傳長度 ``NUM_EVENT_SLOTS`` 的布林遮罩。

    Parameters
    ----------
    sport_id:
        運動項目 id。
    present:
        該片段實際具備的事件。``None`` 表示採用該運動宣告的全部事件。
        提供時只有同時被運動宣告、且該片段有標註的事件才會被啟用——
        資料集標了運動定義以外的事件時視為資料錯誤，不靜默接受。
    """
    spec = get_sport(sport_id)
    mask = np.zeros(NUM_EVENT_SLOTS, dtype=bool)
    names = spec.events if present is None else tuple(present)
    for name in names:
        if present is not None and name not in spec.events:
            raise UnknownEventError(
                f"片段標了 {name!r}，但 {sport_id!r} 沒有宣告這個事件"
            )
        mask[event_index(name)] = True
    return mask


# --------------------------------------------------------------------------
# 內建運動項目
# --------------------------------------------------------------------------

register_sport(
    SportSpec(
        sport_id="golf_swing",
        display_name="高爾夫揮桿",
        # 前 8 個 canonical/specific 事件對應 GolfDB 的真人標註；
        # 中間三個動力鏈峰值只有弱標註來源會提供，GolfDB 片段會遮罩掉。
        events=(
            "address",
            "golf_toe_up",
            "golf_mid_backswing",
            "loading_peak",
            "golf_mid_downswing",
            "pelvis_peak_rotation",
            "torso_peak_rotation",
            "arm_peak_velocity",
            "release_impact",
            "follow_through_mid",
            "finish",
        ),
        weak_rules=(
            WeakRule("address", "rest_start"),
            # 上桿頂點：手腕最高（影像 y 向下，故取高度訊號的最大值）
            WeakRule(
                "loading_peak",
                "signal_extreme",
                {
                    "signal": "wrist_height",
                    "mode": "max",
                    "after": "address",
                    "before": "pelvis_peak_rotation",
                },
            ),
            *_CHAIN_RULES,
            WeakRule(
                "follow_through_mid",
                "midpoint",
                {"start": "release_impact", "end": "finish"},
            ),
            WeakRule("finish", "rest_end"),
        ),
    )
)

register_sport(
    SportSpec(
        sport_id="baseball_pitch",
        display_name="棒球投球",
        events=(
            "address",
            "loading_start",
            "loading_peak",
            "stride_foot_contact",
            "pelvis_peak_rotation",
            "torso_peak_rotation",
            "arm_peak_velocity",
            "release_impact",
            "follow_through_mid",
            "finish",
        ),
        weak_rules=(
            WeakRule("address", "rest_start"),
            WeakRule(
                "loading_start",
                "signal_onset",
                {
                    "signal": "lead_knee_height",
                    "after": "address",
                    "before": "loading_peak",
                },
            ),
            # 最大舉腿：前側膝最高
            WeakRule(
                "loading_peak",
                "signal_extreme",
                {
                    "signal": "lead_knee_height",
                    "mode": "max",
                    "after": "address",
                    "before": "pelvis_peak_rotation",
                },
            ),
            WeakRule(
                "stride_foot_contact",
                "foot_contact",
                {"after": "loading_peak", "before": "pelvis_peak_rotation"},
            ),
            *_CHAIN_RULES,
            WeakRule(
                "follow_through_mid",
                "midpoint",
                {"start": "release_impact", "end": "finish"},
            ),
            WeakRule("finish", "rest_end"),
        ),
    )
)

register_sport(
    SportSpec(
        sport_id="baseball_swing",
        display_name="棒球揮棒",
        # 擊球類的順序與投擲類不同：前腳先落地建立支撐，軀幹才拉到最大分離。
        # `loading_start` 對應文獻的 lead foot off（跨步開始），是打擊六期分法的
        # 五個關鍵事件之一：lead foot off → lead foot down → 重心轉移 →
        # 前腳最大垂直地面反作用力 → 觸球。
        events=(
            "address",
            "loading_start",
            "stride_foot_contact",
            "loading_peak",
            "pelvis_peak_rotation",
            "torso_peak_rotation",
            "arm_peak_velocity",
            "release_impact",
            "follow_through_mid",
            "finish",
        ),
        weak_rules=(
            WeakRule("address", "rest_start"),
            # 前腳離地：前踝高度開始上升，在著地之前
            WeakRule(
                "loading_start",
                "signal_onset",
                {
                    "signal": "lead_ankle_height",
                    "after": "address",
                    "before": "stride_foot_contact",
                },
            ),
            # 拉棒到底：肩髖分離角最大
            WeakRule(
                "loading_peak",
                "signal_extreme",
                {
                    "signal": "separation_angle",
                    "mode": "max",
                    "after": "stride_foot_contact",
                    "before": "pelvis_peak_rotation",
                },
            ),
            WeakRule(
                "stride_foot_contact",
                "foot_contact",
                {"after": "address", "before": "pelvis_peak_rotation"},
            ),
            *_CHAIN_RULES,
            WeakRule(
                "follow_through_mid",
                "midpoint",
                {"start": "release_impact", "end": "finish"},
            ),
            WeakRule("finish", "rest_end"),
        ),
    )
)

register_sport(
    SportSpec(
        sport_id="tennis_serve",
        display_name="網球發球",
        events=(
            "address",
            "loading_start",
            "loading_peak",
            "pelvis_peak_rotation",
            "torso_peak_rotation",
            "arm_peak_velocity",
            "release_impact",
            "follow_through_mid",
            "finish",
        ),
        weak_rules=(
            WeakRule("address", "rest_start"),
            WeakRule(
                "loading_start",
                "signal_onset",
                {
                    "signal": "wrist_height",
                    "after": "address",
                    "before": "loading_peak",
                },
            ),
            # 引拍到底（racket drop）：擊球手腕在擊球前的最低點
            WeakRule(
                "loading_peak",
                "signal_extreme",
                {
                    "signal": "wrist_height",
                    "mode": "min",
                    "after": "address",
                    "before": "pelvis_peak_rotation",
                },
            ),
            *_CHAIN_RULES,
            WeakRule(
                "follow_through_mid",
                "midpoint",
                {"start": "release_impact", "end": "finish"},
            ),
            WeakRule("finish", "rest_end"),
        ),
    )
)

register_sport(
    SportSpec(
        sport_id="tennis_forehand",
        display_name="網球正手拍",
        # 擊球類的順序與投擲類不同：前腳先落地建立支撐，軀幹才拉到最大分離。
        events=(
            "address",
            "stride_foot_contact",
            "loading_peak",
            "pelvis_peak_rotation",
            "torso_peak_rotation",
            "arm_peak_velocity",
            "release_impact",
            "follow_through_mid",
            "finish",
        ),
        weak_rules=(
            WeakRule("address", "rest_start"),
            WeakRule(
                "loading_peak",
                "signal_extreme",
                {
                    "signal": "separation_angle",
                    "mode": "max",
                    "after": "stride_foot_contact",
                    "before": "pelvis_peak_rotation",
                },
            ),
            WeakRule(
                "stride_foot_contact",
                "foot_contact",
                {"after": "address", "before": "pelvis_peak_rotation"},
            ),
            *_CHAIN_RULES,
            WeakRule(
                "follow_through_mid",
                "midpoint",
                {"start": "release_impact", "end": "finish"},
            ),
            WeakRule("finish", "rest_end"),
        ),
    )
)

register_sport(
    SportSpec(
        sport_id="clean_and_jerk",
        display_name="舉重挺舉",
        # 只有頭尾與上肢峰值三個是 canonical；其餘全是專屬節點。
        # 動力鏈本身也不同：舉重是髖→膝的伸展序列，不是骨盆→軀幹的旋轉序列。
        events=(
            "address",
            "clean_liftoff",
            "clean_knee_pass",
            "clean_catch",
            "clean_recovery",
            "clean_jerk_dip",
            "arm_peak_velocity",
            "clean_overhead",
            "finish",
        ),
        # 對稱動作，沒有慣用邊可言；鏡射只會製造不必要的變異
        handedness_sensitive=False,
        weak_rules=_LIFT_RULES,
    )
)

register_sport(
    SportSpec(
        sport_id="bowling",
        display_name="保齡球",
        events=(
            "address",
            "loading_start",
            "loading_peak",
            "stride_foot_contact",
            "pelvis_peak_rotation",
            "torso_peak_rotation",
            "arm_peak_velocity",
            "release_impact",
            "follow_through_mid",
            "finish",
        ),
        weak_rules=(
            WeakRule("address", "rest_start"),
            WeakRule(
                "loading_start",
                "signal_onset",
                {
                    "signal": "wrist_height",
                    "after": "address",
                    "before": "loading_peak",
                },
            ),
            # 後擺最高點
            WeakRule(
                "loading_peak",
                "signal_extreme",
                {
                    "signal": "wrist_height",
                    "mode": "max",
                    "after": "address",
                    "before": "pelvis_peak_rotation",
                },
            ),
            WeakRule(
                "stride_foot_contact",
                "foot_contact",
                {"after": "loading_peak", "before": "pelvis_peak_rotation"},
            ),
            *_CHAIN_RULES,
            WeakRule(
                "follow_through_mid",
                "midpoint",
                {"start": "release_impact", "end": "finish"},
            ),
            WeakRule("finish", "rest_end"),
        ),
    )
)
