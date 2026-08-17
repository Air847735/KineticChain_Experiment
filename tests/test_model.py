"""模型、批次組裝與弱標註推導。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kinetic_chain.data import Clip, collate, split_clips
from kinetic_chain.errors import DatasetError, WeakLabelError
from kinetic_chain.events import NUM_EVENT_SLOTS, event_index, get_sport, sport_index
from kinetic_chain.features import NUM_FEATURES, compute
from kinetic_chain.model import KineticChainNet, ModelConfig, event_loss, soft_targets
from kinetic_chain.weak_labels import derive

from .conftest import synthetic_pose


def make_clip(clip_id: str = "t/0", sport: str = "baseball_pitch", frames: int = 120) -> Clip:
    pose = synthetic_pose(frames)
    spec = get_sport(sport)
    signals = compute(pose, 30.0, handedness_sensitive=spec.handedness_sensitive)
    return Clip(
        clip_id=clip_id,
        sport=sport,
        pose=pose,
        fps=30.0,
        events=derive(signals, spec),
        label_source="weak",
        dataset="synthetic",
    )


# --------------------------------------------------------------------------
# 弱標註
# --------------------------------------------------------------------------


def test_weak_labels_follow_proximal_to_distal_order():
    clip = make_clip()
    frames = [clip.events[e] for e in clip.ordered_events]
    assert frames == sorted(frames)


def test_weak_labels_recover_the_synthetic_ground_truth():
    """合成資料的骨盆峰值在 0.55、軀幹在 0.62、腕速峰值在 0.70。"""
    clip = make_clip(frames=200)
    assert clip.events["pelvis_peak_rotation"] == pytest.approx(0.55 * 200, abs=12)
    assert clip.events["torso_peak_rotation"] == pytest.approx(0.62 * 200, abs=12)
    assert clip.events["arm_peak_velocity"] == pytest.approx(0.70 * 200, abs=12)


def test_weak_labels_reject_a_sport_without_rules():
    from kinetic_chain.events import SportSpec

    spec = SportSpec("no_rules", "無規則", ("address", "finish"))
    signals = compute(synthetic_pose(), 30.0)
    with pytest.raises(WeakLabelError):
        derive(signals, spec)


def test_weak_label_order_violation_is_raised_not_silently_fixed():
    """順序違反必須讓片段被丟掉，不能靜默重排成看起來合理的樣子。"""
    from kinetic_chain.events import SportSpec, WeakRule

    spec = SportSpec(
        "reversed",
        "順序顛倒",
        ("address", "finish"),
        weak_rules=(
            # 故意把 address 綁到片尾、finish 綁到片頭
            WeakRule("address", "rest_end"),
            WeakRule("finish", "rest_start"),
        ),
    )
    signals = compute(synthetic_pose(), 30.0)
    assert derive(signals, spec, enforce_order=False)  # 不檢查時可以推導
    with pytest.raises(WeakLabelError):
        derive(signals, spec)


# --------------------------------------------------------------------------
# Clip 與批次
# --------------------------------------------------------------------------


def test_clip_rejects_invalid_label_source():
    with pytest.raises(DatasetError):
        Clip("x", "baseball_pitch", synthetic_pose(), 30.0, {}, "guessed")  # type: ignore[arg-type]


def test_collate_pads_and_masks():
    clips = [make_clip("a", frames=80), make_clip("b", frames=140)]
    batch = collate(clips)
    assert batch.features.shape == (2, 140, NUM_FEATURES)
    assert batch.frame_mask[0].sum() == 80
    assert batch.frame_mask[1].all()
    assert batch.event_mask.shape == (2, NUM_EVENT_SLOTS)
    assert batch.sport_ids.tolist() == [sport_index("baseball_pitch")] * 2


def test_collate_only_activates_events_the_clip_actually_has():
    clip = make_clip()
    del clip.events["stride_foot_contact"]
    batch = collate([clip])
    assert not batch.event_mask[0, event_index("stride_foot_contact")]
    assert batch.event_mask[0].sum() == len(clip.events)


def test_collate_rejects_event_outside_clip():
    clip = make_clip(frames=60)
    clip.events["finish"] = 999
    with pytest.raises(DatasetError):
        collate([clip])


def test_split_is_stratified_by_sport():
    clips = [make_clip(f"p{i}", "baseball_pitch") for i in range(10)]
    clips += [make_clip(f"g{i}", "golf_swing") for i in range(10)]
    train, val = split_clips(clips, val_fraction=0.2, seed=0)
    assert {c.sport for c in val} == {"baseball_pitch", "golf_swing"}
    assert len(train) + len(val) == 20
    assert not {c.clip_id for c in train} & {c.clip_id for c in val}


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------


def test_forward_shape_and_padding_is_masked_out():
    model = KineticChainNet(ModelConfig(hidden=32, num_layers=2))
    features = torch.randn(2, 50, NUM_FEATURES)
    frame_mask = torch.ones(2, 50, dtype=torch.bool)
    frame_mask[1, 30:] = False
    logits = model(features, torch.tensor([0, 1]), frame_mask)
    assert logits.shape == (2, NUM_EVENT_SLOTS, 50)
    assert torch.isneginf(logits[1, :, 30:]).all()
    assert torch.isfinite(logits[0]).all()


def test_sport_conditioning_actually_changes_the_output():
    """FiLM 若沒接上，換運動項目輸出會完全一樣——這個測試就是在防那件事。"""
    torch.manual_seed(0)
    model = KineticChainNet(ModelConfig(hidden=32, num_layers=2))
    # FiLM 初始化為恆等，先擾動權重讓條件產生作用
    for module in model.modules():
        if hasattr(module, "to_gamma"):
            torch.nn.init.normal_(module.to_gamma.weight, std=0.1)
            torch.nn.init.normal_(module.to_beta.weight, std=0.1)
    model.eval()
    features = torch.randn(1, 40, NUM_FEATURES)
    with torch.no_grad():
        a = model(features, torch.tensor([0]))
        b = model(features, torch.tensor([1]))
    assert not torch.allclose(a, b)


def test_soft_targets_sum_to_one_and_peak_at_the_label():
    frames = torch.tensor([[10, 25]])
    targets = soft_targets(frames, length=40, sigma=1.5)
    assert torch.allclose(targets.sum(dim=-1), torch.ones(1, 2), atol=1e-5)
    assert targets[0, 0].argmax().item() == 10
    assert targets[0, 1].argmax().item() == 25


def test_soft_targets_put_no_mass_on_padding():
    frame_mask = torch.zeros(1, 40, dtype=torch.bool)
    frame_mask[0, :20] = True
    targets = soft_targets(torch.tensor([[10]]), 40, 1.5, frame_mask)
    assert targets[0, 0, 20:].sum() == 0
    assert targets[0, 0].sum() == pytest.approx(1.0, abs=1e-5)


def test_loss_is_zero_when_prediction_matches_target():
    logits = torch.log(soft_targets(torch.tensor([[5]]), 20, 1.5) + 1e-12)
    targets = soft_targets(torch.tensor([[5]]), 20, 1.5)
    mask = torch.ones(1, 1, dtype=torch.bool)
    assert float(event_loss(logits, targets, mask)) == pytest.approx(0.0, abs=1e-4)


def test_loss_ignores_inactive_event_slots():
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 20)
    targets = soft_targets(torch.tensor([[5, 10, 15]]), 20, 1.5)
    active_only = event_loss(logits, targets, torch.tensor([[True, False, False]]))
    # 把被遮罩的槽換成完全不同的預測，損失不應改變
    logits[0, 1:] = torch.randn(2, 20) * 10
    assert float(
        event_loss(logits, targets, torch.tensor([[True, False, False]]))
    ) == pytest.approx(float(active_only))


def test_masked_padding_does_not_produce_nan_loss():
    model = KineticChainNet(ModelConfig(hidden=32, num_layers=2))
    batch = collate([make_clip("a", frames=60), make_clip("b", frames=120)])
    logits = model(batch.features, batch.sport_ids, batch.frame_mask)
    targets = soft_targets(batch.targets, logits.shape[-1], batch.sigma, batch.frame_mask)
    loss = event_loss(logits, targets, batch.event_mask)
    assert torch.isfinite(loss)


def test_receptive_field_covers_a_typical_clip():
    """感受野必須大於典型片段長度，否則模型看不到動作的另一端。"""
    assert ModelConfig().receptive_field >= 300
