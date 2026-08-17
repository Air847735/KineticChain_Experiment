"""端到端：訓練 → 評估 → 推論，全部用合成資料，不需 GPU 或外部資料集。

重點不在分數高低，而在管線接得起來、不變式成立、checkpoint 能來回。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kinetic_chain.data import split_clips
from kinetic_chain.errors import ClipTooShortError, DatasetError
from kinetic_chain.evaluate import (
    evaluate_clips,
    format_reports,
    per_event_delta_table,
    predict_clips,
    sequence_violations,
)
from kinetic_chain.infer import predict_pose_sequence
from kinetic_chain.model import ModelConfig
from kinetic_chain.train import TrainConfig, augment_clip, load_checkpoint, train

from .conftest import synthetic_pose
from .test_model import make_clip


@pytest.fixture(scope="module")
def trained():
    clips = [
        make_clip(f"{sport}/{i}", sport, frames=90 + 7 * i)
        for sport in ("baseball_pitch", "tennis_serve")
        for i in range(12)
    ]
    train_clips, val_clips = split_clips(clips, val_fraction=0.25, seed=0)
    config = TrainConfig(
        epochs=8,
        batch_size=4,
        learning_rate=3e-3,
        warmup_epochs=1,
        device="cpu",
        model=ModelConfig(hidden=32, num_layers=3),
    )
    model, history = train(train_clips, val_clips, config)
    return model, train_clips, val_clips, history


def test_training_reduces_loss(trained):
    _, _, _, history = trained
    losses = [e["train_loss"] for e in history["epochs"]]
    assert losses[-1] < losses[0]


def test_predictions_never_violate_declared_order(trained):
    """架構文件的不變式 I1。解碼器保證，這裡驗證那個保證真的成立。"""
    model, _, val_clips, _ = trained
    predictions = predict_clips(model, val_clips)
    assert sequence_violations(val_clips, predictions) == 0


def test_predictions_cover_exactly_the_annotated_events(trained):
    model, _, val_clips, _ = trained
    for clip, prediction in zip(val_clips, predict_clips(model, val_clips)):
        assert set(prediction) == set(clip.ordered_events)
        assert all(0 <= f < clip.num_frames for f in prediction.values())


def test_reports_are_grouped_by_sport_and_label_source(trained):
    model, _, val_clips, _ = trained
    reports = evaluate_clips(model, val_clips)
    assert "baseball_pitch/weak" in reports
    assert "tennis_serve/weak" in reports
    assert "overall" in reports
    assert isinstance(format_reports(reports), str)


def test_per_event_delta_table_has_one_row_per_sport_event(trained):
    model, _, val_clips, _ = trained
    table = per_event_delta_table(val_clips, predict_clips(model, val_clips))
    assert all(key.count("/") == 1 for key in table)
    assert all(row["n"] > 0 for row in table.values())


def test_checkpoint_round_trip(trained, tmp_path):
    from kinetic_chain.train import save_checkpoint

    model, _, val_clips, _ = trained
    path = tmp_path / "model.pt"
    save_checkpoint(model, TrainConfig(), path)
    restored = load_checkpoint(path)
    before = predict_clips(model, val_clips)
    after = predict_clips(restored, val_clips)
    assert before == after


def test_checkpoint_rejects_a_changed_sport_registry(trained, tmp_path, monkeypatch):
    """sport embedding 的索引由註冊表排序決定，註冊表改了就必須拒絕載入。"""
    from kinetic_chain import train as train_module
    from kinetic_chain.train import save_checkpoint

    model, *_ = trained
    path = tmp_path / "model.pt"
    save_checkpoint(model, TrainConfig(), path)

    import kinetic_chain.events as events_module

    monkeypatch.setattr(events_module, "registered_sports", lambda: ("only_one_sport",))
    with pytest.raises(DatasetError):
        load_checkpoint(path)


def test_inference_from_pose_sequence(trained):
    model, *_ = trained
    result = predict_pose_sequence(model, synthetic_pose(150), 30.0, "baseball_pitch")
    assert [e.event for e in result.events] == list(
        result.events[0].event and __import__(
            "kinetic_chain.events", fromlist=["get_sport"]
        ).get_sport("baseball_pitch").events
    )
    frames = [e.frame for e in result.events]
    assert frames == sorted(frames)
    assert all(0.0 <= e.confidence <= 1.0 for e in result.events)
    assert result.events[-1].time == pytest.approx(frames[-1] / 30.0)


def test_inference_rejects_clip_too_short_for_the_event_set(trained):
    model, *_ = trained
    with pytest.raises(ClipTooShortError):
        predict_pose_sequence(model, synthetic_pose(9), 30.0, "baseball_pitch")


def test_augmentation_rescales_events_with_the_clip():
    clip = make_clip(frames=100)
    rng = np.random.default_rng(0)
    config = TrainConfig(time_scale_range=(2.0, 2.0), feature_noise=0.0)
    augmented = augment_clip(clip, config, rng)

    assert augmented.features().shape[0] == pytest.approx(200, abs=2)
    ratio = (augmented.features().shape[0] - 1) / (clip.features().shape[0] - 1)
    for name, frame in clip.events.items():
        assert augmented.events[name] == pytest.approx(frame * ratio, abs=1)
    # 增強不得改動原片段
    assert clip.features().shape[0] == 100


def test_augmented_event_order_is_preserved():
    clip = make_clip(frames=120)
    rng = np.random.default_rng(1)
    config = TrainConfig(time_scale_range=(0.6, 1.6))
    for _ in range(20):
        augmented = augment_clip(clip, config, rng)
        frames = [augmented.events[e] for e in augmented.ordered_events]
        assert frames == sorted(frames)
