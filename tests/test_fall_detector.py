"""Tests for the classifier-based FallDetector gate logic.

The detector localizes nothing itself: it takes the PPE detector's person boxes,
crops each, and runs a fallen/standing classifier. A fall is reported only after
the 'fallen' class is sustained for `consecutive_frames` on one track; the
per-track cooldown then gates the `alert` flag. These tests exercise that gate
with a mocked classifier, so no model file is required.
"""
from unittest.mock import patch

import numpy as np
import pytest

import src.fall_detector as fd
from src.fall_detector import FallDetector
from config.settings import FallDetectionConfig


class _Probs:
    def __init__(self, fallen_prob):
        self.data = [fallen_prob, 1.0 - fallen_prob]
        self.top1 = 0 if self.data[0] >= self.data[1] else 1


class _Pred:
    def __init__(self, fallen_prob):
        self.probs = _Probs(fallen_prob)


class _FakeModel:
    """Stand-in for ultralytics.YOLO classify model."""
    task = "classify"
    names = {0: "fallen", 1: "standing"}

    def __init__(self, *args, **kwargs):
        self.fallen_prob = 0.0  # set per-frame by the test

    def predict(self, crops, **kwargs):
        return [_Pred(self.fallen_prob) for _ in crops]


def _make(**overrides):
    cfg = FallDetectionConfig(**overrides)
    with patch.object(fd, "YOLO", _FakeModel):
        return FallDetector("ignored.pt", cfg)


FRAME = np.zeros((200, 200, 3), dtype=np.uint8)


def _feed(det, prob, bbox=(0, 0, 100, 150), track_id=1):
    det.model.fallen_prob = prob
    return det.detect([{"person_bbox": bbox, "track_id": track_id}], FRAME)


def test_rejects_non_classify_model():
    class _DetModel(_FakeModel):
        task = "detect"
    with patch.object(fd, "YOLO", _DetModel):
        with pytest.raises(ValueError):
            FallDetector("ignored.pt", FallDetectionConfig())


def test_sustained_fallen_fires_after_consecutive_frames():
    d = _make(consecutive_frames=3, fallen_conf=0.6, cooldown_seconds=100)
    assert _feed(d, 0.9) == []          # streak 1
    assert _feed(d, 0.9) == []          # streak 2
    falls = _feed(d, 0.9)               # streak 3 -> fire
    assert len(falls) == 1
    assert falls[0]["alert"] is True
    assert falls[0]["track_id"] == 1
    assert falls[0]["confidence"] == pytest.approx(0.9)


def test_cooldown_blocks_repeat_alert_then_reports_without_alert():
    d = _make(consecutive_frames=2, fallen_conf=0.6, cooldown_seconds=100)
    _feed(d, 0.9)
    first = _feed(d, 0.9)
    assert first[0]["alert"] is True
    again = _feed(d, 0.9)               # still fallen, but within cooldown
    assert len(again) == 1             # still reported (box stays up)
    assert again[0]["alert"] is False  # but no second backend alert


def test_one_standing_frame_resets_the_streak():
    d = _make(consecutive_frames=3, fallen_conf=0.6, cooldown_seconds=100)
    _feed(d, 0.9)
    _feed(d, 0.9)
    assert _feed(d, 0.2) == []          # standing -> streak reset
    assert _feed(d, 0.9) == []          # rebuild: streak 1
    assert _feed(d, 0.9) == []          # streak 2
    assert len(_feed(d, 0.9)) == 1      # streak 3 -> fire again


def test_below_confidence_is_not_counted_as_fallen():
    d = _make(consecutive_frames=2, fallen_conf=0.6, cooldown_seconds=100)
    # top1 is 'fallen' but prob under threshold -> must not count
    assert _feed(d, 0.55) == []
    assert _feed(d, 0.55) == []


def test_box_below_min_size_is_skipped():
    d = _make(consecutive_frames=1, fallen_conf=0.6, min_size=40)
    falls = _feed(d, 0.99, bbox=(0, 0, 30, 30))  # 30px < min_size
    assert falls == []


def test_untracked_person_still_gates_via_location_key():
    d = _make(consecutive_frames=2, fallen_conf=0.6, cooldown_seconds=100)
    assert _feed(d, 0.9, track_id=None) == []
    falls = _feed(d, 0.9, track_id=None)
    assert len(falls) == 1
    assert falls[0]["track_id"] is None


def test_cooldown_expires_and_alert_fires_again(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(fd.time, "time", lambda: clock[0])
    d = _make(consecutive_frames=1, fallen_conf=0.6, cooldown_seconds=60)

    assert _feed(d, 0.9)[0]["alert"] is True       # t=1000, alert
    clock[0] = 1030.0
    assert _feed(d, 0.9)[0]["alert"] is False      # within cooldown
    clock[0] = 1070.0
    assert _feed(d, 0.9)[0]["alert"] is True       # cooldown expired
