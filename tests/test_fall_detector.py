"""Tests for the detection-based FallDetector gate.

best.pt emits a 'fallen' class directly; this detector is the temporal gate over
those detections. A fall is confirmed only after 'fallen' is sustained for
`consecutive_frames` on one track; brief dropouts are tolerated, long gaps reset
the streak, and the per-track cooldown limits backend alerts to one per event.
A controllable clock (the `clock` fixture) drives the time-based logic.
"""
import pytest

import src.fall_detector as fd
from src.fall_detector import FallDetector
from config.settings import FallDetectionConfig


@pytest.fixture
def clock(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(fd.time, "time", lambda: t[0])
    return t


def _make(**overrides):
    return FallDetector(FallDetectionConfig(**overrides))


def _det(bbox=(0, 0, 100, 150), track_id=1, conf=0.9):
    return {"bbox": list(bbox), "track_id": track_id, "confidence": conf}


def _feed(det, clock, dt=0.1, **det_kwargs):
    clock[0] += dt
    return det.detect([_det(**det_kwargs)])


def test_sustained_fallen_fires_after_consecutive_frames(clock):
    d = _make(consecutive_frames=3, fallen_conf=0.4, cooldown_seconds=100)
    assert _feed(d, clock) == []        # streak 1
    assert _feed(d, clock) == []        # streak 2
    falls = _feed(d, clock)             # streak 3 -> fire
    assert len(falls) == 1
    assert falls[0]["alert"] is True
    assert falls[0]["track_id"] == 1
    assert falls[0]["confidence"] == pytest.approx(0.9)


def test_cooldown_blocks_repeat_alert_but_keeps_reporting(clock):
    d = _make(consecutive_frames=2, fallen_conf=0.4, cooldown_seconds=100)
    _feed(d, clock)
    assert _feed(d, clock)[0]["alert"] is True
    again = _feed(d, clock)             # still fallen, within cooldown
    assert len(again) == 1             # box stays up
    assert again[0]["alert"] is False  # no second backend alert


def test_below_confidence_is_not_counted(clock):
    d = _make(consecutive_frames=2, fallen_conf=0.6)
    assert _feed(d, clock, conf=0.5) == []
    assert _feed(d, clock, conf=0.5) == []


def test_box_below_min_size_is_skipped(clock):
    d = _make(consecutive_frames=1, fallen_conf=0.4, min_size=40)
    assert _feed(d, clock, bbox=(0, 0, 30, 30)) == []


def test_untracked_detection_gates_via_location_key(clock):
    d = _make(consecutive_frames=2, fallen_conf=0.4, cooldown_seconds=100)
    assert _feed(d, clock, track_id=None) == []
    falls = _feed(d, clock, track_id=None)
    assert len(falls) == 1
    assert falls[0]["track_id"] is None


def test_long_gap_resets_the_streak(clock):
    d = _make(consecutive_frames=3, fallen_conf=0.4, cooldown_seconds=100)
    _feed(d, clock)                     # streak 1
    _feed(d, clock)                     # streak 2
    assert _feed(d, clock, dt=2.0) == []  # >1.5s gap -> streak restarts at 1
    assert _feed(d, clock) == []        # streak 2
    assert len(_feed(d, clock)) == 1    # streak 3 -> fire


def test_brief_dropout_does_not_reset_streak(clock):
    d = _make(consecutive_frames=3, fallen_conf=0.4, cooldown_seconds=100)
    _feed(d, clock)                     # streak 1
    _feed(d, clock)                     # streak 2
    # a dropped frame (1.0s gap, still < 1.5) must not reset
    assert len(_feed(d, clock, dt=1.0)) == 1  # streak 3 -> fire


def test_cooldown_expires_and_alert_fires_again(clock):
    d = _make(consecutive_frames=1, fallen_conf=0.4, cooldown_seconds=60)
    assert _feed(d, clock)[0]["alert"] is True     # t~1000, alert
    clock[0] += 30
    assert d.detect([_det()])[0]["alert"] is False  # within cooldown
    clock[0] += 40
    assert d.detect([_det()])[0]["alert"] is True   # cooldown expired
