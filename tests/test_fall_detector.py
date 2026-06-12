"""Tests for the detection-based FallDetector gate and severity scoring.

best.pt emits a 'fallen' class directly; this detector confirms a sustained
fall, rejects implausible boxes, and assigns a LOW/MEDIUM/HIGH triage severity
that escalates the longer/stiller a worker stays down. A backend alert fires on
first confirmation and on each tier escalation. A controllable clock drives the
time-based logic.
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


def _det(bbox=(0, 0, 150, 100), track_id=1, conf=0.9):
    return {"bbox": list(bbox), "track_id": track_id, "confidence": conf}


def _feed(det, clock, dt=0.1, **det_kwargs):
    clock[0] += dt
    return det.detect([_det(**det_kwargs)])


# --- gate ---

def test_sustained_fallen_fires_after_consecutive_frames(clock):
    d = _make(consecutive_frames=3, fallen_conf=0.4)
    assert _feed(d, clock) == []        # streak 1
    assert _feed(d, clock) == []        # streak 2
    falls = _feed(d, clock)             # streak 3 -> fire
    assert len(falls) == 1
    assert falls[0]["alert"] is True
    assert falls[0]["severity"] == "LOW"
    assert falls[0]["track_id"] == 1


def test_below_confidence_is_not_counted(clock):
    d = _make(consecutive_frames=2, fallen_conf=0.6)
    assert _feed(d, clock, conf=0.5) == []
    assert _feed(d, clock, conf=0.5) == []


def test_box_below_min_size_is_skipped(clock):
    d = _make(consecutive_frames=1, fallen_conf=0.4, min_size=40)
    assert _feed(d, clock, bbox=(0, 0, 30, 30)) == []


def test_tall_narrow_box_is_rejected(clock):
    d = _make(consecutive_frames=1, max_aspect_ratio=1.5)
    # h/w = 4.0 -> a fallen body is never this vertical
    assert _feed(d, clock, bbox=(0, 0, 50, 200)) == []


def test_untracked_detection_gates_via_location_key(clock):
    d = _make(consecutive_frames=2, fallen_conf=0.4)
    assert _feed(d, clock, track_id=None) == []
    falls = _feed(d, clock, track_id=None)
    assert len(falls) == 1
    assert falls[0]["track_id"] is None


def test_long_gap_resets_the_streak(clock):
    d = _make(consecutive_frames=3)
    _feed(d, clock)
    _feed(d, clock)
    assert _feed(d, clock, dt=2.0) == []  # >1.5s gap -> restart
    assert _feed(d, clock) == []
    assert len(_feed(d, clock)) == 1


# --- severity ---

def test_severity_escalates_low_medium_high(clock):
    d = _make(consecutive_frames=2, medium_seconds=3, high_still_seconds=8)
    alerts = []
    for _ in range(12):                 # feed a still body at 1s intervals
        clock[0] += 1.0
        for fr in d.detect([_det()]):
            if fr["alert"]:
                alerts.append(fr["severity"])
    assert alerts == ["LOW", "MEDIUM", "HIGH"]


def test_no_realert_within_same_tier(clock):
    d = _make(consecutive_frames=1, medium_seconds=100, high_still_seconds=100)
    first = _feed(d, clock)
    assert first[0]["severity"] == "LOW" and first[0]["alert"] is True
    again = _feed(d, clock)
    assert again[0]["severity"] == "LOW" and again[0]["alert"] is False


def test_motion_prevents_high_severity(clock):
    # A worker who keeps moving never hits HIGH (stillness resets), but still
    # reaches MEDIUM by duration.
    d = _make(consecutive_frames=1, medium_seconds=3, high_still_seconds=5,
              still_motion_px=15)
    alerts = []
    x = 0
    for _ in range(10):
        clock[0] += 1.0
        x += 40                         # move > still_motion_px each frame
        for fr in d.detect([_det(bbox=(x, 0, x + 150, 100))]):
            if fr["alert"]:
                alerts.append(fr["severity"])
    assert "HIGH" not in alerts
    assert alerts == ["LOW", "MEDIUM"]


def test_recovery_starts_a_fresh_low_event(clock):
    d = _make(consecutive_frames=1, medium_seconds=3, high_still_seconds=100)
    for _ in range(5):                  # escalate to MEDIUM
        clock[0] += 1.0
        d.detect([_det()])
    clock[0] += 4.0                     # > recovery_seconds -> recovered, event resets
    falls = d.detect([_det()])
    assert falls[0]["severity"] == "LOW"
    assert falls[0]["alert"] is True    # new event re-alerts
