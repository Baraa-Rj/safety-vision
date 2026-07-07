import pytest

from config.settings import ComplianceConfig
from src.ppe_compliance_tracker import PPEComplianceTracker


def _feed(tracker, track_id, item, observations, ts_start=1000.0):
    """Feed a sequence of confidence values for a single item, return transitions."""
    transitions = []
    for i, conf in enumerate(observations):
        result = tracker.update(track_id, {item: conf}, timestamp=ts_start + i)
        transitions.append(result[item])
    return transitions


def test_no_alert_below_window_size():
    cfg = ComplianceConfig(window_size=10, missing_to_alert=10, present_to_clear=3)
    tracker = PPEComplianceTracker(cfg, {"helmet"})
    transitions = _feed(tracker, 1, "helmet", [None] * 9)
    assert transitions == [None] * 9
    assert tracker.get_state(1)["helmet"] is False


def test_alert_after_threshold_missing():
    cfg = ComplianceConfig(window_size=15, missing_to_alert=10, present_to_clear=3)
    tracker = PPEComplianceTracker(cfg, {"vest"})
    transitions = _feed(tracker, 1, "vest", [None] * 15)
    alerts = [t for t in transitions if t == "alert"]
    assert len(alerts) == 1
    assert transitions[-1] == "alert"  # fires exactly when window first fills
    assert tracker.get_state(1)["vest"] is True


def test_no_alert_when_intermittent():
    cfg = ComplianceConfig(window_size=15, missing_to_alert=10, present_to_clear=3)
    tracker = PPEComplianceTracker(cfg, {"helmet"})
    # 5 missing + 10 present in any order → missing count < 10, no alert.
    sequence = [None] * 5 + [0.9] * 10
    transitions = _feed(tracker, 1, "helmet", sequence)
    assert all(t is None for t in transitions)
    assert tracker.get_state(1)["helmet"] is False


def test_hysteresis_clear_requires_consistent_present():
    cfg = ComplianceConfig(window_size=15, missing_to_alert=10, present_to_clear=3)
    tracker = PPEComplianceTracker(cfg, {"helmet"})
    # Drive into alert state.
    _feed(tracker, 1, "helmet", [None] * 15)
    assert tracker.get_state(1)["helmet"] is True

    # A single present frame is not enough — window still has 14 missings.
    r1 = tracker.update(1, {"helmet": 0.9}, timestamp=2000.0)
    assert r1["helmet"] is None
    assert tracker.get_state(1)["helmet"] is True

    r2 = tracker.update(1, {"helmet": 0.9}, timestamp=2001.0)
    assert r2["helmet"] is None
    assert tracker.get_state(1)["helmet"] is True

    # Third present pushes the window past present_to_clear → clear.
    r3 = tracker.update(1, {"helmet": 0.9}, timestamp=2002.0)
    assert r3["helmet"] == "clear"
    assert tracker.get_state(1)["helmet"] is False


def test_uncertain_band_neither_counts():
    cfg = ComplianceConfig(
        window_size=15, missing_to_alert=1, present_to_clear=1,
        uncertain_lower=0.25, uncertain_upper=0.40,
    )
    tracker = PPEComplianceTracker(cfg, {"helmet"})
    # Confidence sitting squarely in the uncertain band — neither alert nor clear
    # should ever fire even with extremely low thresholds.
    transitions = _feed(tracker, 1, "helmet", [0.30] * 20)
    assert all(t is None for t in transitions)
    assert tracker.get_state(1)["helmet"] is False


def test_per_track_isolation():
    cfg = ComplianceConfig(window_size=15, missing_to_alert=10, present_to_clear=3)
    tracker = PPEComplianceTracker(cfg, {"helmet"})
    last_a = last_b = None
    for i in range(15):
        ra = tracker.update("a", {"helmet": None}, timestamp=1000.0 + i)
        rb = tracker.update("b", {"helmet": 0.9}, timestamp=1000.0 + i)
        last_a, last_b = ra["helmet"], rb["helmet"]
    assert last_a == "alert"
    assert last_b is None
    assert tracker.get_state("a")["helmet"] is True
    assert tracker.get_state("b")["helmet"] is False


def test_cleanup_stale_removes_old_tracks():
    cfg = ComplianceConfig(track_timeout_seconds=5.0)
    tracker = PPEComplianceTracker(cfg, {"helmet"})
    tracker.update(1, {"helmet": 0.9}, timestamp=100.0)
    tracker.update(2, {"helmet": 0.9}, timestamp=110.0)
    assert tracker.tracked_count == 2

    removed = tracker.cleanup_stale(current_ts=110.0)
    # Track 1 last seen at 100 → 10s stale > 5s → dropped.
    # Track 2 last seen at 110 → 0s → kept.
    assert removed == 1
    assert tracker.tracked_count == 1


def test_alert_fires_only_once_per_transition():
    cfg = ComplianceConfig(window_size=15, missing_to_alert=10, present_to_clear=3)
    tracker = PPEComplianceTracker(cfg, {"helmet"})
    transitions = _feed(tracker, 1, "helmet", [None] * 50)
    alerts = [t for t in transitions if t == "alert"]
    assert len(alerts) == 1


def test_update_returns_action_for_each_required_item():
    cfg = ComplianceConfig(window_size=5, missing_to_alert=3, present_to_clear=2)
    tracker = PPEComplianceTracker(cfg, {"helmet", "vest"})
    result = tracker.update(1, {"helmet": 0.9, "vest": None}, timestamp=1000.0)
    assert set(result.keys()) == {"helmet", "vest"}


# --- get_display_state: three-valued on-screen verdict ---
# True = missing (red), False = present (green), None = insufficient evidence
# (yellow "checking"). A verdict during warm-up needs display_min_evidence
# agreeing frames and a strict majority.

def _display_cfg():
    return ComplianceConfig(window_size=8, missing_to_alert=5,
                            present_to_clear=2, display_min_evidence=3)


def test_display_state_unknown_track_is_pending():
    # No evidence at all → no verdict (never claim OK, never claim VIOLATION).
    tracker = PPEComplianceTracker(_display_cfg(), {"helmet"})
    assert tracker.get_display_state(99)["helmet"] is None
    assert tracker.get_state(99)["helmet"] is False


def test_display_state_pending_below_min_evidence():
    # 1-2 frames — whatever they say — is not enough for a verdict. This is
    # what keeps a blurry entry frame or a phantom person box from flashing red.
    tracker = PPEComplianceTracker(_display_cfg(), {"helmet"})
    _feed(tracker, 1, "helmet", [None, None])
    assert tracker.get_display_state(1)["helmet"] is None
    _feed(tracker, 2, "helmet", [0.9, 0.9])
    assert tracker.get_display_state(2)["helmet"] is None


def test_display_state_missing_after_min_evidence():
    # A no-PPE worker turns red after display_min_evidence frames — long
    # before the window fills (get_state stays False until it does).
    tracker = PPEComplianceTracker(_display_cfg(), {"helmet"})
    _feed(tracker, 1, "helmet", [None] * 3)
    assert tracker.get_display_state(1)["helmet"] is True
    assert tracker.get_state(1)["helmet"] is False


def test_display_state_present_after_min_evidence():
    tracker = PPEComplianceTracker(_display_cfg(), {"helmet"})
    _feed(tracker, 1, "helmet", [0.9] * 3)
    assert tracker.get_display_state(1)["helmet"] is False


def test_display_state_mixed_warmup_stays_pending():
    # Conflicting evidence with no strict majority winner → still no verdict.
    tracker = PPEComplianceTracker(_display_cfg(), {"helmet"})
    _feed(tracker, 1, "helmet", [None, 0.9, None, 0.9])
    assert tracker.get_display_state(1)["helmet"] is None


def test_display_state_defers_to_hysteresis_once_window_full():
    tracker = PPEComplianceTracker(_display_cfg(), {"helmet"})
    # Full window, missing=4 < missing_to_alert=5 → never alerted → green.
    _feed(tracker, 1, "helmet", [None, 0.9] * 4)
    assert tracker.get_display_state(1)["helmet"] is False
    assert tracker.get_state(1)["helmet"] is False
    # All-missing window → alerted → red via the same call.
    _feed(tracker, 2, "helmet", [None] * 8)
    assert tracker.get_display_state(2)["helmet"] is True
