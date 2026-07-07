import numpy as np
import pytest
from config.settings import PipelineConfig
from src.pipeline import SafetyPipeline
from src.zone_monitor import ZoneMonitor
from src.renderer import FrameRenderer
from src.event_logger import EventLogger


class StubCamera:
    def read(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def stop(self):
        pass


class StubPPEDetector:
    def detect(self, frame):
        return []


class StubFallDetector:
    def detect(self, fallen_detections):
        return []


class StubWorkerIdentifier:
    def identify(self, crop, track_id=None):
        return None

    def get_cached(self, track_id):
        return None

    def transfer(self, old_track_id, new_track_id):
        return None

    def clear_stale(self, active_track_ids):
        pass


class StubAlertClient:
    def send_ppe_alert(self, violation, frame):
        pass

    def shutdown(self):
        pass


@pytest.fixture
def pipeline():
    config = PipelineConfig()
    zone_monitor = ZoneMonitor()
    return SafetyPipeline(
        config=config,
        camera=StubCamera(),
        ppe_detector=StubPPEDetector(),
        fall_detector=StubFallDetector(),
        worker_identifier=StubWorkerIdentifier(),
        zone_monitor=zone_monitor,
        renderer=FrameRenderer(zone_monitor, config.display),
        event_logger=EventLogger(),
        alert_client=StubAlertClient(),
    )


def test_process_frame_with_no_detections(pipeline, dummy_frame):
    events = pipeline.process_frame(dummy_frame)

    assert events["ppe_violations"] == []
    assert events["compliant_workers"] == []
    assert events["falls"] == []
    assert events["zone_breaches"] == []
    assert "timestamp" in events


def test_process_frame_with_ppe_violation(dummy_frame):
    class PPEWithViolation:
        def detect(self, frame):
            return [{
                "person_bbox": [10, 10, 100, 200],
                "track_id": 1,
                "compliant": False,
                "detected_ppe": ["helmet"],
                "missing_ppe": ["vest"],
            }]

    config = PipelineConfig()
    zone_monitor = ZoneMonitor()
    pipeline = SafetyPipeline(
        config=config,
        camera=StubCamera(),
        ppe_detector=PPEWithViolation(),
        fall_detector=StubFallDetector(),
        worker_identifier=StubWorkerIdentifier(),
        zone_monitor=zone_monitor,
        renderer=FrameRenderer(zone_monitor, config.display),
        event_logger=EventLogger(),
        alert_client=StubAlertClient(),
    )

    # A new track missing PPE: no verdict (yellow "checking") until
    # display_min_evidence frames agree — never a false green, never a red
    # flash off one frame. Red follows as soon as the evidence bar is met.
    for _ in range(config.compliance.display_min_evidence - 1):
        events = pipeline.process_frame(dummy_frame)
        assert events["ppe_violations"] == []
        assert events["compliant_workers"] == []
        assert len(events["pending_workers"]) == 1

    events = pipeline.process_frame(dummy_frame)
    assert len(events["ppe_violations"]) == 1
    assert events["ppe_violations"][0]["missing"] == ["vest"]
    assert events["ppe_violations"][0]["worker_id"] is None
    assert events["ppe_violations"][0]["worker_name"] is None
    assert events["ppe_violations"][0]["qr_data"] is None

    # And it stays red once the window fills (hysteresis takes over).
    for _ in range(config.compliance.window_size):
        events = pipeline.process_frame(dummy_frame)
    assert len(events["ppe_violations"]) == 1


def test_new_track_without_any_ppe_red_after_min_evidence(dummy_frame):
    """First screenshot bug: a worker entering the FOV with no helmet AND no
    vest was drawn green 'OK' until the smoothing window filled. He must never
    be claimed compliant — 'checking' during warm-up, then red with both
    items once display_min_evidence frames agree."""
    class PPENoGear:
        def detect(self, frame):
            return [{
                "person_bbox": [10, 10, 100, 200],
                "track_id": 3,
                "compliant": False,
                "detected_ppe": [],
                "missing_ppe": ["helmet", "vest"],
            }]

    config = PipelineConfig()
    zone_monitor = ZoneMonitor()
    pipeline = SafetyPipeline(
        config=config,
        camera=StubCamera(),
        ppe_detector=PPENoGear(),
        fall_detector=StubFallDetector(),
        worker_identifier=StubWorkerIdentifier(),
        zone_monitor=zone_monitor,
        renderer=FrameRenderer(zone_monitor, config.display),
        event_logger=EventLogger(),
        alert_client=StubAlertClient(),
    )

    for _ in range(config.compliance.display_min_evidence):
        events = pipeline.process_frame(dummy_frame)
        assert events["compliant_workers"] == []   # never claimed OK

    assert len(events["ppe_violations"]) == 1
    assert events["ppe_violations"][0]["missing"] == ["helmet", "vest"]


def test_transient_track_never_shown_red(dummy_frame):
    """Second screenshot bug: a phantom person box (or one blurry entry frame)
    that lives fewer than display_min_evidence frames must never be drawn as a
    VIOLATION — it stays in the yellow 'checking' state."""
    class PPEPhantom:
        def detect(self, frame):
            return [{
                "person_bbox": [10, 10, 40, 200],
                "track_id": 42,
                "compliant": False,
                "detected_ppe": [],
                "missing_ppe": ["helmet", "vest"],
            }]

    config = PipelineConfig()
    zone_monitor = ZoneMonitor()
    pipeline = SafetyPipeline(
        config=config,
        camera=StubCamera(),
        ppe_detector=PPEPhantom(),
        fall_detector=StubFallDetector(),
        worker_identifier=StubWorkerIdentifier(),
        zone_monitor=zone_monitor,
        renderer=FrameRenderer(zone_monitor, config.display),
        event_logger=EventLogger(),
        alert_client=StubAlertClient(),
    )

    for _ in range(config.compliance.display_min_evidence - 1):
        events = pipeline.process_frame(dummy_frame)
        assert events["ppe_violations"] == []
        assert events["compliant_workers"] == []
        assert len(events["pending_workers"]) == 1


class ConfigurablePPE:
    """Stub detector whose track id and detections can be changed mid-test —
    simulates BoT-SORT id churn and intermittent PPE detection on a moving
    worker."""

    def __init__(self, track_id=7, bbox=None):
        self.track_id = track_id
        self.bbox = bbox or [10, 10, 100, 200]
        self.detected = ["helmet", "vest"]
        self.missing = []
        self.ppe_confidences = None    # optionally simulate low-conf sightings

    def detect(self, frame):
        result = {
            "person_bbox": list(self.bbox),
            "track_id": self.track_id,
            "compliant": not self.missing,
            "detected_ppe": list(self.detected),
            "missing_ppe": list(self.missing),
        }
        if self.ppe_confidences is not None:
            result["ppe_confidences"] = dict(self.ppe_confidences)
        return [result]


def _pipeline_with(detector, config, identifier=None, wet=None):
    zone_monitor = ZoneMonitor()
    return SafetyPipeline(
        config=config,
        camera=StubCamera(),
        ppe_detector=detector,
        fall_detector=StubFallDetector(),
        worker_identifier=identifier or StubWorkerIdentifier(),
        zone_monitor=zone_monitor,
        renderer=FrameRenderer(zone_monitor, config.display),
        event_logger=EventLogger(),
        alert_client=StubAlertClient(),
        wet_floor_detector=wet,
    )


def test_compliant_worker_entry_never_shows_red(dummy_frame):
    """The reported flow: worker appears, vest is only glimpsed at low
    confidence for the first frames (half-visible, motion blur), then detects
    cleanly. Must go checking -> green with NO red in between — low-conf
    sightings are uncertain evidence, not 'missing' votes."""
    config = PipelineConfig()
    detector = ConfigurablePPE(track_id=7)
    detector.detected = ["helmet"]
    detector.missing = ["vest"]                              # raw: below threshold
    detector.ppe_confidences = {"helmet": 0.9, "vest": 0.30}  # but glimpsed
    pipeline = _pipeline_with(detector, config)

    reds = []
    for _ in range(3):
        events = pipeline.process_frame(dummy_frame)
        reds += events["ppe_violations"]
    assert reds == []                       # entry frames: no false red

    # Vest now detected properly -> green after the short validation period.
    detector.detected = ["helmet", "vest"]
    detector.missing = []
    detector.ppe_confidences = {"helmet": 0.9, "vest": 0.85}
    for _ in range(config.compliance.display_min_evidence):
        events = pipeline.process_frame(dummy_frame)
        reds += events["ppe_violations"]
    assert reds == []
    assert len(events["compliant_workers"]) == 1


def test_track_id_churn_keeps_compliant_worker_green(dummy_frame):
    """A confirmed-compliant worker whose track id churns mid-walk (and whose
    PPE detection drops out from motion blur) must stay green — no Checking
    relapse, no red flash. The new id inherits the displayed verdict from the
    overlapping dead track."""
    config = PipelineConfig()
    detector = ConfigurablePPE(track_id=7)
    pipeline = _pipeline_with(detector, config)

    # Earn the compliant verdict on the original id.
    for _ in range(config.compliance.window_size):
        events = pipeline.process_frame(dummy_frame)
    assert len(events["compliant_workers"]) == 1

    # Churn: new id, same place, and the vest/helmet detections drop out
    # (blur) for just under a full window.
    detector.track_id = 8
    detector.detected = []
    detector.missing = ["helmet", "vest"]
    for _ in range(config.compliance.window_size - 1):
        events = pipeline.process_frame(dummy_frame)
        assert events["ppe_violations"] == []
        assert events["pending_workers"] == []
        assert len(events["compliant_workers"]) == 1


def test_genuine_violation_after_churn_still_turns_red(dummy_frame):
    """The inherited green verdict must not mask a real violation: if the
    missing evidence persists to a full window, hysteresis flags it red."""
    config = PipelineConfig()
    detector = ConfigurablePPE(track_id=7)
    pipeline = _pipeline_with(detector, config)

    for _ in range(config.compliance.window_size):
        pipeline.process_frame(dummy_frame)

    detector.track_id = 8
    detector.detected = ["helmet"]
    detector.missing = ["vest"]
    for _ in range(config.compliance.window_size):
        events = pipeline.process_frame(dummy_frame)

    assert len(events["ppe_violations"]) == 1
    assert events["ppe_violations"][0]["missing"] == ["vest"]


def test_inheritance_requires_overlap(dummy_frame):
    """A new track far from any recently-dead one inherits nothing — it goes
    through Checking like any unknown worker (no verdict leakage across the
    room)."""
    config = PipelineConfig()
    detector = ConfigurablePPE(track_id=7, bbox=[10, 10, 100, 200])
    pipeline = _pipeline_with(detector, config)

    for _ in range(config.compliance.window_size):
        pipeline.process_frame(dummy_frame)

    # New id on the other side of the frame: no IoU with the dead track.
    detector.track_id = 9
    detector.bbox = [400, 10, 490, 200]
    detector.detected = []
    detector.missing = ["helmet", "vest"]
    events = pipeline.process_frame(dummy_frame)
    # The new track earned nothing — it's Checking. (Dead track 7's ghost box
    # legitimately holds at ITS old spot; only non-held entries matter here.)
    live = [w for w in events["compliant_workers"] if not w.get("held")]
    assert live == []
    assert len(events["pending_workers"]) == 1


def test_identity_survives_track_churn_and_occlusion(dummy_frame):
    """Once scanned, a worker's QR identity must follow them through a
    detection dropout (same id absent for a few frames) and through tracker
    id churn — no rescan, no 'unknown worker'. Uses the real WorkerIdentifier
    so the transfer/TTL wiring is exercised end to end."""
    import time as _time
    from src.worker_id import WorkerIdentifier

    config = PipelineConfig()
    detector = ConfigurablePPE(track_id=7)
    identifier = WorkerIdentifier()
    identifier._cache[7] = {
        "result": {"qr_data": "W042", "worker_name": None, "worker_id": "W042"},
        "timestamp": _time.time(),
    }
    pipeline = _pipeline_with(detector, config, identifier=identifier)

    for _ in range(config.compliance.window_size):
        events = pipeline.process_frame(dummy_frame)
    assert events["compliant_workers"][0]["worker_id"] == "W042"

    # Occlusion: the person vanishes from detections for a few frames — the
    # identity must NOT be purged (old clear_stale deleted it here).
    vanished = detector.detect
    detector.detect = lambda frame: []
    for _ in range(3):
        pipeline.process_frame(dummy_frame)
    detector.detect = vanished
    events = pipeline.process_frame(dummy_frame)
    assert events["compliant_workers"][0]["worker_id"] == "W042"

    # Churn: re-acquired under a new id at the same spot → identity follows.
    detector.track_id = 8
    events = pipeline.process_frame(dummy_frame)
    workers = events["compliant_workers"] + events["pending_workers"]
    assert workers[0]["worker_id"] == "W042"


def test_held_box_bridges_detection_dropout(dummy_frame):
    """One missed detection must not erase the worker's rectangle: the last
    box is re-emitted (with verdict and identity) until box_hold_seconds
    passes, then disappears — a worker who left takes their box along."""
    config = PipelineConfig()
    detector = ConfigurablePPE(track_id=7)
    pipeline = _pipeline_with(detector, config)

    for _ in range(config.compliance.window_size):
        events = pipeline.process_frame(dummy_frame)
    assert len(events["compliant_workers"]) == 1

    # Detection drops out entirely — the box holds at the last position.
    detector.detect = lambda frame: []
    events = pipeline.process_frame(dummy_frame)
    held = events["compliant_workers"]
    assert len(held) == 1
    assert held[0]["held"] is True
    assert held[0]["bbox"] == [10, 10, 100, 200]
    assert held[0]["track_id"] == 7

    # Past the hold window the ghost expires.
    pipeline._track_last_seen[7]["ts"] -= config.display.box_hold_seconds + 1
    events = pipeline.process_frame(dummy_frame)
    assert events["compliant_workers"] == []
    assert events["ppe_violations"] == []


def test_held_box_keeps_violation_verdict(dummy_frame):
    """A violator's ghost box stays red with the same missing items — the
    dropout must not launder a violation into a blank frame."""
    config = PipelineConfig()
    detector = ConfigurablePPE(track_id=7)
    detector.detected = ["helmet"]
    detector.missing = ["vest"]
    pipeline = _pipeline_with(detector, config)

    for _ in range(config.compliance.window_size):
        events = pipeline.process_frame(dummy_frame)
    assert len(events["ppe_violations"]) == 1

    detector.detect = lambda frame: []
    events = pipeline.process_frame(dummy_frame)
    assert len(events["ppe_violations"]) == 1
    assert events["ppe_violations"][0]["held"] is True
    assert events["ppe_violations"][0]["missing"] == ["vest"]


def test_held_box_suppressed_by_overlapping_detection(dummy_frame):
    """If a person IS detected where the dead track was (re-acquired under a
    new id that missed the continuity IoU bar), no ghost is drawn — one
    worker, one box."""
    config = PipelineConfig()
    detector = ConfigurablePPE(track_id=7)
    pipeline = _pipeline_with(detector, config)

    for _ in range(config.compliance.window_size):
        pipeline.process_frame(dummy_frame)

    # Same spot, new id 99 — continuity transfers state to 99; and even if it
    # didn't, the overlap guard blocks a second box for dead track 7.
    detector.track_id = 99
    events = pipeline.process_frame(dummy_frame)
    boxes = (events["compliant_workers"] + events["ppe_violations"]
             + events["pending_workers"])
    assert len(boxes) == 1


def test_continuity_bridges_moving_worker_gap(dummy_frame):
    """A walking worker re-acquired PAST the overlap point (IoU = 0 with the
    dead track) still keeps their verdict: a unique dead track within one
    body height is the same worker. This is what removes the ghost-trail —
    the old entry transfers instead of lingering behind them."""
    config = PipelineConfig()
    detector = ConfigurablePPE(track_id=7, bbox=[10, 10, 100, 200])
    pipeline = _pipeline_with(detector, config)
    for _ in range(config.compliance.window_size):
        pipeline.process_frame(dummy_frame)

    # New id, 120 px ahead: boxes disjoint, centre distance < height (190).
    detector.track_id = 8
    detector.bbox = [130, 10, 220, 200]
    events = pipeline.process_frame(dummy_frame)
    assert len(events["compliant_workers"]) == 1
    assert events["compliant_workers"][0]["track_id"] == 8
    assert all(not w.get("held") for w in events["compliant_workers"])


def test_ghost_suppressed_near_current_detection(dummy_frame):
    """A ghost within ~1.2 body heights of ANY current detection is dropped —
    the trail-effect guard for workers re-acquired beyond continuity range."""
    config = PipelineConfig()
    detector = ConfigurablePPE(track_id=7, bbox=[10, 10, 100, 200])
    pipeline = _pipeline_with(detector, config)
    for _ in range(config.compliance.window_size):
        pipeline.process_frame(dummy_frame)

    # 200 px ahead: too far for continuity (dist > height) but inside ghost
    # reach (1.2 * height = 228) -> the dead track leaves no box behind.
    detector.track_id = 8
    detector.bbox = [210, 10, 300, 200]
    events = pipeline.process_frame(dummy_frame)
    held = [w for w in events["compliant_workers"] + events["ppe_violations"]
            if w.get("held")]
    assert held == []


def test_wet_floor_runs_every_nth_frame(dummy_frame):
    """The seg model shares the loop with PPE detection; it must only run on
    every Nth frame, with confirmed events re-emitted in between (no blink)."""
    from config.settings import WetFloorConfig

    class CountingWet:
        def __init__(self):
            self.calls = 0
            self.config = WetFloorConfig(enabled=True, process_every_n=3)

        def detect(self, frame):
            self.calls += 1
            return [{"bbox": [0, 0, 50, 50], "confidence": 0.9, "area_pct": 1.0}]

    config = PipelineConfig()
    wet = CountingWet()
    pipeline = _pipeline_with(ConfigurablePPE(), config, wet=wet)
    for _ in range(15):
        events = pipeline.process_frame(dummy_frame)
    assert wet.calls == 5                        # every 3rd frame
    assert len(events["wet_floor_events"]) == 1  # 5 processed frames = confirmed
    events = pipeline.process_frame(dummy_frame)  # skipped frame
    assert len(events["wet_floor_events"]) == 1  # cached event re-emitted


def test_track_without_verdict_never_ghosts(dummy_frame):
    """A track that vanished while still Checking (usually a phantom person
    box) leaves nothing behind — ghosts require an earned verdict."""
    config = PipelineConfig()
    detector = ConfigurablePPE(track_id=5)
    pipeline = _pipeline_with(detector, config)

    # Fewer frames than display_min_evidence -> no verdict earned.
    pipeline.process_frame(dummy_frame)
    detector.detect = lambda frame: []
    events = pipeline.process_frame(dummy_frame)
    assert events["compliant_workers"] == []
    assert events["ppe_violations"] == []


def test_single_frame_violation_not_shown(dummy_frame):
    """A genuinely compliant worker whose vest flickers off for a frame or two
    must NOT be drawn red — the on-screen state is smoothed per track. The
    track needs an established compliant history first; a NEW track missing
    an item is red immediately (fail-closed)."""
    class PPEFlicker:
        def __init__(self):
            self.detected = ["helmet", "vest"]
            self.missing = []

        def detect(self, frame):
            return [{
                "person_bbox": [10, 10, 100, 200],
                "track_id": 7,
                "compliant": not self.missing,
                "detected_ppe": list(self.detected),
                "missing_ppe": list(self.missing),
            }]

    detector = PPEFlicker()
    config = PipelineConfig()
    zone_monitor = ZoneMonitor()
    pipeline = SafetyPipeline(
        config=config,
        camera=StubCamera(),
        ppe_detector=detector,
        fall_detector=StubFallDetector(),
        worker_identifier=StubWorkerIdentifier(),
        zone_monitor=zone_monitor,
        renderer=FrameRenderer(zone_monitor, config.display),
        event_logger=EventLogger(),
        alert_client=StubAlertClient(),
    )

    # Establish a fully compliant history (window fills all-present).
    for _ in range(config.compliance.window_size):
        events = pipeline.process_frame(dummy_frame)
    assert events["ppe_violations"] == []

    # Vest drops out for two frames — well under missing_to_alert, stays green.
    detector.detected = ["helmet"]
    detector.missing = ["vest"]
    for _ in range(2):
        events = pipeline.process_frame(dummy_frame)
    assert events["ppe_violations"] == []
    assert len(events["compliant_workers"]) == 1


def test_process_frame_violation_with_qr(dummy_frame):
    class PPEWithViolation:
        def detect(self, frame):
            return [{
                "person_bbox": [10, 10, 100, 200],
                "track_id": 1,
                "compliant": False,
                "detected_ppe": ["helmet"],
                "missing_ppe": ["vest"],
            }]

    class QRWorkerIdentifier:
        def identify(self, crop, track_id=None):
            return {
                "qr_data": '{"id": "W042", "name": "John Doe"}',
                "worker_name": "John Doe",
                "worker_id": "W042",
            }

        def clear_stale(self, active_track_ids):
            pass

    config = PipelineConfig()
    zone_monitor = ZoneMonitor()
    pipeline = SafetyPipeline(
        config=config,
        camera=StubCamera(),
        ppe_detector=PPEWithViolation(),
        fall_detector=StubFallDetector(),
        worker_identifier=QRWorkerIdentifier(),
        zone_monitor=zone_monitor,
        renderer=FrameRenderer(zone_monitor, config.display),
        event_logger=EventLogger(),
        alert_client=StubAlertClient(),
    )

    for _ in range(config.compliance.window_size):
        events = pipeline.process_frame(dummy_frame)

    assert len(events["ppe_violations"]) == 1
    v = events["ppe_violations"][0]
    assert v["missing"] == ["vest"]
    assert v["worker_id"] == "W042"
    assert v["worker_name"] == "John Doe"
    assert v["qr_data"] == '{"id": "W042", "name": "John Doe"}'
