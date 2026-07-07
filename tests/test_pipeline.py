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

    def detect(self, frame):
        return [{
            "person_bbox": list(self.bbox),
            "track_id": self.track_id,
            "compliant": not self.missing,
            "detected_ppe": list(self.detected),
            "missing_ppe": list(self.missing),
        }]


def _pipeline_with(detector, config):
    zone_monitor = ZoneMonitor()
    return SafetyPipeline(
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
    assert events["compliant_workers"] == []
    assert len(events["pending_workers"]) == 1


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
