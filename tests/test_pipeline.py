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

    # On-screen red is now smoothed: it appears only after the violation is
    # sustained long enough for the tracker to flag it (window fills, missing
    # >= missing_to_alert). Drive enough frames to cross that threshold.
    for _ in range(config.compliance.window_size):
        events = pipeline.process_frame(dummy_frame)

    assert len(events["ppe_violations"]) == 1
    assert events["ppe_violations"][0]["missing"] == ["vest"]
    assert events["ppe_violations"][0]["worker_id"] is None
    assert events["ppe_violations"][0]["worker_name"] is None
    assert events["ppe_violations"][0]["qr_data"] is None


def test_single_frame_violation_not_shown(dummy_frame):
    """A genuinely compliant worker whose vest flickers off for a frame or two
    must NOT be drawn red — the on-screen state is smoothed per track."""
    class PPEFlicker:
        def detect(self, frame):
            return [{
                "person_bbox": [10, 10, 100, 200],
                "track_id": 7,
                "compliant": False,
                "detected_ppe": ["helmet"],
                "missing_ppe": ["vest"],
            }]

    config = PipelineConfig()
    zone_monitor = ZoneMonitor()
    pipeline = SafetyPipeline(
        config=config,
        camera=StubCamera(),
        ppe_detector=PPEFlicker(),
        fall_detector=StubFallDetector(),
        worker_identifier=StubWorkerIdentifier(),
        zone_monitor=zone_monitor,
        renderer=FrameRenderer(zone_monitor, config.display),
        event_logger=EventLogger(),
        alert_client=StubAlertClient(),
    )

    # A couple of bad frames is well under missing_to_alert -> stays green.
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
