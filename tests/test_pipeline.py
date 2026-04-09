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
    def detect(self, frame):
        return []


class StubWorkerIdentifier:
    def identify(self, crop):
        return None


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
    )

    events = pipeline.process_frame(dummy_frame)

    assert len(events["ppe_violations"]) == 1
    assert events["ppe_violations"][0]["missing"] == ["vest"]
