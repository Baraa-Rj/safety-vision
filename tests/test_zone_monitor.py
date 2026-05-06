import pytest
from src.zone_monitor import ZoneMonitor


@pytest.fixture
def monitor():
    zm = ZoneMonitor()
    zm.add_zone("zone_a", [[100, 100], [300, 100], [300, 300], [100, 300]])
    zm.add_zone("zone_b", [[400, 400], [600, 400], [600, 600], [400, 600]])
    return zm


def test_person_inside_zone_a(monitor):
    assert monitor.check_person([150, 100, 250, 250]) == "zone_a"


def test_person_inside_zone_b(monitor):
    assert monitor.check_person([450, 400, 550, 550]) == "zone_b"


def test_person_outside_all_zones(monitor):
    assert monitor.check_person([0, 0, 100, 50]) is None


def test_person_on_boundary(monitor):
    assert monitor.check_person([50, 100, 150, 200]) == "zone_a"


def test_no_zones_defined():
    empty_monitor = ZoneMonitor()
    assert empty_monitor.check_person([150, 100, 250, 250]) is None


def test_zone_breach_in_pipeline(dummy_frame):
    """Zone breach is detected and included in pipeline events."""
    from config.settings import PipelineConfig
    from src.pipeline import SafetyPipeline
    from src.renderer import FrameRenderer
    from src.event_logger import EventLogger

    class PPEWithPerson:
        def detect(self, frame):
            return [{
                "person_bbox": [150, 100, 250, 250],
                "track_id": 1,
                "compliant": True,
                "detected_ppe": ["helmet", "vest"],
                "missing_ppe": [],
            }]

    class StubWorkerIdentifier:
        def identify(self, crop, track_id=None):
            return None

        def clear_stale(self, active_track_ids):
            pass

    class StubFallDetector:
        def detect(self, person_bboxes, frame):
            return []

    class StubCamera:
        def read(self):
            return dummy_frame

        def stop(self):
            pass

    config = PipelineConfig()
    zone_monitor = ZoneMonitor()
    zone_monitor.add_zone("danger_zone", [[100, 100], [300, 100], [300, 300], [100, 300]])

    pipeline = SafetyPipeline(
        config=config,
        camera=StubCamera(),
        ppe_detector=PPEWithPerson(),
        fall_detector=StubFallDetector(),
        worker_identifier=StubWorkerIdentifier(),
        zone_monitor=zone_monitor,
        renderer=FrameRenderer(zone_monitor, config.display),
        event_logger=EventLogger(),
        alert_client=None,
    )

    events = pipeline.process_frame(dummy_frame)

    assert len(events["zone_breaches"]) == 1
    assert events["zone_breaches"][0]["zone_id"] == "danger_zone"
