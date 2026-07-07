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


def test_permitted_worker_allowed():
    zm = ZoneMonitor()
    zm.add_zone("zone_a", [[100, 100], [300, 100], [300, 300], [100, 300]], allowed_workers=["W001"])
    assert zm.is_permitted("zone_a", "W001") is True


def test_unpermitted_worker_denied():
    zm = ZoneMonitor()
    zm.add_zone("zone_a", [[100, 100], [300, 100], [300, 300], [100, 300]], allowed_workers=["W001"])
    assert zm.is_permitted("zone_a", "W999") is False


def test_unidentified_worker_denied():
    zm = ZoneMonitor()
    zm.add_zone("zone_a", [[100, 100], [300, 100], [300, 300], [100, 300]], allowed_workers=["W001"])
    assert zm.is_permitted("zone_a", None) is False


def test_no_allowed_workers_denies_all():
    zm = ZoneMonitor()
    zm.add_zone("zone_a", [[100, 100], [300, 100], [300, 300], [100, 300]])
    assert zm.is_permitted("zone_a", "W001") is False


# --- machine proximity (radius) ---

def _machine_monitor(radius):
    zm = ZoneMonitor()
    # 200x200 square footprint at [100..300, 100..300]
    zm.add_zone("machine", [[100, 100], [300, 100], [300, 300], [100, 300]],
                radius=radius)
    return zm


# bbox [340, 0, 360, 200] sits to the right of the machine; its nearest
# lower-body point (left edge x=340) is 40 px from the footprint's right edge.

def test_radius_zero_is_plain_containment():
    zm = _machine_monitor(radius=0)
    assert zm.check_person([340, 0, 360, 200]) is None


def test_proximity_triggers_within_radius():
    zm = _machine_monitor(radius=60)   # 40 px gap < 60 px keep-out -> breach
    assert zm.check_person([340, 0, 360, 200]) == "machine"


def test_proximity_clear_beyond_radius():
    zm = _machine_monitor(radius=30)   # 40 px gap > 30 px keep-out -> clear
    assert zm.check_person([340, 0, 360, 200]) is None


def test_proximity_still_triggers_inside():
    zm = _machine_monitor(radius=60)
    assert zm.check_person([190, 0, 210, 200]) == "machine"


def test_lower_band_detects_when_feet_occluded():
    """Feet truncated/occluded: bbox bottom lands at the knees, above the zone.
    The single-foot test would miss it; the lower band still catches it."""
    zm = ZoneMonitor()
    zm.add_zone("area", [[100, 300], [300, 300], [300, 400], [100, 400]])
    # Person whose visible box bottom (y2=360) is inside the zone band even
    # though it's not their true feet.
    assert zm.check_person([150, 100, 250, 360]) == "area"


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
        def detect(self, fallen_detections):
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


def test_permitted_worker_no_zone_breach(dummy_frame):
    """Permitted worker in a zone does NOT produce a zone breach."""
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
            return {"qr_data": "W042", "worker_id": "W042", "worker_name": None}

        def clear_stale(self, active_track_ids):
            pass

    class StubCamera:
        def read(self):
            return dummy_frame

        def stop(self):
            pass

    config = PipelineConfig()
    zone_monitor = ZoneMonitor()
    zone_monitor.add_zone("danger_zone", [[100, 100], [300, 100], [300, 300], [100, 300]],
                          allowed_workers=["W042"])

    pipeline = SafetyPipeline(
        config=config,
        camera=StubCamera(),
        ppe_detector=PPEWithPerson(),
        fall_detector=None,
        worker_identifier=StubWorkerIdentifier(),
        zone_monitor=zone_monitor,
        renderer=FrameRenderer(zone_monitor, config.display),
        event_logger=EventLogger(),
        alert_client=None,
    )

    events = pipeline.process_frame(dummy_frame)

    assert len(events["zone_breaches"]) == 0


def test_zone_rescaled_to_runtime_frame():
    # Defined on the 704x576 substream, run on the 2880x1620 main stream:
    # the polygon must scale up ~4x so the same physical spot still triggers.
    zm = ZoneMonitor()
    zm.add_zone("z", [[100, 100], [300, 100], [300, 300], [100, 300]],
                frame_size=(704, 576))
    zm.set_frame_size(2880, 1620)
    # Center of the original zone, scaled to main-stream coords.
    sx, sy = 2880 / 704, 1620 / 576
    cx, cy = int(200 * sx), int(200 * sy)
    assert zm.check_person([cx - 50, cy - 200, cx + 50, cy]) == "z"
    # The unscaled center must no longer trigger (it now sits outside).
    assert zm.check_person([150, 50, 250, 200]) is None


def test_zone_without_frame_size_untouched():
    zm = ZoneMonitor()
    zm.add_zone("z", [[100, 100], [300, 100], [300, 300], [100, 300]])
    zm.set_frame_size(2880, 1620)
    assert zm.check_person([150, 100, 250, 250]) == "z"


def test_zone_scaling_idempotent_and_matching_size_noop():
    zm = ZoneMonitor()
    zm.add_zone("z", [[100, 100], [300, 100], [300, 300], [100, 300]],
                frame_size=(704, 576))
    zm.set_frame_size(704, 576)      # same size as defined -> unchanged
    assert zm.check_person([150, 100, 250, 250]) == "z"
    zm.set_frame_size(704, 576)      # repeated call -> still fine
    assert zm.check_person([150, 100, 250, 250]) == "z"


def test_zone_radius_scales_with_frame():
    zm = ZoneMonitor()
    zm.add_zone("z", [[100, 100], [200, 100], [200, 200], [100, 200]],
                radius=10, frame_size=(100, 100))
    zm.set_frame_size(200, 200)      # 2x -> polygon [200..400], radius 20
    assert zm.radii["z"] == 20.0
    # Feet 15px outside the scaled edge: inside the scaled 20px keep-out.
    assert zm.check_person([415, 250, 455, 400]) == "z"
    # 30px outside: beyond the keep-out.
    assert zm.check_person([430, 250, 470, 400]) is None
