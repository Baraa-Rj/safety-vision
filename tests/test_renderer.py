import numpy as np
import pytest
from src.zone_monitor import ZoneMonitor
from src.renderer import FrameRenderer, project_bbox


@pytest.fixture
def renderer():
    return FrameRenderer(ZoneMonitor())


def test_draw_returns_same_shape(renderer, dummy_frame):
    events = {
        "ppe_violations": [],
        "compliant_workers": [],
        "falls": [],
        "zone_breaches": [],
    }
    result = renderer.draw(dummy_frame, events)
    assert result.shape == dummy_frame.shape


def test_draw_with_events(renderer, dummy_frame):
    original = dummy_frame.copy()
    events = {
        "compliant_workers": [{"bbox": [10, 10, 50, 50], "worker_id": 1}],
        "pending_workers": [{"bbox": [210, 10, 250, 50], "worker_id": 3}],
        "ppe_violations": [{"bbox": [60, 60, 100, 100], "worker_id": 2, "missing": ["helmet"]}],
        "falls": [{"bbox": [110, 110, 150, 150]}],
        "zone_breaches": [{"bbox": [160, 160, 200, 200], "zone_id": "zone_a"}],
    }
    result = renderer.draw(dummy_frame, events)
    assert result.shape == original.shape
    assert not np.array_equal(result, original)


# --- project_bbox: velocity projection over detection latency ---

FRAME_SHAPE = (480, 640, 3)


def test_project_bbox_identity_without_velocity_or_age():
    assert project_bbox([10, 10, 50, 90], None, 1.0, FRAME_SHAPE) == [10, 10, 50, 90]
    assert project_bbox([10, 10, 50, 90], (100.0, 0.0), 0.0, FRAME_SHAPE) == [10, 10, 50, 90]
    assert project_bbox([10, 10, 50, 90], (0.0, 0.0), 1.0, FRAME_SHAPE) == [10, 10, 50, 90]


def test_project_bbox_shifts_along_velocity():
    # 100 px/s for 0.5 s -> +50 px in x, size preserved.
    assert project_bbox([10, 10, 50, 90], (100.0, 0.0), 0.5, FRAME_SHAPE) == [60, 10, 100, 90]


def test_project_bbox_caps_extrapolation_time():
    # Stale events (age 5 s) only project 0.8 s worth of motion.
    assert project_bbox([10, 10, 50, 90], (100.0, 0.0), 5.0, FRAME_SHAPE) == [90, 10, 130, 90]


def test_project_bbox_clamps_to_frame():
    out = project_bbox([600, 10, 640, 90], (1000.0, 0.0), 0.8, FRAME_SHAPE)
    assert out == [600, 10, 640, 90]     # already at the right edge, w=40 kept
