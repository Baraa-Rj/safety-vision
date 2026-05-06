import numpy as np
import pytest
from src.zone_monitor import ZoneMonitor
from src.renderer import FrameRenderer


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
        "ppe_violations": [{"bbox": [60, 60, 100, 100], "worker_id": 2, "missing": ["helmet"]}],
        "falls": [{"bbox": [110, 110, 150, 150]}],
        "zone_breaches": [{"bbox": [160, 160, 200, 200], "zone_id": "zone_a"}],
    }
    result = renderer.draw(dummy_frame, events)
    assert result.shape == original.shape
    assert not np.array_equal(result, original)
