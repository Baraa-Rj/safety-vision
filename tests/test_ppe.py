import pytest
from src.ppe_detector import PPEDetector


class TestHasSufficientOverlap:
    @pytest.fixture
    def detector(self):
        return PPEDetector.__new__(PPEDetector)

    def setup_method(self):
        self._detector = PPEDetector.__new__(PPEDetector)
        self._detector.overlap_threshold = 0.5

    def test_fully_inside(self):
        inner = [150, 150, 200, 200]
        outer = [100, 100, 300, 300]
        assert self._detector._has_sufficient_overlap(inner, outer) is True

    def test_no_overlap(self):
        inner = [0, 0, 50, 50]
        outer = [100, 100, 300, 300]
        assert self._detector._has_sufficient_overlap(inner, outer) is False

    def test_partial_overlap_below_threshold(self):
        inner = [90, 90, 110, 200]
        outer = [100, 100, 300, 300]
        assert self._detector._has_sufficient_overlap(inner, outer) is False

    def test_partial_overlap_above_threshold(self):
        inner = [100, 100, 200, 200]
        outer = [50, 50, 250, 250]
        assert self._detector._has_sufficient_overlap(inner, outer) is True

    def test_zero_area_inner(self):
        inner = [100, 100, 100, 100]
        outer = [50, 50, 200, 200]
        assert self._detector._has_sufficient_overlap(inner, outer) is False


@pytest.mark.integration
def test_detect_on_real_frame():
    import time
    from src.camera import CameraStream

    cam = CameraStream("data/sample_videos/output.mp4")
    detector = PPEDetector("models/best.pt")
    time.sleep(2)

    frame = cam.read()
    cam.stop()
    assert frame is not None

    results = detector.detect(frame)
    assert isinstance(results, list)
    for r in results:
        assert "person_bbox" in r
        assert "compliant" in r
        assert "detected_ppe" in r
        assert "missing_ppe" in r
