"""PPEDetector.detect() with a mocked YOLO model (no weights needed)."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from src.ppe_detector import PPEDetector

NAMES = {0: "person", 1: "helmet", 2: "vest", 3: "fallen"}
CLS = {name: idx for idx, name in NAMES.items()}


def box(cls_name, bbox, conf, track_id=None):
    return SimpleNamespace(
        xyxy=np.array([bbox], dtype=float),
        cls=np.array([CLS[cls_name]]),
        conf=np.array([conf]),
        id=None if track_id is None else np.array([track_id]),
    )


class FakeYOLO:
    def __init__(self, model_path):
        self.model_path = model_path
        self.names = NAMES
        self.boxes = []
        self.track_kwargs = None
        self.device = None

    def to(self, device):
        self.device = device

    def track(self, frame, **kwargs):
        self.track_kwargs = kwargs
        return [SimpleNamespace(boxes=self.boxes)]


@pytest.fixture
def make_detector():
    def _make(**kwargs):
        with patch("src.ppe_detector.YOLO", FakeYOLO), \
                patch("src.ppe_detector.torch.cuda.is_available", return_value=False):
            return PPEDetector("models/best.pt", **kwargs)
    return _make


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
PERSON = [100, 100, 200, 400]        # height 300: head near y=100, feet near y=400
HELMET = [130, 100, 170, 140]        # centre y=120 -> top of the box
VEST = [110, 180, 190, 280]          # centre y=230 -> torso
BOOTS_LEVEL = [120, 360, 180, 400]   # centre y=380 -> feet


def test_init_uses_cpu_and_lowest_threshold_for_inference(make_detector):
    detector = make_detector(confidence=0.5, class_confidences={"vest": 0.3, "person": 0.6},
                             ppe_uncertain_floor=0.25)
    assert detector.device == "cpu"
    assert detector.use_half is False
    assert detector.model.device == "cpu"
    assert detector.model.model_path == "models/best.pt"
    assert detector._inference_conf == 0.25
    assert detector._class_threshold("vest") == 0.3
    assert detector._class_threshold("helmet") == 0.5


def test_detect_passes_tracking_options(make_detector):
    detector = make_detector(imgsz=480, tracker_config="custom.yaml")
    detector.detect(FRAME)
    kwargs = detector.model.track_kwargs
    assert kwargs["imgsz"] == 480
    assert kwargs["tracker"] == "custom.yaml"
    assert kwargs["persist"] is True
    assert kwargs["conf"] == detector._inference_conf


def test_worker_with_helmet_and_vest_is_compliant(make_detector):
    detector = make_detector()
    detector.model.boxes = [
        box("person", PERSON, 0.9, track_id=7),
        box("helmet", HELMET, 0.8),
        box("vest", VEST, 0.7),
    ]
    [result] = detector.detect(FRAME)
    assert result["compliant"] is True
    assert result["track_id"] == 7
    assert set(result["detected_ppe"]) == {"helmet", "vest"}
    assert result["missing_ppe"] == []
    assert result["person_bbox"] == PERSON


def test_missing_vest_is_reported(make_detector):
    detector = make_detector()
    detector.model.boxes = [box("person", PERSON, 0.9), box("helmet", HELMET, 0.8)]
    [result] = detector.detect(FRAME)
    assert result["compliant"] is False
    assert result["missing_ppe"] == ["vest"]
    assert result["track_id"] is None


def test_sub_threshold_vest_is_uncertain_evidence_not_detection(make_detector):
    detector = make_detector(class_confidences={"vest": 0.35}, ppe_uncertain_floor=0.25)
    detector.model.boxes = [
        box("person", PERSON, 0.9),
        box("helmet", HELMET, 0.8),
        box("vest", VEST, 0.30),
    ]
    [result] = detector.detect(FRAME)
    assert "vest" in result["missing_ppe"]
    assert result["ppe_confidences"]["vest"] == pytest.approx(0.30)


def test_boxes_below_their_thresholds_are_dropped(make_detector):
    detector = make_detector(confidence=0.35, ppe_uncertain_floor=0.25)
    detector.model.boxes = [
        box("person", PERSON, 0.2),           # below person threshold
        box("person", [300, 100, 400, 400], 0.9),
        box("vest", [310, 180, 390, 280], 0.1),  # below the uncertain floor
    ]
    results = detector.detect(FRAME)
    assert [r["person_bbox"] for r in results] == [[300, 100, 400, 400]]
    assert results[0]["ppe_confidences"] == {}


def test_ppe_outside_the_expected_body_region_does_not_count(make_detector):
    detector = make_detector()
    detector.model.boxes = [
        box("person", PERSON, 0.9),
        box("helmet", BOOTS_LEVEL, 0.9),   # helmet at the feet
        box("vest", BOOTS_LEVEL, 0.9),     # vest on the floor
    ]
    [result] = detector.detect(FRAME)
    assert set(result["missing_ppe"]) == {"helmet", "vest"}


def test_vest_between_two_workers_is_credited_to_one(make_detector):
    detector = make_detector()
    left = [100, 100, 200, 400]
    right = [180, 100, 280, 400]
    vest = [150, 180, 230, 280]  # mostly inside the right worker (x 180-230)
    detector.model.boxes = [
        box("person", left, 0.9, 1),
        box("person", right, 0.9, 2),
        box("vest", vest, 0.9),
    ]
    results = {r["track_id"]: r for r in detector.detect(FRAME)}
    assert "vest" in results[2]["detected_ppe"]
    assert "vest" not in results[1]["detected_ppe"]


def test_fallen_worker_replaces_overlapping_person(make_detector):
    detector = make_detector()
    fallen_box = [90, 300, 400, 400]
    detector.model.boxes = [
        box("fallen", fallen_box, 0.8, track_id=3),
        box("person", [100, 310, 390, 400], 0.9),   # same body, also seen as person
        box("person", [450, 100, 550, 400], 0.9),   # standing worker elsewhere
    ]
    results = detector.detect(FRAME)
    assert [r["person_bbox"] for r in results] == [[450, 100, 550, 400]]
    assert detector.fallen_detections == [
        {"bbox": fallen_box, "track_id": 3, "confidence": pytest.approx(0.8)}
    ]


def test_fallen_detections_reset_each_frame(make_detector):
    detector = make_detector()
    detector.model.boxes = [box("fallen", [90, 300, 400, 400], 0.8)]
    detector.detect(FRAME)
    assert len(detector.fallen_detections) == 1
    detector.model.boxes = []
    assert detector.detect(FRAME) == []
    assert detector.fallen_detections == []


def test_region_and_overlap_edge_cases(make_detector):
    detector = make_detector()
    assert detector._in_expected_region("helmet", HELMET, [0, 50, 10, 50]) is False  # zero height
    assert detector._in_expected_region("other", BOOTS_LEVEL, PERSON) is True
    assert detector._overlap_ratio([5, 5, 5, 5], [0, 0, 10, 10]) == 0.0  # zero-area inner box
