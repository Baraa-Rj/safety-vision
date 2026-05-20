import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from config.settings import WetFloorConfig
from src.wet_floor_detector import WetFloorDetector


@pytest.fixture
def missing_model_config(tmp_path):
    return WetFloorConfig(
        enabled=True,
        model_path=str(tmp_path / "does_not_exist.pt"),
        confidence_threshold=0.5,
        min_area_pct=1.0,
        consecutive_frames_required=5,
    )


def test_instantiation_without_model_file_does_not_raise(missing_model_config):
    detector = WetFloorDetector(missing_model_config)
    assert detector.is_loaded is False
    assert detector.config is missing_model_config


def test_detect_returns_empty_when_model_missing(missing_model_config, dummy_frame, caplog):
    detector = WetFloorDetector(missing_model_config)

    with caplog.at_level(logging.WARNING, logger="src.wet_floor_detector"):
        first = detector.detect(dummy_frame)
        second = detector.detect(dummy_frame)
        third = detector.detect(dummy_frame)

    assert first == []
    assert second == []
    assert third == []

    # Warning fires exactly once even across many calls.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Wet floor model not found" in warnings[0].message


def test_detect_parses_yolo_output(dummy_frame):
    cfg = WetFloorConfig(enabled=True, confidence_threshold=0.5, min_area_pct=0.0)
    detector = WetFloorDetector.__new__(WetFloorDetector)
    detector.config = cfg
    detector.device = "cpu"
    detector.use_half = False
    detector._missing_model_warned = False

    box = MagicMock()
    box.xyxy = [np.array([100.0, 100.0, 200.0, 200.0])]
    box.conf = [np.array(0.83)]
    fake_result = MagicMock()
    fake_result.boxes = [box]
    # Pre-set model so _ensure_model short-circuits without importing ultralytics.
    detector.model = MagicMock(return_value=[fake_result])

    results = detector.detect(dummy_frame)

    assert len(results) == 1
    r = results[0]
    assert r["bbox"] == [100, 100, 200, 200]
    assert r["confidence"] == pytest.approx(0.83, abs=1e-5)
    assert r["area_pct"] > 0


def test_area_pct_calculation():
    # 480x640 frame = 307_200 px; a 100x100 box covers 10_000 px ≈ 3.255%
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cfg = WetFloorConfig(enabled=True, confidence_threshold=0.1, min_area_pct=0.0)
    detector = WetFloorDetector.__new__(WetFloorDetector)
    detector.config = cfg
    detector.device = "cpu"
    detector.use_half = False
    detector._missing_model_warned = False

    box = MagicMock()
    box.xyxy = [np.array([0.0, 0.0, 100.0, 100.0])]
    box.conf = [np.array(0.9)]
    fake_result = MagicMock()
    fake_result.boxes = [box]
    detector.model = MagicMock(return_value=[fake_result])

    results = detector.detect(frame)

    assert len(results) == 1
    expected = 10_000 / (480 * 640) * 100.0
    assert results[0]["area_pct"] == pytest.approx(expected, rel=1e-4)


def test_min_area_pct_filters_small_boxes():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cfg = WetFloorConfig(enabled=True, confidence_threshold=0.1, min_area_pct=5.0)
    detector = WetFloorDetector.__new__(WetFloorDetector)
    detector.config = cfg
    detector.device = "cpu"
    detector.use_half = False
    detector._missing_model_warned = False

    # 100x100 box = ~3.25% — below the 5% threshold.
    box = MagicMock()
    box.xyxy = [np.array([0.0, 0.0, 100.0, 100.0])]
    box.conf = [np.array(0.9)]
    fake_result = MagicMock()
    fake_result.boxes = [box]
    detector.model = MagicMock(return_value=[fake_result])

    assert detector.detect(frame) == []
