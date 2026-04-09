import cv2
import numpy as np
import pytest
from src.worker_id import WorkerIdentifier


def _generate_marker(marker_id, size=200, border=50):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
    padded = cv2.copyMakeBorder(marker, border, border, border, border,
                                cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def identifier():
    return WorkerIdentifier()


def test_detect_marker_42(identifier):
    assert identifier.identify(_generate_marker(42)) == 42


def test_detect_marker_7(identifier):
    assert identifier.identify(_generate_marker(7)) == 7


def test_no_marker_returns_none(identifier):
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    assert identifier.identify(blank) is None


def test_empty_crop_returns_none(identifier):
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert identifier.identify(empty) is None


def test_none_crop_returns_none(identifier):
    assert identifier.identify(None) is None
