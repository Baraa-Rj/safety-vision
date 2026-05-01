import json
import cv2
import numpy as np
import pytest
from src.worker_id import WorkerIdentifier


def _generate_qr(data, size=300):
    encoder = cv2.QRCodeEncoder.create()
    qr_img = encoder.encode(data)
    # Resize to desired size and convert to BGR
    qr_img = cv2.resize(qr_img, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(qr_img, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def identifier():
    return WorkerIdentifier()


def test_detect_qr_json(identifier):
    data = json.dumps({"id": "W042", "name": "John Doe"})
    result = identifier.identify(_generate_qr(data))
    assert result is not None
    assert result["worker_id"] == "W042"
    assert result["worker_name"] == "John Doe"
    assert result["qr_data"] == data


def test_detect_qr_plain_text(identifier):
    result = identifier.identify(_generate_qr("Alice"))
    assert result is not None
    assert result["worker_name"] == "Alice"
    assert result["worker_id"] is None
    assert result["qr_data"] == "Alice"


def test_no_qr_returns_none(identifier):
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    assert identifier.identify(blank) is None


def test_empty_crop_returns_none(identifier):
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert identifier.identify(empty) is None


def test_none_crop_returns_none(identifier):
    assert identifier.identify(None) is None
