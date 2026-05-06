import cv2
import numpy as np
import pytest
from src.worker_id import WorkerIdentifier


def _generate_qr(data, size=300):
    encoder = cv2.QRCodeEncoder.create()
    qr_img = encoder.encode(data)
    qr_img = cv2.resize(qr_img, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(qr_img, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def identifier():
    return WorkerIdentifier()


def test_detect_qr_uuid(identifier):
    uuid = "0cb43da1-eb90-4d5d-bae9-c4020a729030"
    result = identifier.identify(_generate_qr(uuid))
    assert result is not None
    assert result["worker_id"] == uuid
    assert result["worker_name"] is None
    assert result["qr_data"] == uuid


def test_no_qr_returns_none(identifier):
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    assert identifier.identify(blank) is None


def test_empty_crop_returns_none(identifier):
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert identifier.identify(empty) is None


def test_none_crop_returns_none(identifier):
    assert identifier.identify(None) is None
