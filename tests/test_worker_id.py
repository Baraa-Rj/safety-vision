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


# --- identity persistence ---

UUID = "0cb43da1-eb90-4d5d-bae9-c4020a729030"


def test_identity_survives_absence_from_active_set(identifier):
    """A missed detection drops the track from one frame's active set — the
    identity must survive that (the old clear_stale deleted it instantly)."""
    identifier.identify(_generate_qr(UUID), track_id=5)
    for _ in range(10):
        identifier.clear_stale(active_track_ids=[])
    assert identifier.get_cached(5)["worker_id"] == UUID


def test_identity_pruned_after_ttl():
    identifier = WorkerIdentifier(cache_ttl=10.0)
    identifier.identify(_generate_qr(UUID), track_id=5)
    identifier._cache[5]["timestamp"] -= 11.0    # absent longer than the TTL
    identifier.clear_stale(active_track_ids=[])
    assert identifier.get_cached(5) is None


def test_transfer_moves_identity_to_new_track(identifier):
    identifier.identify(_generate_qr(UUID), track_id=5)
    moved = identifier.transfer(5, 9)
    assert moved["worker_id"] == UUID
    assert identifier.get_cached(9)["worker_id"] == UUID
    assert identifier.get_cached(5) is None      # old id no longer resolves


def test_transfer_without_cached_identity_is_noop(identifier):
    assert identifier.transfer(123, 456) is None
    assert identifier.get_cached(456) is None
