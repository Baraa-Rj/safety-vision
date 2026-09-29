"""CameraStream with a mocked cv2.VideoCapture (no video files or cameras)."""
import time
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.camera import CameraStream, _is_rtsp

FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


class FakeCapture:
    """Scripted stand-in for cv2.VideoCapture."""

    instances = []
    open_results = []   # consumed per construction; default True
    frames = 3          # frames returned by read() before EOF / failures

    def __init__(self, src, backend=None):
        self.src = src
        self.backend = backend
        self.opened = FakeCapture.open_results.pop(0) if FakeCapture.open_results else True
        self.reads = 0
        self.released = False
        self.props = {}
        FakeCapture.instances.append(self)

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return 1000.0 if prop == cv2.CAP_PROP_FPS else 0.0

    def set(self, prop, value):
        self.props[prop] = value
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.reads = 0
        return True

    def read(self):
        if self.reads < FakeCapture.frames:
            self.reads += 1
            return True, FRAME
        return False, None

    def release(self):
        self.released = True


@pytest.fixture(autouse=True)
def fake_capture():
    FakeCapture.instances = []
    FakeCapture.open_results = []
    FakeCapture.frames = 3
    with patch("src.camera.cv2.VideoCapture", FakeCapture):
        yield FakeCapture


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def test_is_rtsp():
    assert _is_rtsp("rtsp://user:pw@cam/stream")
    assert _is_rtsp("RTSP://cam")
    assert not _is_rtsp("video.mp4")
    assert not _is_rtsp(0)


def test_file_source_reads_frames_and_rewinds_at_eof(fake_capture):
    cam = CameraStream("video.mp4")
    try:
        assert wait_for(lambda: cam.read() is not None)
        cap = fake_capture.instances[0]
        assert cap.backend is None
        assert wait_for(lambda: cv2.CAP_PROP_POS_FRAMES in cap.props)
        assert cap.props[cv2.CAP_PROP_POS_FRAMES] == 0
        assert cam._frame_delay == pytest.approx(1 / 1000)
    finally:
        cam.stop()
    assert cam.running is False
    assert fake_capture.instances[0].released


def test_file_source_is_opened_once_and_failure_raises(fake_capture):
    fake_capture.open_results = [False]
    with pytest.raises(RuntimeError, match="Cannot open video source: missing.mp4"):
        CameraStream("missing.mp4", open_max_attempts=5, open_retry_backoff=0)
    assert len(fake_capture.instances) == 1
    assert fake_capture.instances[0].released


def test_rtsp_open_retries_then_uses_ffmpeg_with_small_buffer(fake_capture):
    fake_capture.open_results = [False, False, True]
    cam = CameraStream("rtsp://cam/stream", open_max_attempts=5, open_retry_backoff=0)
    try:
        assert len(fake_capture.instances) == 3
        cap = cam.stream
        assert cap.backend == cv2.CAP_FFMPEG
        assert cap.props[cv2.CAP_PROP_BUFFERSIZE] == 1
        assert cam._frame_delay == 0.0
    finally:
        cam.stop()


def test_rtsp_open_gives_up_after_max_attempts(fake_capture):
    fake_capture.open_results = [False, False, False]
    with pytest.raises(RuntimeError, match="rtsp://cam/stream"):
        CameraStream("rtsp://cam/stream", open_max_attempts=3, open_retry_backoff=0)
    assert len(fake_capture.instances) == 3


def test_rtsp_reconnects_after_stalled_reads(fake_capture):
    fake_capture.frames = 1
    cam = CameraStream("rtsp://cam/stream", reconnect_after_seconds=0.05,
                       open_retry_backoff=0)
    try:
        first = fake_capture.instances[0]
        assert wait_for(lambda: len(fake_capture.instances) >= 2)
        assert first.released
        assert cam.read() is FRAME
    finally:
        cam.stop()
