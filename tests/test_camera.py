import pytest
from config.settings import _DEFAULT_CAMERA_SOURCE
from src.camera import CameraStream


def test_rejects_invalid_source():
    with pytest.raises(RuntimeError):
        CameraStream("nonexistent_file.mp4")


@pytest.mark.integration
def test_reads_frame_from_video():
    import time
    cam = CameraStream(_DEFAULT_CAMERA_SOURCE)
    time.sleep(2)
    frame = cam.read()
    cam.stop()
    assert frame is not None
    assert len(frame.shape) == 3
