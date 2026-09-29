"""`python -m src.main` must explain missing inputs instead of crashing."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_main_exits_with_one_line_message_when_inputs_missing(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "CAMERA_RTSP_URL"}
    env["PYTHONPATH"] = str(ROOT)
    # Run from an empty directory so the default relative paths never exist.
    result = subprocess.run(
        [sys.executable, "-m", "src.main"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    message = result.stderr.strip().splitlines()[-1]
    assert "data/sample_videos/0712.mp4" in message
    assert "CAMERA_RTSP_URL" in message
    assert "models/best.pt" in message


def test_missing_inputs_message(tmp_path):
    from src.preflight import missing_inputs_message

    video = tmp_path / "video.mp4"
    weights = tmp_path / "best.pt"
    assert "video source" in missing_inputs_message(str(video), str(weights))
    video.write_bytes(b"")
    weights.write_bytes(b"")
    assert missing_inputs_message(str(video), str(weights)) is None
    # Streams and webcam indices are not files and are not checked.
    assert missing_inputs_message("rtsp://cam/stream", str(weights)) is None
    assert missing_inputs_message("0", str(weights)) is None
    assert "model weights" in missing_inputs_message("rtsp://cam/stream", str(tmp_path / "x.pt"))
