"""Plain `pytest` must pass on a fresh clone, where the integration tests'
video and model weights (gitignored) are absent."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _collect(*args):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    return result.stdout


def test_integration_tests_are_deselected_by_default():
    collected = _collect()
    assert "test_reads_frame_from_video" not in collected
    assert "test_detect_on_real_frame" not in collected
    assert "test_rejects_invalid_source" in collected


def test_integration_tests_can_be_selected_explicitly():
    collected = _collect("-m", "integration")
    assert "test_reads_frame_from_video" in collected
    assert "test_detect_on_real_frame" in collected
