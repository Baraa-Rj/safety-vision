"""Startup checks for inputs that are not in git (sample video, model weights)."""
import os
from typing import Optional


def _is_stream_or_device(source) -> bool:
    source = str(source)
    return "://" in source or source.isdigit()


def missing_inputs_message(camera_source, model_path) -> Optional[str]:
    """Return a one-line description of missing inputs, or None if all exist."""
    problems = []
    if not _is_stream_or_device(camera_source) and not os.path.isfile(camera_source):
        problems.append(
            f"video source '{camera_source}' not found "
            "(copy a video there or set CAMERA_RTSP_URL)"
        )
    if not os.path.isfile(model_path):
        problems.append(
            f"model weights '{model_path}' not found (copy the detector weights there)"
        )
    if not problems:
        return None
    return "Cannot start: " + "; ".join(problems) + "."
