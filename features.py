"""Per-frame motion features and per-track buffering.

Imported by both extract_dataset.py (training data) and fall_detector.py
(inference) so the feature definition is guaranteed identical on both sides.
"""
from collections import deque
import numpy as np

# Features per frame. All pixel quantities are normalized by bbox height or
# frame height so values are roughly invariant to how far the worker is from
# the camera (a fall near the lens and one far away produce similar values).
FEATURE_NAMES = ["cy_norm", "aspect", "h_norm", "v_centroid", "v_bottom", "disp"]
N_FEATURES = len(FEATURE_NAMES)


def frame_features(prev_box, cur_box, frame_h):
    """prev_box / cur_box are (x1, y1, x2, y2) in pixels; prev_box may be None."""
    x1, y1, x2, y2 = cur_box
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    cy_norm = cy / frame_h
    aspect = w / h
    h_norm = h / frame_h

    if prev_box is None:
        return [cy_norm, aspect, h_norm, 0.0, 0.0, 0.0]

    px1, py1, px2, py2 = prev_box
    pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
    v_centroid = (cy - pcy) / h            # downward centroid velocity (+ = falling)
    v_bottom = (y2 - py2) / h              # bottom-edge velocity
    disp = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5 / h  # total motion magnitude
    return [cy_norm, aspect, h_norm, v_centroid, v_bottom, disp]


class TrackBuffer:
    """Sliding window of per-frame features for a single track id."""

    def __init__(self, window, max_gap=5):
        self.window = window
        self.max_gap = max_gap          # frames; a longer detection gap resets the buffer
        self.buf = deque(maxlen=window)
        self._prev = None
        self._last_frame = None

    def update(self, box, frame_idx, frame_h):
        if self._last_frame is not None and frame_idx - self._last_frame > self.max_gap:
            self.buf.clear()
            self._prev = None
        self.buf.append(frame_features(self._prev, box, frame_h))
        self._prev = box
        self._last_frame = frame_idx

    def ready(self):
        return len(self.buf) == self.window

    def array(self):
        return np.asarray(self.buf, dtype=np.float32)  # shape [window, N_FEATURES]
