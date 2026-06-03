import logging
import math
import os
from collections import deque

import torch

logger = logging.getLogger(__name__)

# COCO-17 keypoint indices
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _mid(p, q):
    if p is None:
        return q
    if q is None:
        return p
    return ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)


class FallDetector:
    """Pose-based fall detection.

    A fall is inferred from body orientation rather than a trained classifier:
    when the torso (shoulders->hips vector) is more horizontal than vertical, the
    person is lying down. A temporal gate requires the posture to persist for a
    few frames so a quick bend/crouch doesn't trigger a fall.

    detect() keeps the same signature the pipeline already calls — it takes the
    PPE detector's person results and returns falls indexed back to them, so
    worker identity and zone logic still attach to a fall event.
    """

    def __init__(self, model_path, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_half = self.device != "cpu"
        self.model_path = model_path
        self.model = None
        self._missing_model_warned = False
        # Per-track posture history for the temporal gate.
        self._history = {}

    def _ensure_model(self):
        if self.model is not None:
            return True
        if not os.path.exists(self.model_path):
            if not self._missing_model_warned:
                logger.warning(
                    "Pose model not found at %s; fall detection disabled.",
                    self.model_path,
                )
                self._missing_model_warned = True
            return False
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        return True

    def _is_fallen_posture(self, keypoints):
        """Single-frame posture test from one person's 17 keypoints [x,y,conf]."""
        min_c = self.config.min_keypoint_conf

        def pt(i):
            x, y, c = keypoints[i]
            return (x, y) if c >= min_c else None

        shoulders = _mid(pt(L_SHOULDER), pt(R_SHOULDER))
        hips = _mid(pt(L_HIP), pt(R_HIP))

        if shoulders is not None and hips is not None:
            dx = hips[0] - shoulders[0]
            dy = hips[1] - shoulders[1]
            # Angle of the torso from vertical: 0 deg = upright, 90 deg = flat.
            angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
            return angle >= self.config.torso_angle_threshold

        # Fallback when torso keypoints are missing: aspect ratio of whatever
        # keypoints we do have. A horizontal body is wider than it is tall.
        pts = [(x, y) for x, y, c in keypoints if c >= min_c]
        if len(pts) < 3:
            return False
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if h <= 1:
            return False
        return (w / h) >= self.config.aspect_ratio_threshold

    def _confirm(self, track_id, is_fallen):
        """Temporal gate: require sustained fallen posture before flagging.

        Untracked persons (track_id is None) bypass the gate — safety-first, we
        can't accumulate history without an identity.
        """
        if track_id is None:
            return is_fallen
        n = self.config.consecutive_frames
        dq = self._history.setdefault(track_id, deque(maxlen=n))
        dq.append(bool(is_fallen))
        return len(dq) == n and all(dq)

    def detect(self, person_bboxes, frame):
        """Return [{person_index, bbox}] for persons whose posture reads as fallen.

        person_bboxes: PPE detector results (each has 'person_bbox', 'track_id').
        """
        if not self._ensure_model():
            return []

        results = self.model(
            frame, conf=self.config.person_conf, verbose=False,
            imgsz=640, half=self.use_half,
        )[0]

        if results.keypoints is None or results.boxes is None or len(results.boxes) == 0:
            self._prune(person_bboxes)
            return []

        pose_boxes = results.boxes.xyxy.tolist()
        pose_kpts = results.keypoints.data.tolist()  # [n, 17, 3]

        falls = []
        for idx, person in enumerate(person_bboxes):
            pbox = person["person_bbox"]
            # Match this PPE person to the closest pose detection.
            best_kp, best_iou = None, self.config.match_iou
            for pbox2, kp in zip(pose_boxes, pose_kpts):
                iou = _iou(pbox, pbox2)
                if iou >= best_iou:
                    best_iou, best_kp = iou, kp
            if best_kp is None:
                continue

            fallen = self._is_fallen_posture(best_kp)
            if self._confirm(person.get("track_id"), fallen):
                falls.append({"person_index": idx, "bbox": pbox})

        self._prune(person_bboxes)
        return falls

    def _prune(self, person_bboxes):
        """Drop posture history for tracks no longer present."""
        active = {p.get("track_id") for p in person_bboxes if p.get("track_id") is not None}
        for tid in list(self._history):
            if tid not in active:
                del self._history[tid]
