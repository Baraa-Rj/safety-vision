"""Classifier-based fall detection.

The PPE detector already produces a person box (and track id) for every worker
each frame. For each box we crop the worker and run a small image classifier
(fallen vs standing). A fall is reported only after the 'fallen' class is
sustained for `consecutive_frames` on the same track, which filters single-frame
misclassifications. Once confirmed, the person stays reported every frame they
remain fallen (so the renderer keeps the box up and the PPE check is suppressed
for someone on the ground); the per-track cooldown instead gates the `alert`
flag so the backend is notified once per event, not every frame.

This is intentionally single-frame per worker: the detector localizes people,
the classifier judges each crop — identical to how the model was trained on
person crops.
"""
import logging
import time

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class FallDetector:
    def __init__(self, model_path, config):
        self.config = config
        self.model = YOLO(model_path)
        if self.model.task != "classify":
            raise ValueError(
                f"{model_path} is a '{self.model.task}' model; fall detection "
                f"needs a fallen/standing classification model.")
        self.fallen_idx = self._resolve_class("fallen")
        self.imgsz = config.imgsz
        self.fallen_conf = config.fallen_conf
        self.consecutive = config.consecutive_frames
        self.cooldown_s = config.cooldown_seconds
        self.min_size = config.min_size
        self._streak = {}            # key -> consecutive fallen frames (capped)
        self._cooldown_until = {}    # key -> ts until which no new alert fires
        self._last_seen = {}         # key -> ts (for pruning stale tracks)

    def _resolve_class(self, name):
        for k, v in self.model.names.items():
            if str(v).lower() == name:
                return int(k)
        raise ValueError(f"No '{name}' class in {self.model.names}")

    def detect(self, ppe_results, frame):
        """Return a list of confirmed-fallen people.

        Each item: {person_index, track_id, bbox, confidence, alert}. `alert` is
        True only on the first confirmed frame and once per cooldown thereafter.
        `person_index` indexes into `ppe_results`.
        """
        now = time.time()
        crops, meta = [], []
        for i, r in enumerate(ppe_results):
            x1, y1, x2, y2 = r["person_bbox"]
            if (x2 - x1) < self.min_size or (y2 - y1) < self.min_size:
                continue
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            meta.append((i, r))

        falls = []
        if crops:
            preds = self.model.predict(crops, imgsz=self.imgsz, verbose=False)
            for (i, r), p in zip(meta, preds):
                prob = float(p.probs.data[self.fallen_idx])
                is_fallen = int(p.probs.top1) == self.fallen_idx and prob >= self.fallen_conf

                tid = r.get("track_id")
                key = tid if tid is not None else f"loc_{i}"  # weak, untracked fallback
                self._last_seen[key] = now

                streak = self._streak.get(key, 0)
                streak = min(streak + 1, self.consecutive) if is_fallen else 0
                self._streak[key] = streak

                if streak >= self.consecutive:
                    alert = now >= self._cooldown_until.get(key, 0)
                    if alert:
                        self._cooldown_until[key] = now + self.cooldown_s
                    falls.append({
                        "person_index": i,
                        "track_id": tid,
                        "bbox": r["person_bbox"],
                        "confidence": prob,
                        "alert": alert,
                    })
        self._prune(now)
        return falls

    def _prune(self, now, ttl=10.0):
        stale = [k for k, t in self._last_seen.items() if now - t > ttl]
        for k in stale:
            self._streak.pop(k, None)
            self._cooldown_until.pop(k, None)
            self._last_seen.pop(k, None)
