"""Detection-based fall detection.

The PPE detector (best.pt) now has a dedicated ``fallen`` class, so a worker on
the ground is detected directly — no crop, no separate classifier. This module
is purely the temporal gate over those ``fallen`` detections: a fall is reported
only after ``fallen`` is sustained for ``consecutive_frames`` on one track, which
filters single-frame false positives. Brief detection dropouts are tolerated (a
short gap doesn't reset the streak), but a long gap restarts it so sporadic false
boxes can't accumulate into an alert.

Once confirmed, the person stays reported every frame they remain on the ground
(so the renderer keeps the box up); the per-track cooldown gates the ``alert``
flag so the backend is notified once per event, not every frame.
"""
import time

# A gap longer than this (seconds) since a track was last seen as fallen means
# the presence wasn't sustained — restart its streak. Shorter gaps (flicker)
# keep the streak, which makes the gate robust to momentary detector dropouts.
_PRESENCE_GAP = 1.5


class FallDetector:
    def __init__(self, config):
        self.config = config
        self.fallen_conf = config.fallen_conf
        self.consecutive = config.consecutive_frames
        self.cooldown_s = config.cooldown_seconds
        self.min_size = config.min_size
        self.max_aspect_ratio = config.max_aspect_ratio
        self._streak = {}            # key -> consecutive fallen frames (capped)
        self._cooldown_until = {}    # key -> ts until which no new alert fires
        self._last_seen = {}         # key -> ts (gap detection + pruning)

    def detect(self, fallen_detections):
        """Gate raw ``fallen`` detections into confirmed falls.

        `fallen_detections`: list of {bbox, track_id, confidence} from the PPE
        detector. Returns a list of {track_id, bbox, confidence, alert}; `alert`
        is True only on the first confirmed frame and once per cooldown after.
        """
        now = time.time()
        falls = []
        for d in fallen_detections:
            x1, y1, x2, y2 = d["bbox"]
            w, h = x2 - x1, y2 - y1
            if w < self.min_size or h < self.min_size:
                continue
            if d.get("confidence", 1.0) < self.fallen_conf:
                continue
            # A fallen body is horizontal; reject implausibly tall/narrow boxes.
            if w > 0 and h / w > self.max_aspect_ratio:
                continue

            tid = d.get("track_id")
            key = tid if tid is not None else f"loc_{x1 // 50}_{y1 // 50}"

            last = self._last_seen.get(key, 0)
            streak = self._streak.get(key, 0)
            if now - last > _PRESENCE_GAP:
                streak = 0  # presence not sustained -> restart
            streak = min(streak + 1, self.consecutive)
            self._streak[key] = streak
            self._last_seen[key] = now

            if streak >= self.consecutive:
                alert = now >= self._cooldown_until.get(key, 0)
                if alert:
                    self._cooldown_until[key] = now + self.cooldown_s
                falls.append({
                    "track_id": tid,
                    "bbox": d["bbox"],
                    "confidence": d.get("confidence"),
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
