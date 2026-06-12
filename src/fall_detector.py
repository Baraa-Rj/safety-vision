"""Detection-based fall detection with severity scoring.

The PPE detector (best.pt) has a dedicated ``fallen`` class, so a worker on the
ground is detected directly — no crop, no separate classifier. This module is the
temporal gate over those detections plus a triage **severity** score.

Gate: a fall is confirmed only after ``fallen`` is sustained for
``consecutive_frames`` on one track (single-frame false positives filtered; brief
dropouts tolerated, long gaps reset). Tall/narrow boxes are rejected — a fallen
body is horizontal.

Severity (LOW/MEDIUM/HIGH) is a priority proxy, not a medical measure: it rises
the longer a worker stays down and the stiller they are. A backend ``alert`` is
flagged on the first confirmation and again each time the tier escalates, so help
can be re-prioritised for someone who isn't getting up — but never spammed.
"""
import time

# A gap longer than this (seconds) since a track was last seen as fallen means
# the presence wasn't sustained — restart its streak (and end the fall event).
_PRESENCE_GAP = 1.5

# Severity tiers as ordered levels so escalation is a simple comparison.
_TIERS = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


class FallDetector:
    def __init__(self, config):
        self.config = config
        self.fallen_conf = config.fallen_conf
        self.consecutive = config.consecutive_frames
        self.min_size = config.min_size
        self.max_aspect_ratio = config.max_aspect_ratio
        self.alert_delay_seconds = config.alert_delay_seconds
        self.still_motion_px = config.still_motion_px
        self.medium_seconds = config.medium_seconds
        self.high_still_seconds = config.high_still_seconds
        self.recovery_seconds = config.recovery_seconds
        self._streak = {}            # key -> consecutive fallen frames (capped)
        self._last_seen = {}         # key -> ts (gap detection + pruning)
        self._fall_since = {}        # key -> ts of first confirmation (event start)
        self._last_centroid = {}     # key -> (cx, cy) for motion
        self._moving_since = {}      # key -> ts last significant movement
        self._alerted_tier = {}      # key -> highest tier already alerted

    def _severity(self, key, now):
        """Tier from how long the worker has been down and how still they are."""
        still_seconds = now - self._moving_since.get(key, now)
        duration = now - self._fall_since.get(key, now)
        if still_seconds >= self.high_still_seconds:
            tier = "HIGH"          # motionless a long time -> urgent
        elif duration >= self.medium_seconds:
            tier = "MEDIUM"        # sustained on the ground
        else:
            tier = "LOW"           # just fell
        return tier, still_seconds

    def detect(self, fallen_detections):
        """Gate raw ``fallen`` detections into confirmed falls with severity.

        `fallen_detections`: list of {bbox, track_id, confidence}. Returns a list
        of {track_id, bbox, confidence, severity, still_seconds, alert}. `alert`
        is True on the first confirmed frame and whenever severity escalates.
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
            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            gap = now - self._last_seen.get(key, 0)
            streak = self._streak.get(key, 0)
            if gap > self.recovery_seconds:
                streak = 0
                self._reset_event(key)   # long absence -> worker recovered, new event
            elif gap > _PRESENCE_GAP:
                streak = 0               # brief dropout -> re-confirm, keep event timing
            streak = min(streak + 1, self.consecutive)
            self._streak[key] = streak
            self._last_seen[key] = now

            # Motion tracking (runs every frame the track is fallen).
            prev = self._last_centroid.get(key)
            if prev is None or _dist(centroid, prev) > self.still_motion_px:
                self._moving_since[key] = now
            self._last_centroid[key] = centroid

            if streak < self.consecutive:
                continue

            # Confirmed fall. Mark event start once.
            if key not in self._fall_since:
                self._fall_since[key] = now
                self._moving_since.setdefault(key, now)

            tier, still_seconds = self._severity(key, now)
            duration = now - self._fall_since[key]
            # Hold the first backend alert until the fall has persisted
            # alert_delay_seconds; after that, alert on each escalated tier.
            prev_tier = self._alerted_tier.get(key, -1)
            alert = duration >= self.alert_delay_seconds and _TIERS[tier] > prev_tier
            if alert:
                self._alerted_tier[key] = _TIERS[tier]

            falls.append({
                "track_id": tid,
                "bbox": d["bbox"],
                "confidence": d.get("confidence"),
                "severity": tier,
                "still_seconds": round(still_seconds, 1),
                "alert": alert,
            })
        self._prune(now)
        return falls

    def _reset_event(self, key):
        self._fall_since.pop(key, None)
        self._moving_since.pop(key, None)
        self._alerted_tier.pop(key, None)

    def _prune(self, now, ttl=10.0):
        stale = [k for k, t in self._last_seen.items() if now - t > ttl]
        for k in stale:
            self._streak.pop(k, None)
            self._last_seen.pop(k, None)
            self._last_centroid.pop(k, None)
            self._reset_event(k)


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
