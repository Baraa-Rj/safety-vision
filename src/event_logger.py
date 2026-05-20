import logging
import time

logger = logging.getLogger("safety_vision")


class EventLogger:
    def __init__(self, cooldown=15.0, wet_floor_cooldown=60.0):
        self._cooldown = cooldown
        self._wet_floor_cooldown = wet_floor_cooldown
        self._last_logged = {}

    def _should_log(self, key, cooldown=None):
        now = time.time()
        last = self._last_logged.get(key, 0)
        effective = cooldown if cooldown is not None else self._cooldown
        if now - last >= effective:
            self._last_logged[key] = now
            return True
        return False

    def log_compliance_transition(self, track_id, item, transition, window_summary):
        # Kept at DEBUG so production runs aren't noisy; the warning-level
        # [PPE VIOLATION] line in log_events is the user-facing signal.
        logger.debug(
            "[COMPLIANCE TRANSITION] track=%s item=%s transition=%s window=%s",
            track_id, item, transition, window_summary,
        )

    def log_events(self, events):
        # PPE alerts log on the smoothed list — falls back to single-frame
        # ppe_violations for tests/older callers that don't populate it yet.
        ppe_source = events.get("confirmed_ppe_violations")
        if ppe_source is None:
            ppe_source = events["ppe_violations"]
        for violation in ppe_source:
            tid = violation.get("track_id")
            key = ("ppe", tid)
            if not self._should_log(key):
                continue
            name = violation.get("worker_name")
            wid = violation.get("worker_id")
            if name:
                label = f"Worker {name}"
            elif wid is not None:
                label = f"Worker {wid}"
            elif tid is not None:
                label = f"Track {tid}"
            else:
                label = "UNIDENTIFIED"
            logger.warning("[PPE VIOLATION] %s | Missing: %s", label, violation["missing"])

        for fall_event in events["falls"]:
            tid = fall_event.get("track_id")
            key = ("fall", tid)
            if not self._should_log(key):
                continue
            name = fall_event.get("worker_name")
            wid = fall_event.get("worker_id")
            if name:
                label = f"Worker {name}"
            elif wid is not None:
                label = f"Worker {wid}"
            elif tid is not None:
                label = f"Track {tid}"
            else:
                label = "UNIDENTIFIED"
            logger.warning("[FALL DETECTED] %s", label)

        for zone_event in events["zone_breaches"]:
            tid = zone_event.get("track_id")
            zone_id = zone_event.get("zone_id")
            key = ("zone", tid, zone_id)
            if not self._should_log(key):
                continue
            name = zone_event.get("worker_name")
            wid = zone_event.get("worker_id")
            if name:
                label = f"Worker {name}"
            elif wid is not None:
                label = f"Worker {wid}"
            elif tid is not None:
                label = f"Track {tid}"
            else:
                label = "UNIDENTIFIED"
            logger.warning("[ZONE BREACH] %s in zone %s", label, zone_id)

        # Wet floor events: bucket the bbox centroid so a stable detection
        # doesn't re-log every cooldown window from minor pixel drift.
        for wf_event in events.get("wet_floor_events", []):
            x1, y1, x2, y2 = wf_event["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            key = ("wet_floor", cx // 50, cy // 50)
            if not self._should_log(key, cooldown=self._wet_floor_cooldown):
                continue
            logger.warning(
                "[WET FLOOR] bbox=%s conf=%.2f area=%.1f%% streak=%d",
                wf_event["bbox"], wf_event["confidence"],
                wf_event["area_pct"], wf_event["consecutive_count"],
            )
