import logging
import time

logger = logging.getLogger("safety_vision")


class EventLogger:
    def __init__(self, cooldown=15.0):
        self._cooldown = cooldown
        self._last_logged = {}

    def _should_log(self, key):
        now = time.time()
        last = self._last_logged.get(key, 0)
        if now - last >= self._cooldown:
            self._last_logged[key] = now
            return True
        return False

    def log_events(self, events):
        for violation in events["ppe_violations"]:
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
