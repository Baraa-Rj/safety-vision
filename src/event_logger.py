import logging

logger = logging.getLogger("safety_vision")


class EventLogger:
    def log_events(self, events):
        for violation in events["ppe_violations"]:
            name = violation.get("worker_name")
            wid = violation.get("worker_id")
            tid = violation.get("track_id")
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
            name = fall_event.get("worker_name")
            wid = fall_event.get("worker_id")
            tid = fall_event.get("track_id")
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
            name = zone_event.get("worker_name")
            wid = zone_event.get("worker_id")
            tid = zone_event.get("track_id")
            if name:
                label = f"Worker {name}"
            elif wid is not None:
                label = f"Worker {wid}"
            elif tid is not None:
                label = f"Track {tid}"
            else:
                label = "UNIDENTIFIED"
            logger.warning("[ZONE BREACH] %s in zone %s", label, zone_event["zone_id"])
