import logging

logger = logging.getLogger("safety_vision")


class EventLogger:
    def log_events(self, events):
        for violation in events["ppe_violations"]:
            wid = violation["worker_id"] or "UNIDENTIFIED"
            logger.warning("[PPE VIOLATION] Worker %s | Missing: %s", wid, violation["missing"])

        for fall_event in events["falls"]:
            wid = fall_event["worker_id"] or "UNIDENTIFIED"
            logger.warning("[FALL DETECTED] Worker %s", wid)

        for zone_event in events["zone_breaches"]:
            wid = zone_event["worker_id"] or "UNIDENTIFIED"
            logger.warning("[ZONE BREACH] Worker %s in zone %s", wid, zone_event["zone_id"])
