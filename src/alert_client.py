import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import requests

logger = logging.getLogger("safety_vision")


class AlertClient:
    def __init__(self, endpoint, enabled=True, cooldown_seconds=30.0):
        self.endpoint = endpoint
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._last_alert_time = {}

    def send_ppe_alert(self, violation, frame):
        if not self.enabled:
            return

        worker_id = violation.get("worker_id")
        track_id = violation.get("track_id")

        if worker_id is not None:
            key = str(worker_id)
        elif track_id is not None:
            key = f"track_{track_id}"
        else:
            x1, y1, x2, y2 = violation["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            key = f"unknown_{cx // 50}_{cy // 50}"

        now = time.time()
        last = self._last_alert_time.get(key, 0)
        if now - last < self.cooldown_seconds:
            return

        self._last_alert_time[key] = now
        self._executor.submit(self._post_alert, violation, frame)

    def send_wet_floor_alert(self, wf_event, frame):
        if not self.enabled:
            return

        # Bucket centroid so jittery bboxes share a cooldown key.
        x1, y1, x2, y2 = wf_event["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        key = f"wet_floor_{cx // 50}_{cy // 50}"

        now = time.time()
        last = self._last_alert_time.get(key, 0)
        if now - last < self.cooldown_seconds:
            return

        self._last_alert_time[key] = now
        self._executor.submit(self._post_wet_floor_alert, wf_event, frame)

    def _post_wet_floor_alert(self, wf_event, frame):
        try:
            _, buffer = cv2.imencode(".jpg", frame)
            img_b64 = base64.b64encode(buffer).decode("utf-8")

            # zone_id tagging is deferred — ZoneMonitor exposes only
            # check_person (foot-of-bbox semantics for people). A point-in-zone
            # helper for arbitrary centroids can be added when needed.
            payload = {
                "imgImage": img_b64,
                "event_type": "wet_floor",
                "bbox": wf_event["bbox"],
                "confidence": wf_event["confidence"],
                "areaPct": wf_event["area_pct"],
                "consecutiveCount": wf_event.get("consecutive_count"),
                "firstSeenTs": wf_event.get("first_seen_ts"),
                "message": "Wet floor detected",
            }

            response = requests.post(self.endpoint, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("[WET FLOOR ALERT SENT] bbox=%s", wf_event["bbox"])
        except Exception as e:
            logger.error("[WET FLOOR ALERT FAILED] %s", e)

    def send_zone_alert(self, zone_event, frame):
        if not self.enabled:
            return

        worker_id = zone_event.get("worker_id")
        track_id = zone_event.get("track_id")
        zone_id = zone_event.get("zone_id")

        if worker_id is not None:
            key = f"zone_{zone_id}_{worker_id}"
        elif track_id is not None:
            key = f"zone_{zone_id}_track_{track_id}"
        else:
            x1, y1, x2, y2 = zone_event["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            key = f"zone_{zone_id}_unknown_{cx // 50}_{cy // 50}"

        now = time.time()
        last = self._last_alert_time.get(key, 0)
        if now - last < self.cooldown_seconds:
            return

        self._last_alert_time[key] = now
        self._executor.submit(self._post_zone_alert, zone_event, frame)

    def _post_zone_alert(self, zone_event, frame):
        try:
            _, buffer = cv2.imencode(".jpg", frame)
            img_b64 = base64.b64encode(buffer).decode("utf-8")

            worker_id = zone_event.get("worker_id")
            worker_name = zone_event.get("worker_name")
            zone_id = zone_event.get("zone_id")

            if worker_name:
                message = f"Worker {worker_name} entered restricted zone {zone_id}"
            elif worker_id:
                message = f"Worker {worker_id} entered restricted zone {zone_id}"
            else:
                message = f"Unknown worker entered restricted zone {zone_id}"

            payload = {
                "imgImage": img_b64,
                "zoneId": zone_id,
                "message": message,
            }
            if worker_id is not None:
                payload["userId"] = str(worker_id)
            if worker_name is not None:
                payload["workerName"] = worker_name

            response = requests.post(self.endpoint, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("[ZONE ALERT SENT] %s", message)
        except Exception as e:
            logger.error("[ZONE ALERT FAILED] %s", e)

    def _post_alert(self, violation, frame):
        try:
            _, buffer = cv2.imencode(".jpg", frame)
            img_b64 = base64.b64encode(buffer).decode("utf-8")

            missing = ", ".join(violation["missing"])
            worker_id = violation.get("worker_id")

            if worker_id is not None:
                message = f"Worker {worker_id} missing {missing}"
            else:
                message = f"Unknown worker missing {missing}"

            payload = {
                "imgImage": img_b64,
                "missingItem": missing,
                "message": message,
            }
            if worker_id is not None:
                payload["userId"] = str(worker_id)

            response = requests.post(self.endpoint, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("[ALERT SENT] %s", message)
        except Exception as e:
            logger.error("[ALERT FAILED] %s", e)

    def shutdown(self):
        self._executor.shutdown(wait=False)
