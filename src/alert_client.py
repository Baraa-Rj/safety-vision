import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import requests

logger = logging.getLogger("safety_vision")


class AlertClient:
    def __init__(self, endpoint, enabled=True, cooldown_seconds=30.0,
                 zone_endpoint="", wet_floor_endpoint="", fall_endpoint=""):
        self.endpoint = endpoint
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.zone_endpoint = zone_endpoint
        self.wet_floor_endpoint = wet_floor_endpoint
        self.fall_endpoint = fall_endpoint
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
        if not self.enabled or not self.wet_floor_endpoint:
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

            response = requests.post(self.wet_floor_endpoint, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("[WET FLOOR ALERT SENT] bbox=%s", wf_event["bbox"])
        except Exception as e:
            logger.error("[WET FLOOR ALERT FAILED] %s", e)

    def send_zone_alert(self, zone_event, frame):
        if not self.enabled or not self.zone_endpoint:
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
                "zoneId": zone_id,
                "message": message,
            }
            if worker_id is not None:
                # Identified worker: skip the frame to keep the payload small.
                payload["userId"] = str(worker_id)
            else:
                # Unidentified worker: attach the frame for manual review.
                _, buffer = cv2.imencode(".jpg", frame)
                payload["imgImage"] = base64.b64encode(buffer).decode("utf-8")
            if worker_name is not None:
                payload["workerName"] = worker_name

            response = requests.post(self.zone_endpoint, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("[ZONE ALERT SENT] %s", message)
        except Exception as e:
            logger.error("[ZONE ALERT FAILED] %s", e)

    def send_fall_alert(self, fall, frame):
        # Detector already cooldown-gates falls (one per event per track), so no
        # second cooldown here — just gate on config and an explicit endpoint.
        if not self.enabled or not self.fall_endpoint:
            return
        self._executor.submit(self._post_fall_alert, fall, frame)

    def _post_fall_alert(self, fall, frame):
        try:
            worker_id = fall.get("worker_id")
            worker_name = fall.get("worker_name")
            severity = fall.get("severity") or "LOW"

            who = worker_name or worker_id or "Unknown worker"
            message = f"{who} has fallen ({severity})"

            # Falls always carry the frame regardless of identity — responders
            # need to see the scene to gauge the situation.
            _, buffer = cv2.imencode(".jpg", frame)
            payload = {
                "userId": str(worker_id) if worker_id is not None else "",
                "severity": severity,
                "imgImage": base64.b64encode(buffer).decode("utf-8"),
                "message": message,
            }

            response = requests.post(self.fall_endpoint, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("[FALL ALERT SENT] %s", message)
        except Exception as e:
            logger.error("[FALL ALERT FAILED] %s", e)

    def _post_alert(self, violation, frame):
        try:
            missing = ", ".join(violation["missing"])
            worker_id = violation.get("worker_id")

            if worker_id is not None:
                message = f"Worker {worker_id} missing {missing}"
            else:
                message = f"Unknown worker missing {missing}"

            payload = {
                "missingItem": missing,
                "message": message,
            }
            if worker_id is not None:
                # Identified worker: backend already knows who they are, so the
                # frame adds no value — skip it to keep the payload small.
                payload["userId"] = str(worker_id)
            else:
                # Unidentified worker: attach the frame so the violation can be
                # reviewed manually.
                _, buffer = cv2.imencode(".jpg", frame)
                payload["imgImage"] = base64.b64encode(buffer).decode("utf-8")

            response = requests.post(self.endpoint, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("[ALERT SENT] %s", message)
        except Exception as e:
            logger.error("[ALERT FAILED] %s", e)

    def shutdown(self):
        self._executor.shutdown(wait=False)
