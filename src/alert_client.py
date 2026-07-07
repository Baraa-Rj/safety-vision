import base64
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import requests

logger = logging.getLogger("safety_vision")


def _log_outgoing(url, payload):
    """Print the request we're about to send, with the base64 image shortened
    so the terminal stays readable."""
    shown = dict(payload)
    img = shown.get("imgImage")
    if isinstance(img, str):
        shown["imgImage"] = f"<base64 jpg, {len(img)} chars>"
    logger.info("[POST %s] %s", url, json.dumps(shown))


def _log_response(response):
    """Print what the server actually returned — status + body — so a 'sent'
    that didn't persist (redirect, empty 200, silent reject) is visible."""
    body = (response.text or "").replace("\n", " ")[:400]
    logger.info("[RESP %s] %s -> %s", response.status_code, response.url, body)


class AlertClient:
    def __init__(self, endpoint, enabled=True, cooldown_seconds=30.0,
                 zone_endpoint="", wet_floor_endpoint="", fall_endpoint="",
                 fall_unidentified_user_id="", zone_unidentified_user_id=""):
        self.endpoint = endpoint
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.zone_endpoint = zone_endpoint
        self.wet_floor_endpoint = wet_floor_endpoint
        self.fall_endpoint = fall_endpoint
        self.fall_unidentified_user_id = fall_unidentified_user_id
        self.zone_unidentified_user_id = zone_unidentified_user_id
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
            # Backend contract (POST /api/wet/alert/create) is exactly
            # {description, imgImage}. A wet floor is a location hazard, not a
            # person, so the frame is always attached (no id-only case).
            _, buffer = cv2.imencode(".jpg", frame)
            payload = {
                "description": f"Wet floor detected (confidence {wf_event['confidence']:.0%})",
                "imgImage": base64.b64encode(buffer).decode("utf-8"),
            }

            _log_outgoing(self.wet_floor_endpoint, payload)
            response = requests.post(self.wet_floor_endpoint, json=payload, timeout=10)
            _log_response(response)
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

            # Always send — identified or not. userId is the worker's id if known,
            # else the configured sentinel, else "" (the frame still shows who/where).
            # If the backend rejects a blank userId FK, set ZONE_UNIDENTIFIED_USER_ID
            # so anonymous breaches carry a valid id.
            user_id = worker_id or self.zone_unidentified_user_id or ""

            who = worker_name or worker_id or "Unidentified worker"
            message = f"{who} entered restricted zone {zone_id}"

            # Mark the breach on the photo so the alert clearly shows who/where —
            # box the breaching worker and label the zone.
            annotated = frame.copy()
            x1, y1, x2, y2 = (int(v) for v in zone_event["bbox"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 3)
            cv2.putText(annotated, f"ZONE BREACH: {zone_id}", (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # Backend contract is exactly {imgImage, userId, zoneId, message}.
            # zoneId is the backend's integer zone id (threaded from zones.json).
            _, buffer = cv2.imencode(".jpg", annotated)
            payload = {
                "imgImage": base64.b64encode(buffer).decode("utf-8"),
                "userId": str(user_id),
                "zoneId": int(zone_event.get("zone_backend_id", 0)),
                "message": message,
            }

            _log_outgoing(self.zone_endpoint, payload)
            response = requests.post(self.zone_endpoint, json=payload, timeout=10)
            _log_response(response)
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

            # The backend requires a userId FK. Use the worker's id if known,
            # else the configured sentinel for anonymous falls. With no sentinel
            # set, skip rather than POST a request the backend will reject.
            user_id = worker_id if worker_id is not None else self.fall_unidentified_user_id
            if not user_id:
                logger.warning(
                    "[FALL ALERT SKIPPED] unidentified worker and no "
                    "fall_unidentified_user_id configured")
                return

            who = worker_name or worker_id or "Unidentified worker"
            message = f"{who} has fallen ({severity})"

            # Falls always carry the frame regardless of identity — responders
            # need to see the scene to gauge the situation.
            _, buffer = cv2.imencode(".jpg", frame)
            payload = {
                "userId": str(user_id),
                "severity": severity,
                "imgImage": base64.b64encode(buffer).decode("utf-8"),
                "message": message,
            }

            _log_outgoing(self.fall_endpoint, payload)
            response = requests.post(self.fall_endpoint, json=payload, timeout=10)
            _log_response(response)
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

            _log_outgoing(self.endpoint, payload)
            response = requests.post(self.endpoint, json=payload, timeout=10)
            _log_response(response)
            response.raise_for_status()
            logger.info("[ALERT SENT] %s", message)
        except Exception as e:
            logger.error("[ALERT FAILED] %s", e)

    def shutdown(self):
        self._executor.shutdown(wait=False)
