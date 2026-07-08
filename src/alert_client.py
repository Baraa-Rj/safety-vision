import base64
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import requests

logger = logging.getLogger("safety_vision")


def _same_spill(box_a, box_b):
    """Two wet-floor boxes are the same spill if they overlap or their
    centers sit within the larger box dimension of each other."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 > ix1 and iy2 > iy1:
        return True
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    reach = max(ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1, 1)
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 <= reach


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
                 fall_unidentified_user_id="", zone_unidentified_user_id="",
                 wet_floor_first_alert_seconds=10.0,
                 wet_floor_cooldown_seconds=10.0):
        self.endpoint = endpoint
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.wet_floor_first_alert_seconds = wet_floor_first_alert_seconds
        self.wet_floor_cooldown_seconds = wet_floor_cooldown_seconds
        self.zone_endpoint = zone_endpoint
        self.wet_floor_endpoint = wet_floor_endpoint
        self.fall_endpoint = fall_endpoint
        self.fall_unidentified_user_id = fall_unidentified_user_id
        self.zone_unidentified_user_id = zone_unidentified_user_id
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._last_alert_time = {}
        self._alert_count = {}
        self._wet_spills = {}   # backoff key -> last bbox of that spill

    # A key silent this long starts a fresh episode (backoff counter resets).
    _BACKOFF_RESET_SECONDS = 300.0

    def _allow(self, key, now, base=None):
        """Arithmetic backoff per alert key instead of a fixed cooldown.

        The first alert goes immediately (upstream already gates on sustained
        evidence); after k alerts the next needs a gap of base * (k + 1) —
        with an 8s base that is 16s, then 24s, then 32s, ... — so an ongoing
        violation keeps notifying, but ever less often instead of spamming at
        a fixed rate. `base` defaults to cooldown_seconds; alert types with
        their own cadence (wet floor) pass theirs."""
        if base is None:
            base = self.cooldown_seconds
        count = self._alert_count.get(key, 0)
        last = self._last_alert_time.get(key)
        if last is not None:
            if now - last >= self._BACKOFF_RESET_SECONDS:
                count = 0
            elif now - last < base * (count + 1):
                return False
        self._last_alert_time[key] = now
        self._alert_count[key] = count + 1
        return True

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

        if not self._allow(key, time.time()):
            return
        self._executor.submit(self._post_alert, violation, frame)

    def send_wet_floor_alert(self, wf_event, frame):
        if not self.enabled or not self.wet_floor_endpoint:
            return

        # One backoff key per physical spill, matched by overlap/proximity.
        # The old 50px centroid grid minted a fresh key whenever segmentation
        # jitter nudged the centroid across a cell line — each "new" spill
        # got an immediate first alert (3 alerts in seconds, observed live).
        key = self._wet_key(wf_event["bbox"])

        now = time.time()
        # First alert waits until the spill has persisted — the 5-frame streak
        # confirms it's real, this confirms it's not a transient glisten.
        first_seen = wf_event.get("first_seen_ts")
        if (first_seen is not None
                and now - first_seen < self.wet_floor_first_alert_seconds):
            return
        if not self._allow(key, now, base=self.wet_floor_cooldown_seconds):
            return
        self._executor.submit(self._post_wet_floor_alert, wf_event, frame)

    def _wet_key(self, bbox):
        """Stable backoff key for the spill this bbox belongs to."""
        for key, prev in self._wet_spills.items():
            if _same_spill(bbox, prev):
                self._wet_spills[key] = bbox   # follow the spill's drift
                return key
        key = f"wet_floor_{len(self._wet_spills)}"
        self._wet_spills[key] = bbox
        return key

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

        if not self._allow(key, time.time()):
            return
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

            # Every alert carries the frame — visual evidence lets a supervisor
            # verify the violation without trusting the detector — plus the
            # userId when the worker is identified.
            _, buffer = cv2.imencode(".jpg", frame)
            payload = {
                "missingItem": missing,
                "message": message,
                "imgImage": base64.b64encode(buffer).decode("utf-8"),
            }
            if worker_id is not None:
                payload["userId"] = str(worker_id)

            _log_outgoing(self.endpoint, payload)
            response = requests.post(self.endpoint, json=payload, timeout=10)
            _log_response(response)
            response.raise_for_status()
            logger.info("[ALERT SENT] %s", message)
        except Exception as e:
            logger.error("[ALERT FAILED] %s", e)

    def shutdown(self):
        self._executor.shutdown(wait=False)
