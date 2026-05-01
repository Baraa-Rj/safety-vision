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

    def _post_alert(self, violation, frame):
        try:
            _, buffer = cv2.imencode(".jpg", frame)
            img_b64 = base64.b64encode(buffer).decode("utf-8")

            missing = ", ".join(violation["missing"])
            worker_id = violation.get("worker_id")
            worker_name = violation.get("worker_name")
            qr_data = violation.get("qr_data")

            if worker_name:
                message = f"Worker {worker_name} missing {missing}"
            elif worker_id is not None:
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
            if qr_data is not None:
                payload["qrData"] = qr_data
            if worker_name is not None:
                payload["workerName"] = worker_name

            response = requests.post(self.endpoint, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("[ALERT SENT] %s", message)
        except Exception as e:
            logger.error("[ALERT FAILED] %s", e)

    def shutdown(self):
        self._executor.shutdown(wait=False)
