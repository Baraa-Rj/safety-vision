import cv2
import time


class WorkerIdentifier:
    def __init__(self, cache_ttl=30.0, max_failed_attempts=450):
        self.detector = cv2.QRCodeDetector()
        self._cache = {}
        self._fail_count = {}
        self._cache_ttl = cache_ttl
        self._max_failed_attempts = max_failed_attempts

    def identify(self, person_crop, track_id=None):
        if person_crop is None or person_crop.size == 0:
            return None

        now = time.time()

        if track_id is not None:
            cached = self._cache.get(track_id)
            if cached is not None and (now - cached["timestamp"]) < self._cache_ttl:
                return cached["result"]

            if self._fail_count.get(track_id, 0) >= self._max_failed_attempts:
                return None

        data, _, _ = self.detector.detectAndDecode(person_crop)
        if not data:
            if track_id is not None:
                self._fail_count[track_id] = self._fail_count.get(track_id, 0) + 1
            return None

        result = {
            "qr_data": data,
            "worker_name": None,
            "worker_id": data,
        }

        if track_id is not None:
            self._cache[track_id] = {"result": result, "timestamp": now}
            self._fail_count.pop(track_id, None)

        return result

    def clear_stale(self, active_track_ids):
        stale = set(self._cache.keys()) - set(active_track_ids)
        for tid in stale:
            del self._cache[tid]
            self._fail_count.pop(tid, None)
