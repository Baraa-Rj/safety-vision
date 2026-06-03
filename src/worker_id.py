import cv2
import time


class WorkerIdentifier:
    def __init__(self, cache_ttl=30.0, max_failed_attempts=450, min_decode_size=400):
        self.detector = cv2.QRCodeDetector()
        self._cache = {}
        self._fail_count = {}
        self._cache_ttl = cache_ttl
        self._max_failed_attempts = max_failed_attempts
        # Upscale crops whose smaller side is below this before decoding — a QR
        # needs enough pixels per module, and small/low-res crops (e.g. the RTSP
        # substream) otherwise fail to decode.
        self._min_decode_size = min_decode_size

    def _decode(self, crop):
        """Robustly read a QR from a person crop.

        Uses detectAndDecodeMulti (markedly more reliable than detectAndDecode,
        which fails even on clean codes) and retries on an upscaled copy when the
        crop is small. Returns the QR string, or None.
        """
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return None

        variants = [crop]
        if min(h, w) < self._min_decode_size:
            s = self._min_decode_size / float(min(h, w))
            variants.append(cv2.resize(
                crop, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC,
            ))

        for img in variants:
            try:
                ok, infos, _, _ = self.detector.detectAndDecodeMulti(img)
            except cv2.error:
                ok, infos = False, ()
            if ok:
                for d in infos:
                    if d:
                        return d
        return None

    def identify(self, person_crop, track_id=None):
        if person_crop is None or person_crop.size == 0:
            return None

        now = time.time()

        if track_id is not None:
            cached = self._cache.get(track_id)
            if cached is not None and (now - cached["timestamp"]) < self._cache_ttl:
                # Refresh on hit so a worker's identity stays attached for the
                # whole life of the track — once scanned, we don't force a
                # re-read just because the QR went out of view for a while.
                cached["timestamp"] = now
                return cached["result"]

            if self._fail_count.get(track_id, 0) >= self._max_failed_attempts:
                return None

        data = self._decode(person_crop)
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
