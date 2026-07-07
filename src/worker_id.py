import cv2
import time


class WorkerIdentifier:
    # cache_ttl is the identity's survival window while its track is NOT being
    # seen (occlusion, detection dropout, worker briefly off-frame). While the
    # track is visible every identify() hit refreshes the timestamp, so a
    # present worker never expires. Generous by design: BoT-SORT never reuses
    # a track id within a run, so a lingering identity can't attach to the
    # wrong person — it's only re-adopted via the same id resurfacing or an
    # explicit continuity transfer().
    def __init__(self, cache_ttl=300.0, max_failed_attempts=450, min_decode_size=400):
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

    def get_cached(self, track_id):
        """Return a track's cached identity (no crop/decode), or None.

        Lets a fallen worker keep the ID they were scanned with while upright,
        even though their QR can't be read on the ground.
        """
        if track_id is None:
            return None
        cached = self._cache.get(track_id)
        if cached is not None and (time.time() - cached["timestamp"]) < self._cache_ttl:
            return cached["result"]
        return None

    def transfer(self, old_track_id, new_track_id):
        """Move a cached identity to a new track id.

        The tracker sometimes drops a moving worker's track and re-acquires
        them under a fresh id (churn). The pipeline detects that spatially and
        calls this so the QR identity follows the person instead of dying with
        the old id. Refreshes the timestamp — the worker is visibly present.
        Returns the moved identity result, or None if there was nothing cached.
        """
        cached = self._cache.pop(old_track_id, None)
        self._fail_count.pop(old_track_id, None)
        if cached is None:
            return None
        cached["timestamp"] = time.time()
        self._cache[new_track_id] = cached
        return cached["result"]

    def clear_stale(self, active_track_ids):
        """Prune identities whose cache entry has outlived the TTL.

        Deliberately NOT keyed on the active set: a single missed person
        detection (blur, occlusion, pose change) drops a track from one
        frame's results, and deleting the identity there forced a fresh QR
        scan every time detection blipped. Identities survive absence for
        cache_ttl; track ids are never reused within a run, so a lost track's
        identity can only ever return to the same worker. The active set is
        still used to drop decode-failure counters for gone tracks.
        """
        now = time.time()
        stale = [
            tid for tid, c in self._cache.items()
            if (now - c["timestamp"]) >= self._cache_ttl
        ]
        for tid in stale:
            del self._cache[tid]
        active = set(active_track_ids)
        for tid in list(self._fail_count.keys()):
            if tid not in active:
                del self._fail_count[tid]
