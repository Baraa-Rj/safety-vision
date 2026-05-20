import logging
import threading
import time

import cv2

logger = logging.getLogger(__name__)


def _is_rtsp(src):
    return isinstance(src, str) and src.lower().startswith("rtsp://")


class CameraStream:
    def __init__(self, src, reconnect_after_seconds=3.0,
                 open_max_attempts=5, open_retry_backoff=2.0):
        self.src = src
        self._rtsp = _is_rtsp(src)
        self._reconnect_after = reconnect_after_seconds
        self._open_max_attempts = open_max_attempts
        self._open_retry_backoff = open_retry_backoff

        self.stream = self._open(raise_on_fail=True)
        self._lock = threading.Lock()
        self.latest_frame = None
        self.running = True

        fps = self.stream.get(cv2.CAP_PROP_FPS)
        # For RTSP, fps from VideoCapture is unreliable and we don't want to pace
        # the read loop ourselves anyway — let the camera push frames.
        self._frame_delay = 0.0 if self._rtsp else (1.0 / fps if fps > 0 else 1.0 / 30)

        self._thread = threading.Thread(target=self.update, daemon=True)
        self._thread.start()

    def _open(self, raise_on_fail=False):
        attempts = self._open_max_attempts if self._rtsp else 1
        last_err = None
        for i in range(attempts):
            stream = (
                cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                if self._rtsp else cv2.VideoCapture(self.src)
            )
            if stream.isOpened():
                if self._rtsp:
                    # Smallest buffer keeps us at the live edge instead of
                    # serving frames seconds behind real time.
                    stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if i > 0:
                    logger.info("Camera opened on attempt %d/%d", i + 1, attempts)
                return stream
            stream.release()
            last_err = f"VideoCapture.isOpened() returned False for {self.src}"
            if i + 1 < attempts:
                logger.warning(
                    "Camera open attempt %d/%d failed; retrying in %.1fs",
                    i + 1, attempts, self._open_retry_backoff,
                )
                time.sleep(self._open_retry_backoff)

        if raise_on_fail:
            raise RuntimeError(f"Cannot open video source: {self.src} ({last_err})")
        return None

    def read(self):
        with self._lock:
            return self.latest_frame

    def stop(self):
        self.running = False
        self._thread.join(timeout=2.0)
        if self.stream is not None:
            self.stream.release()

    def update(self):
        last_good = time.monotonic()
        while self.running:
            ret, frame = False, None
            if self.stream is not None:
                ret, frame = self.stream.read()

            if ret:
                with self._lock:
                    self.latest_frame = frame
                last_good = time.monotonic()
                if self._frame_delay > 0:
                    time.sleep(self._frame_delay)
                continue

            # For RTSP, treat a stretch of failed reads as a dropped connection
            # and rebuild the capture. For files, just keep looping until EOF
            # behavior settles (preserves prior behavior).
            if self._rtsp and time.monotonic() - last_good > self._reconnect_after:
                logger.warning("No RTSP frames for %.1fs; reconnecting...",
                               time.monotonic() - last_good)
                try:
                    if self.stream is not None:
                        self.stream.release()
                except Exception:
                    pass
                self.stream = self._open(raise_on_fail=False)
                last_good = time.monotonic()
                if self.stream is None:
                    # _open already slept through its backoff; loop and retry.
                    continue
            else:
                time.sleep(0.01)
