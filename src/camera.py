import cv2
import time
import threading
from collections import deque


class CameraStream:
    def __init__(self, src, buffer_size=64):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise RuntimeError(f"Cannot open video source: {src}")
        self._lock = threading.Lock()
        self.latest_frame = None
        self._frame_buffer = deque(maxlen=buffer_size)
        self.running = True
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        self._frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30
        self._thread = threading.Thread(target=self.update, daemon=True)
        self._thread.start()

    def read(self):
        """Return the latest frame (for display loop)."""
        with self._lock:
            return self.latest_frame

    def read_next(self):
        """Return the next sequential frame from the buffer (for detection loop)."""
        with self._lock:
            if self._frame_buffer:
                return self._frame_buffer.popleft()
            return None

    def stop(self):
        self.running = False
        self._thread.join(timeout=2.0)
        self.stream.release()

    def update(self):
        while self.running:
            ret, frame = self.stream.read()
            if ret:
                with self._lock:
                    self.latest_frame = frame
                    self._frame_buffer.append(frame)
            else:
                time.sleep(0.001)
            time.sleep(self._frame_delay)
