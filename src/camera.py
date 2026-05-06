import cv2
import time
import threading


class CameraStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise RuntimeError(f"Cannot open video source: {src}")
        self._lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        self._frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30
        self._thread = threading.Thread(target=self.update, daemon=True)
        self._thread.start()

    def read(self):
        with self._lock:
            return self.latest_frame

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
            else:
                time.sleep(0.001)
            time.sleep(self._frame_delay)
