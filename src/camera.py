import cv2
import threading


class CameraStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.latest_frame = None
        self.running = True
        self._thread = threading.Thread(target=self.update, daemon=True)
        self._thread.start()

    def read(self):
        return self.latest_frame

    def stop(self):
        self.running = False
        self._thread.join(timeout=2.0)
        self.stream.release()
    def update(self):
        while self.running:
            ret, frame = self.stream.read() 
            if ret:
                self.latest_frame = frame
