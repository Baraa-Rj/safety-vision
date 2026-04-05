import cv2
import threading


class CameraStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.latest_frame = None
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()
    def read(self):
        return self.latest_frame 
    def stop(self):
        self.stream.release()
        self.running = False
    def update(self):
        while self.running:
            ret, frame = self.stream.read() 
            if ret:
                self.latest_frame = frame
