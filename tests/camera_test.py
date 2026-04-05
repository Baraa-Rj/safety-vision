# test_camera.py — put this in tests/
from src.camera import CameraStream
import cv2
import time

cam = CameraStream("data/output.mp4")
time.sleep(3)  # let the thread grab a few frames (4K needs more time)

frame = cam.read()
if frame is not None:
    print(f"Frame shape: {frame.shape}")
    cv2.imwrite("tests/test_frame.jpg", frame)
    print("Saved test_frame.jpg")
else:
    print("No frame received")

cam.stop()