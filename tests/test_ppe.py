from src.camera import CameraStream
from src.ppe_detector import PPEDetector
import time

cam = CameraStream("data/sample_videos/output.mp4")
detector = PPEDetector("models/best.pt")
time.sleep(3)

frame = cam.read()
if frame is not None:
    results = detector.detect(frame)
    for r in results:
        status = "COMPLIANT" if r["compliant"] else "VIOLATION"
        print(f"{status} | PPE: {r['detected_ppe']} | Missing: {r['missing_ppe']}")

cam.stop()
