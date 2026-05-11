"""Standalone QR detection test — no YOLO, no pipeline.

Reads video frame by frame, scans the full frame for QR codes,
and displays the result on screen.

Usage:
    PYTHONPATH=. python3 scripts/test_qr.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
from config.settings import PipelineConfig

config = PipelineConfig()
cap = cv2.VideoCapture(config.camera.source)

if not cap.isOpened():
    print(f"Cannot open {config.camera.source}")
    sys.exit(1)

detector = cv2.QRCodeDetector()
fps = cap.get(cv2.CAP_PROP_FPS) or 25
last_qr = None

print("QR test started. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    data, points, _ = detector.detectAndDecode(frame)

    if data:
        last_qr = data
        print(f"[QR DETECTED] {data}")

    # Draw QR bounding box if detected
    if points is not None and len(points) > 0:
        pts = points[0].astype(int)
        for i in range(len(pts)):
            cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 255, 0), 3)
        cv2.putText(frame, f"QR: {data}", (pts[0][0], pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show last known QR
    label = f"Last QR: {last_qr}" if last_qr else "No QR detected yet"
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Resize for display
    h, w = frame.shape[:2]
    if w > 960:
        scale = 960 / w
        frame = cv2.resize(frame, (960, int(h * scale)))

    cv2.imshow("QR Test", frame)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal result: {last_qr or 'No QR detected'}")
