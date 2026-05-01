"""Extract person crops from video for fall detection classifier training.

Usage:
    python scripts/extract_crops.py

Output:
    data/crops/unlabeled/  — person crops to sort into fallen/ or standing/

After running, manually sort the crops:
    data/crops/train/fallen/   — fallen person crops
    data/crops/train/standing/ — standing person crops
    data/crops/val/fallen/     — ~20% held out for validation
    data/crops/val/standing/
"""

import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.ppe_detector import PPEDetector
from config.settings import PipelineConfig

config = PipelineConfig()
detector = PPEDetector(config.ppe.model_path, confidence=config.ppe.confidence)

video_path = config.camera.source
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Cannot open {video_path}")
    sys.exit(1)

output_dir = "data/crops/unlabeled"
os.makedirs(output_dir, exist_ok=True)

frame_idx = 0
crop_idx = 0
sample_every = 5  # extract from every 5th frame to avoid near-duplicates

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Processing {video_path} ({total_frames} frames, sampling every {sample_every}th)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    if frame_idx % sample_every != 0:
        continue

    results = detector.detect(frame)
    for i, result in enumerate(results):
        x1, y1, x2, y2 = result["person_bbox"]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        filename = f"frame{frame_idx:05d}_p{i}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), crop)
        crop_idx += 1

    if frame_idx % 100 == 0:
        print(f"  Frame {frame_idx}/{total_frames} — {crop_idx} crops saved")

cap.release()
print(f"\nDone. {crop_idx} crops saved to {output_dir}/")
print(f"\nNext steps:")
print(f"  1. Create folders:  data/crops/train/fallen/  data/crops/train/standing/")
print(f"                      data/crops/val/fallen/    data/crops/val/standing/")
print(f"  2. Sort the crops from {output_dir}/ into the train/ folders")
print(f"  3. Move ~20% of each class into val/ folders")
print(f"  4. Run:  python scripts/train_classifier.py")
