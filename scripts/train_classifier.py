"""Train a fall detection classifier using YOLO classification mode.

Expects this folder structure:
    data/crops/train/fallen/   — fallen person crops
    data/crops/train/standing/ — standing person crops
    data/crops/val/fallen/
    data/crops/val/standing/

Usage:
    python scripts/train_classifier.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ultralytics import YOLO

data_path = os.path.abspath("data/crops")

# Verify folder structure
for split in ["train", "val"]:
    for cls in ["fallen", "standing"]:
        folder = os.path.join(data_path, split, cls)
        if not os.path.isdir(folder):
            print(f"Missing folder: {folder}")
            sys.exit(1)
        count = len([f for f in os.listdir(folder) if f.endswith((".jpg", ".png"))])
        print(f"  {split}/{cls}: {count} images")

print()

# Train classifier — start from a small pretrained YOLO classification model
model = YOLO("yolo11n-cls.pt")
model.train(
    data=data_path,
    epochs=30,
    imgsz=128,
    batch=32,
    project="models",
    name="fall_classifier",
    exist_ok=True,
)

print(f"\nModel saved to models/fall_classifier/weights/best.pt")
print(f"Update config/settings.py with the model path.")
