"""Train the fall classifier (fallen vs standing) on Kaggle GPU.

Expects a YOLO classification folder layout:
    <crops>/train/fallen/   <crops>/train/standing/
    <crops>/val/fallen/     <crops>/val/standing/

Upload that folder as a Kaggle Dataset, attach it, enable GPU + Internet, then:
    !pip install -q ultralytics
    !python train_classifier_kaggle.py

The crops dir is auto-located under /kaggle/input (any dataset containing a
train/fallen folder). Override with CROPS_DIR. Runs locally too.

Output: /kaggle/working/runs/fall_classifier/weights/best.pt
"""

import glob
import os

import torch
from ultralytics import YOLO


def _find_crops_dir():
    env = os.environ.get("CROPS_DIR")
    if env and os.path.isdir(os.path.join(env, "train", "fallen")):
        return env
    hits = glob.glob("/kaggle/input/**/train/fallen", recursive=True)
    if hits:
        return os.path.dirname(os.path.dirname(hits[0]))  # .../train/fallen -> ...
    # local fallback
    if os.path.isdir("data/crops/train/fallen"):
        return os.path.abspath("data/crops")
    raise FileNotFoundError(
        "No 'train/fallen' folder found. Upload/attach the crops dataset, "
        "or set CROPS_DIR."
    )


CROPS_DIR = _find_crops_dir()
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/kaggle/working/runs"
                            if os.path.isdir("/kaggle/working") else "runs")

# Report class balance so a lopsided dataset is obvious before training.
for split in ("train", "val"):
    for cls in ("fallen", "standing"):
        d = os.path.join(CROPS_DIR, split, cls)
        n = len(glob.glob(os.path.join(d, "*.jpg"))) + len(glob.glob(os.path.join(d, "*.png")))
        print(f"  {split}/{cls}: {n} images")

device = 0 if torch.cuda.is_available() else "cpu"
print(f"\ndevice={device} | crops={CROPS_DIR}")

# Small pretrained classification backbone (YOLO26 nano); imgsz 224 is more
# robust than 128. Ultralytics auto-downloads this (needs Internet on Kaggle).
model = YOLO("yolo26n-cls.pt")
model.train(
    data=CROPS_DIR,
    epochs=40,
    imgsz=224,
    batch=64,
    device=device,
    project=OUTPUT_DIR,
    name="fall_classifier",
    exist_ok=True,
    # Augmentation: falls vary in orientation/lighting; flips + HSV help.
    fliplr=0.5,
    hsv_h=0.02,
    hsv_s=0.6,
    hsv_v=0.4,
)

print("\nDone. Best:", os.path.join(OUTPUT_DIR, "fall_classifier", "weights", "best.pt"))
