"""Fine-tune the PPE model on Kaggle GPU with vest-color augmentation.

Run this in a Kaggle notebook after uploading two Kaggle Datasets:
  1. the image dataset  -> /kaggle/input/ppe-dataset/ppe_dataset/{train,valid,test}/{images,labels}
  2. the base weights   -> /kaggle/input/ppe-weights/best.pt

Then, in a notebook cell:
    !pip install -q ultralytics
    !DATASET_DIR=/kaggle/input/ppe-dataset/ppe_dataset \
     MODEL_PATH=/kaggle/input/ppe-weights/best.pt \
     python train_ppe_kaggle.py

Override any path via env var. Runs locally too — device auto-detects CPU/GPU.

Why this exists separately from train_ppe.py:
  - GPU device + larger batch (Kaggle has a T4/P100, the local script was CPU).
  - Regenerates dataset.yaml because the uploaded one carries a stale absolute
    `path:` that won't exist on Kaggle.
  - Widens hue augmentation (hsv_h) so the `vest` class generalizes across
    orange AND lime instead of memorizing one color. The Ultralytics default
    hsv_h=0.015 (~5 deg hue) is far too narrow to span orange -> lime.
"""

import os

import torch
from ultralytics import YOLO

DATASET_DIR = os.environ.get("DATASET_DIR", "/kaggle/input/ppe-dataset/ppe_dataset")
MODEL_PATH = os.environ.get("MODEL_PATH", "/kaggle/input/ppe-weights/best.pt")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/kaggle/working/runs")

# Ultralytics needs a data yaml whose `path` matches where the images actually
# live. Regenerate it so the same script works on Kaggle and locally.
os.makedirs(OUTPUT_DIR, exist_ok=True)
yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
with open(yaml_path, "w") as f:
    f.write(
        f"path: {DATASET_DIR}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 3\n"
        "names: ['helmet', 'person', 'vest']\n"
    )

device = 0 if torch.cuda.is_available() else "cpu"
print(f"device={device} | dataset={DATASET_DIR} | weights={MODEL_PATH}")

model = YOLO(MODEL_PATH)
model.train(
    data=yaml_path,
    epochs=100,
    imgsz=640,
    batch=16,                 # GPU headroom; drop to 8 if you hit OOM
    patience=20,
    lr0=0.001,
    lrf=0.01,
    device=device,
    workers=2,
    project=OUTPUT_DIR,
    name="ppe_finetune",
    exist_ok=True,
    pretrained=True,
    plots=True,
    cache=True,
    # --- vest color-invariance ---
    # hsv_h=0.1 -> ~+/-18 OpenCV hue units (~+/-36 deg), enough to bridge
    # orange and lime. Real lime samples still do the heavy lifting; this just
    # stops the net from overfitting either color. NOTE: this also recolors
    # helmets/persons — confirm those classes don't regress on the val set.
    hsv_h=0.1,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
)

print("Done. Best weights:", os.path.join(OUTPUT_DIR, "ppe_finetune", "weights", "best.pt"))
print("Download it from /kaggle/working and copy to models/best.pt")
