"""Fine-tune the PPE detection model on the annotated dataset.

Usage:
    PYTHONPATH=. python3 scripts/train_ppe.py

Loads the existing models/best.pt weights and fine-tunes on
data/ppe_dataset/ for 100 epochs.  Results are saved under runs/detect/.
"""

from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
DATASET_YAML = "data/ppe_dataset/dataset.yaml"
EPOCHS = 100
IMGSZ = 640
BATCH = 8

model = YOLO(MODEL_PATH)

model.train(
    data=DATASET_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    patience=20,
    lr0=0.001,
    lrf=0.01,
    device="cpu",
    workers=2,
    project="runs/detect",
    name="ppe_finetune",
    exist_ok=True,
    pretrained=True,
    plots=True,
)

print("\nTraining complete.")
print("Best weights: runs/detect/ppe_finetune/weights/best.pt")
print("\nTo use the new model, copy it:")
print("  cp runs/detect/ppe_finetune/weights/best.pt models/best.pt")
