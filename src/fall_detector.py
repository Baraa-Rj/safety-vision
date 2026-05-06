import os

import torch
from ultralytics import YOLO


class FallDetector:
    def __init__(self, classifier_model_path, confidence=0.7):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_half = self.device != "cpu"
        if os.path.exists(classifier_model_path):
            self.model = YOLO(classifier_model_path)
            self.model.to(self.device)
        else:
            self.model = None
        self.confidence = confidence

    def detect(self, person_bboxes, frame):
        """Classify each detected person as fallen or standing.

        Args:
            person_bboxes: list of dicts with 'person_bbox' [x1,y1,x2,y2]
                           from PPE detector results.
            frame: the full frame to crop persons from.
        Returns:
            list of fall dicts with 'person_index' and 'bbox'.
        """
        falls = []

        if self.model is None:
            return falls

        for i, person in enumerate(person_bboxes):
            x1, y1, x2, y2 = person["person_bbox"]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            results = self.model(crop, verbose=False, imgsz=128, half=self.use_half)[0]
            probs = results.probs

            class_idx = int(probs.top1)
            class_name = results.names[class_idx]
            conf = float(probs.top1conf)

            if class_name == "fallen" and conf >= self.confidence:
                falls.append({
                    "person_index": i,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })

        return falls
