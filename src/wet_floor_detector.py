
import logging
import os

import torch

logger = logging.getLogger(__name__)


class WetFloorDetector:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_half = self.device != "cpu"
        self.model = None
        self._missing_model_warned = False

    @property
    def is_loaded(self):
        return self.model is not None

    def _ensure_model(self):
        if self.model is not None:
            return True
        if not os.path.exists(self.config.model_path):
            if not self._missing_model_warned:
                logger.warning(
                    "Wet floor model not found at %s; detector will return no detections.",
                    self.config.model_path,
                )
                self._missing_model_warned = True
            return False

        # Imported lazily so tests can instantiate without ultralytics overhead.
        from ultralytics import YOLO

        self.model = YOLO(self.config.model_path)
        self.model.to(self.device)
        return True

    def detect(self, frame):
        if not self._ensure_model():
            return []

        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            verbose=False,
            imgsz=640,
            half=self.use_half,
        )[0]

        h, w = frame.shape[:2]
        frame_area = float(h * w) if h and w else 0.0

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
            area_pct = (area / frame_area * 100.0) if frame_area > 0 else 0.0

            if area_pct < self.config.min_area_pct:
                continue

            detections.append({
                "bbox": bbox,
                "confidence": conf,
                "area_pct": area_pct,
            })

        return detections
