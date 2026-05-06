import torch
from ultralytics import YOLO


class PPEDetector:
    def __init__(self, model_path, confidence=0.35, required_ppe=None,
                 overlap_threshold=0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_half = self.device != "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.confidence = confidence
        self.required_ppe = required_ppe or {"helmet", "vest"}
        self.overlap_threshold = overlap_threshold
        self.person_class = "person"

    def detect(self, frame):
        detections = self.model.track(
            frame, conf=self.confidence, verbose=False,
            imgsz=480, persist=True, half=self.use_half
        )[0]

        persons = []
        ppe_items = []

        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])

            track_id = int(box.id[0]) if box.id is not None else None

            detection = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf,
                "class_name": cls_name,
                "track_id": track_id,
            }

            if cls_name == self.person_class:
                persons.append(detection)
            elif cls_name in self.required_ppe:
                ppe_items.append(detection)

        results_list = []
        for person in persons:
            detected_ppe = set()
            for ppe in ppe_items:
                if self._has_sufficient_overlap(ppe["bbox"], person["bbox"]):
                    detected_ppe.add(ppe["class_name"])

            missing_ppe = self.required_ppe - detected_ppe

            results_list.append({
                "person_bbox": person["bbox"],
                "track_id": person.get("track_id"),
                "compliant": len(missing_ppe) == 0,
                "detected_ppe": list(detected_ppe),
                "missing_ppe": list(missing_ppe),
            })

        return results_list

    def _has_sufficient_overlap(self, inner_box, outer_box):
        ix1, iy1, ix2, iy2 = inner_box
        ox1, oy1, ox2, oy2 = outer_box

        inter_x1 = max(ix1, ox1)
        inter_y1 = max(iy1, oy1)
        inter_x2 = min(ix2, ox2)
        inter_y2 = min(iy2, oy2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return False

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        inner_area = (ix2 - ix1) * (iy2 - iy1)

        if inner_area == 0:
            return False

        overlap_ratio = inter_area / inner_area
        return overlap_ratio >= self.overlap_threshold
