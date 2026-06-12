import torch
from ultralytics import YOLO


class PPEDetector:
    def __init__(self, model_path, confidence=0.35, required_ppe=None,
                 overlap_threshold=0.5, class_confidences=None,
                 tracker_config="botsort.yaml"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_half = self.device != "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.confidence = confidence
        self.class_confidences = class_confidences or {}
        self.required_ppe = required_ppe or {"helmet", "vest"}
        self.overlap_threshold = overlap_threshold
        self.tracker_config = tracker_config
        self.person_class = "person"
        self.fallen_class = "fallen"
        # Latest 'fallen' detections, refreshed each detect(). The FallDetector
        # reads these so best.pt runs only once per frame.
        self.fallen_detections = []

        # YOLO drops boxes below `conf` before they reach us, so run inference
        # at the floor of all per-class thresholds and filter in Python below.
        if self.class_confidences:
            self._inference_conf = min(self.confidence, min(self.class_confidences.values()))
        else:
            self._inference_conf = self.confidence

    def _class_threshold(self, cls_name):
        return self.class_confidences.get(cls_name, self.confidence)

    def detect(self, frame):
        detections = self.model.track(
            frame, conf=self._inference_conf, verbose=False,
            imgsz=480, persist=True, half=self.use_half,
            tracker=self.tracker_config,
        )[0]

        persons = []
        ppe_items = []
        fallen = []

        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])

            if conf < self._class_threshold(cls_name):
                continue

            track_id = int(box.id[0]) if box.id is not None else None

            detection = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf,
                "class_name": cls_name,
                "track_id": track_id,
            }

            if cls_name == self.person_class:
                persons.append(detection)
            elif cls_name == self.fallen_class:
                fallen.append(detection)
            elif cls_name in self.required_ppe:
                ppe_items.append(detection)

        # A worker on the ground is detected as 'fallen'. If the model also emits
        # an overlapping 'person' box for the same body, drop it so the fall isn't
        # double-counted as a standing PPE subject.
        if fallen:
            persons = [
                p for p in persons
                if not any(self._has_sufficient_overlap(p["bbox"], f["bbox"])
                           for f in fallen)
            ]

        self.fallen_detections = [
            {"bbox": f["bbox"], "track_id": f.get("track_id"),
             "confidence": f["confidence"]}
            for f in fallen
        ]

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
