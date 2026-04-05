from ultralytics import YOLO
from collections import defaultdict


class FallDetector:
    def __init__(self, pose_model_path, window_size=15):
        self.model = YOLO(pose_model_path)
        self.history = defaultdict(list)
        self.window_size = window_size

    def detect(self, frame):
        """
        Run pose estimation and check for falls.
        Returns list of fall events.
        """
        results = self.model.track(frame, persist=True, verbose=False)[0]
        falls = []

        if results.boxes.id is None:
            return falls

        for i, box in enumerate(results.boxes):
            person_id = int(box.id[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            bbox_w = x2 - x1
            bbox_h = y2 - y1
            aspect_ratio = bbox_h / bbox_w if bbox_w > 0 else 1.0

            # Keypoint indices 11 (left hip) and 12 (right hip) in COCO format
            kps = results.keypoints.xy[i]
            left_hip_y  = float(kps[11][1])
            right_hip_y = float(kps[12][1])
            hip_y = (left_hip_y + right_hip_y) / 2

            self.history[person_id].append({
                "aspect_ratio": aspect_ratio,
                "hip_y": hip_y,
                "bbox_h": bbox_h,
            })

            if len(self.history[person_id]) > self.window_size:
                self.history[person_id].pop(0)

            if self._is_fall(person_id):
                falls.append({
                    "person_id": person_id,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })

        return falls

    def _is_fall(self, person_id):
        """
        Check if person's history indicates a fall.
        Rules:
          - bbox aspect ratio went from >1 (tall) to <1 (wide)
          - hip height dropped rapidly
        """
        history = self.history[person_id]
        if len(history) < self.window_size // 2:
            return False

        mid = len(history) // 2
        early, recent = history[:mid], history[mid:]

        was_tall = any(e["aspect_ratio"] > 1.0 for e in early)
        is_wide  = recent[-1]["aspect_ratio"] < 1.0

        # Hip drop: y increases downward in image coordinates
        early_hip_y  = sum(e["hip_y"] for e in early) / len(early)
        recent_hip_y = sum(e["hip_y"] for e in recent) / len(recent)
        avg_bbox_h   = sum(e["bbox_h"] for e in history) / len(history)
        hip_dropped  = (recent_hip_y - early_hip_y) > 0.25 * avg_bbox_h

        return was_tall and is_wide and hip_dropped