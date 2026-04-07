import cv2
import time
from src.camera import CameraStream
from src.ppe_detector import PPEDetector
from src.fall_detector import FallDetector
from src.worker_id import WorkerIdentifier
from src.zone_monitor import ZoneMonitor


class SafetyPipeline:
    def __init__(self, source, ppe_model_path, pose_model_path):
        self.camera = CameraStream(source)
        self.ppe_detector = PPEDetector(ppe_model_path)
        self.fall_detector = FallDetector(pose_model_path)
        self.worker_id = WorkerIdentifier()
        self.zone_monitor = ZoneMonitor()
        self.running = False

    def add_zone(self, zone_id, points):
        self.zone_monitor.add_zone(zone_id, points)

    def process_frame(self, frame):
        """
        Run all detections on a single frame.
        Returns a dict with all safety events.
        """
        events = {
            "ppe_violations": [],
            "falls": [],
            "zone_breaches": [],
            "timestamp": time.time(),
        }

        # 1. PPE detection
        ppe_results = self.ppe_detector.detect(frame)
        for result in ppe_results:
            # Crop person for ArUco ID
            x1, y1, x2, y2 = result["person_bbox"]
            person_crop = frame[y1:y2, x1:x2]
            worker_id = self.worker_id.identify(person_crop)

            # Zone check
            zone = self.zone_monitor.check_person(result["person_bbox"])

            if not result["compliant"]:
                events["ppe_violations"].append({
                    "worker_id": worker_id,
                    "bbox": result["person_bbox"],
                    "missing": result["missing_ppe"],
                    "detected": result["detected_ppe"],
                    "zone": zone,
                })

            if zone is not None:
                events["zone_breaches"].append({
                    "worker_id": worker_id,
                    "bbox": result["person_bbox"],
                    "zone_id": zone,
                })

        # 2. Fall detection
        falls = self.fall_detector.detect(frame)
        for fall in falls:
            x1, y1, x2, y2 = fall["bbox"]
            person_crop = frame[y1:y2, x1:x2]
            worker_id = self.worker_id.identify(person_crop)

            events["falls"].append({
                "worker_id": worker_id,
                "person_id": fall["person_id"],
                "bbox": fall["bbox"],
            })

        return events

    def run(self, display=True):
        """
        Main loop. Reads frames and processes them continuously.
        """
        self.running = True
        time.sleep(1)  # let camera thread start

        print("Pipeline started. Press 'q' to stop.")

        while self.running:
            frame = self.camera.read()
            if frame is None:
                continue

            events = self.process_frame(frame)

            # Log events
            if events["ppe_violations"]:
                for v in events["ppe_violations"]:
                    wid = v["worker_id"] or "UNIDENTIFIED"
                    print(f"[PPE VIOLATION] Worker {wid} | Missing: {v['missing']}")

            if events["falls"]:
                for f in events["falls"]:
                    wid = f["worker_id"] or "UNIDENTIFIED"
                    print(f"[FALL DETECTED] Worker {wid}")

            if events["zone_breaches"]:
                for z in events["zone_breaches"]:
                    wid = z["worker_id"] or "UNIDENTIFIED"
                    print(f"[ZONE BREACH] Worker {wid} in zone {z['zone_id']}")

            # Display
            if display:
                annotated = self._draw(frame, events)
                h, w = annotated.shape[:2]
                max_w = 960
                if w > max_w:
                    scale = max_w / w
                    annotated = cv2.resize(annotated, (max_w, int(h * scale)))
                cv2.imshow("Safety Vision", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.stop()

    def _draw(self, frame, events):
        """
        Draw bounding boxes and labels on the frame.
        Green = compliant, Red = violation/fall/breach.
        """
        display = frame.copy()

        for v in events["ppe_violations"]:
            x1, y1, x2, y2 = v["bbox"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"VIOLATION: {', '.join(v['missing'])}"
            wid = v["worker_id"] or "?"
            cv2.putText(display, f"W:{wid} {label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for f in events["falls"]:
            x1, y1, x2, y2 = f["bbox"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(display, "FALL DETECTED", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

        for z in events["zone_breaches"]:
            x1, y1, x2, y2 = z["bbox"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(display, f"ZONE: {z['zone_id']}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # Draw zone polygons
        for zone_id, polygon in self.zone_monitor.zones.items():
            cv2.polylines(display, [polygon], True, (0, 165, 255), 2)
            cv2.putText(display, zone_id, tuple(polygon[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        return display

    def stop(self):
        self.running = False
        self.camera.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped.")