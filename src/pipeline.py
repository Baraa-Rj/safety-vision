import cv2
import time
import threading


class SafetyPipeline:
    def __init__(self, config, camera, ppe_detector, fall_detector,
                 worker_identifier, zone_monitor, renderer, event_logger):
        self.config = config
        self.camera = camera
        self.ppe_detector = ppe_detector
        self.fall_detector = fall_detector
        self.worker_id = worker_identifier
        self.zone_monitor = zone_monitor
        self.renderer = renderer
        self.event_logger = event_logger
        self.running = False
        self._fall_frame_counter = 0
        self._fall_detection_interval = config.fall.detection_interval
        self._cached_falls = []
        self._latest_events = {
            "ppe_violations": [],
            "compliant_workers": [],
            "falls": [],
            "zone_breaches": [],
            "timestamp": 0,
        }
        self._events_lock = threading.Lock()

    def add_zone(self, zone_id, points):
        self.zone_monitor.add_zone(zone_id, points)

    def process_frame(self, frame):
        events = {
            "ppe_violations": [],
            "compliant_workers": [],
            "falls": [],
            "zone_breaches": [],
            "timestamp": time.time(),
        }

        # 1. PPE detection
        ppe_results = self.ppe_detector.detect(frame)
        for result in ppe_results:
            x1, y1, x2, y2 = result["person_bbox"]
            person_crop = frame[y1:y2, x1:x2]
            worker_id = self.worker_id.identify(person_crop)
            zone = self.zone_monitor.check_person(result["person_bbox"])

            if not result["compliant"]:
                events["ppe_violations"].append({
                    "worker_id": worker_id,
                    "bbox": result["person_bbox"],
                    "missing": result["missing_ppe"],
                    "detected": result["detected_ppe"],
                    "zone": zone,
                })
            else:
                events["compliant_workers"].append({
                    "worker_id": worker_id,
                    "bbox": result["person_bbox"],
                })

            if zone is not None:
                events["zone_breaches"].append({
                    "worker_id": worker_id,
                    "bbox": result["person_bbox"],
                    "zone_id": zone,
                })

        # 2. Fall detection (every Nth frame)
        self._fall_frame_counter += 1
        if self._fall_frame_counter % self._fall_detection_interval == 0:
            self._cached_falls = self.fall_detector.detect(frame)
        for fall in self._cached_falls:
            x1, y1, x2, y2 = fall["bbox"]
            person_crop = frame[y1:y2, x1:x2]
            worker_id = self.worker_id.identify(person_crop)
            events["falls"].append({
                "worker_id": worker_id,
                "person_id": fall["person_id"],
                "bbox": fall["bbox"],
            })

        return events

    def _detection_loop(self):
        while self.running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.005)
                continue

            events = self.process_frame(frame)

            with self._events_lock:
                self._latest_events = events

            self.event_logger.log_events(events)

    def run(self, display=True):
        self.running = True
        time.sleep(self.config.camera.startup_delay)

        self._det_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._det_thread.start()

        print("Pipeline started. Press 'q' to stop.")

        while self.running:
            frame = self.camera.read()
            if frame is None:
                continue

            with self._events_lock:
                events = self._latest_events

            if display:
                quit_requested = self.renderer.render_to_window(frame, events)
                if quit_requested:
                    break

        self.stop()

    def stop(self):
        self.running = False
        self.camera.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped.")
