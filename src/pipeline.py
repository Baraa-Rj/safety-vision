import cv2
import time
import threading


class SafetyPipeline:
    def __init__(self, config, camera, ppe_detector, fall_detector,
                 worker_identifier, zone_monitor, renderer, event_logger,
                 alert_client=None):
        self.config = config
        self.camera = camera
        self.ppe_detector = ppe_detector
        self.fall_detector = fall_detector
        self.worker_id = worker_identifier
        self.zone_monitor = zone_monitor
        self.renderer = renderer
        self.event_logger = event_logger
        self.alert_client = alert_client
        self.running = False

        self._events_lock = threading.Lock()
        self._latest_events = {
            "ppe_violations": [],
            "compliant_workers": [],
            "falls": [],
            "zone_breaches": [],
            "timestamp": 0,
        }

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

        ppe_results = self.ppe_detector.detect(frame)

        if self.fall_detector:
            fall_results = self.fall_detector.detect(ppe_results, frame)
            fall_indices = {f["person_index"] for f in fall_results}
        else:
            fall_indices = set()

        for i, result in enumerate(ppe_results):
            x1, y1, x2, y2 = result["person_bbox"]
            person_crop = frame[y1:y2, x1:x2]
            qr_result = self.worker_id.identify(person_crop)
            worker_id = qr_result["worker_id"] if qr_result else None
            worker_name = qr_result["worker_name"] if qr_result else None
            qr_data = qr_result["qr_data"] if qr_result else None
            zone = self.zone_monitor.check_person(result["person_bbox"])
            track_id = result.get("track_id")

            if i in fall_indices:
                events["falls"].append({
                    "worker_id": worker_id,
                    "worker_name": worker_name,
                    "track_id": track_id,
                    "bbox": result["person_bbox"],
                })
            elif not result["compliant"]:
                events["ppe_violations"].append({
                    "worker_id": worker_id,
                    "worker_name": worker_name,
                    "qr_data": qr_data,
                    "track_id": track_id,
                    "bbox": result["person_bbox"],
                    "missing": result["missing_ppe"],
                    "detected": result["detected_ppe"],
                    "zone": zone,
                })
            else:
                events["compliant_workers"].append({
                    "worker_id": worker_id,
                    "worker_name": worker_name,
                    "track_id": track_id,
                    "bbox": result["person_bbox"],
                })

            if zone is not None:
                events["zone_breaches"].append({
                    "worker_id": worker_id,
                    "worker_name": worker_name,
                    "track_id": track_id,
                    "bbox": result["person_bbox"],
                    "zone_id": zone,
                })

        return events

    def _detection_loop(self):
        while self.running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.005)
                continue

            events = self.process_frame(frame)

            if self.alert_client:
                for violation in events["ppe_violations"]:
                    self.alert_client.send_ppe_alert(violation, frame)

            with self._events_lock:
                self._latest_events = events

            self.event_logger.log_events(events)

    def run(self, display=True):
        self.running = True
        time.sleep(self.config.camera.startup_delay)

        threading.Thread(target=self._detection_loop, daemon=True).start()

        print("Pipeline started. Press 'q' to stop.")

        while self.running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.005)
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
