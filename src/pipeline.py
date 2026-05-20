import cv2
import logging
import time
import threading
from collections import deque

from src.ppe_compliance_tracker import PPEComplianceTracker

logger = logging.getLogger(__name__)

_COMPLIANCE_CLEANUP_EVERY = 30  # frames; ~1s at 30 FPS


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class SafetyPipeline:
    def __init__(self, config, camera, ppe_detector, fall_detector,
                 worker_identifier, zone_monitor, renderer, event_logger,
                 alert_client=None, wet_floor_detector=None):
        self.config = config
        self.camera = camera
        self.ppe_detector = ppe_detector
        self.fall_detector = fall_detector
        self.wet_floor_detector = wet_floor_detector
        self.worker_id = worker_identifier
        self.zone_monitor = zone_monitor
        self.renderer = renderer
        self.event_logger = event_logger
        self.alert_client = alert_client
        self.running = False

        self.required_items = set(config.ppe.required_ppe)
        self.compliance_tracker = PPEComplianceTracker(
            config.compliance, self.required_items,
        )
        self._compliance_cleanup_counter = 0

        wf_required = (
            wet_floor_detector.config.consecutive_frames_required
            if wet_floor_detector is not None else 5
        )
        self._wet_floor_history = deque(maxlen=wf_required)
        self._wet_floor_first_seen_ts = None

        self._events_lock = threading.Lock()
        self._latest_events = {
            "ppe_violations": [],
            "confirmed_ppe_violations": [],
            "compliant_workers": [],
            "falls": [],
            "zone_breaches": [],
            "wet_floor_events": [],
            "timestamp": 0,
        }

    def add_zone(self, zone_id, points):
        self.zone_monitor.add_zone(zone_id, points)

    def process_frame(self, frame):
        events = {
            "ppe_violations": [],
            "confirmed_ppe_violations": [],
            "compliant_workers": [],
            "falls": [],
            "zone_breaches": [],
            "wet_floor_events": [],
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
            track_id = result.get("track_id")
            person_crop = frame[y1:y2, x1:x2]
            qr_result = self.worker_id.identify(person_crop, track_id=track_id)
            worker_id = qr_result["worker_id"] if qr_result else None
            worker_name = qr_result["worker_name"] if qr_result else None
            qr_data = qr_result["qr_data"] if qr_result else None
            zone = self.zone_monitor.check_person(result["person_bbox"])

            # Feed the temporal smoother every tracked person, compliant or not,
            # so present observations accumulate and can clear prior alerts.
            # PPEDetector aggregates by class and does not expose per-item
            # confidence, so we encode presence as 1.0 and absence as None.
            # The tracker's uncertain-band logic is therefore dormant in
            # production but exercised by tests against the API directly.
            tracker_actions = None
            if track_id is not None:
                observations = {
                    item: 1.0 if item in result["detected_ppe"] else None
                    for item in self.required_items
                }
                tracker_actions = self.compliance_tracker.update(
                    track_id, observations, timestamp=events["timestamp"],
                )
                for item, action in tracker_actions.items():
                    if action is None:
                        continue
                    summary = self.compliance_tracker.get_window_summary(track_id, item)
                    self.event_logger.log_compliance_transition(
                        track_id, item, action, summary,
                    )
                    if action == "clear":
                        logger.info(
                            "Track %s: PPE %s compliance cleared", track_id, item,
                        )

            if i in fall_indices:
                events["falls"].append({
                    "worker_id": worker_id,
                    "worker_name": worker_name,
                    "track_id": track_id,
                    "bbox": result["person_bbox"],
                })
            elif not result["compliant"]:
                # Renderer reads ppe_violations for per-frame red boxes —
                # keep it instantaneous so visuals don't lag the smoother.
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

                # Untracked detections bypass the smoother (safety-first:
                # we can't accumulate history without an identity).
                if track_id is None:
                    events["confirmed_ppe_violations"].append({
                        "worker_id": worker_id,
                        "worker_name": worker_name,
                        "qr_data": qr_data,
                        "track_id": None,
                        "bbox": result["person_bbox"],
                        "missing": result["missing_ppe"],
                        "detected": result["detected_ppe"],
                        "zone": zone,
                    })
                elif tracker_actions and any(a == "alert" for a in tracker_actions.values()):
                    current = self.compliance_tracker.get_state(track_id)
                    alerted_items = [item for item, on in current.items() if on]
                    events["confirmed_ppe_violations"].append({
                        "worker_id": worker_id,
                        "worker_name": worker_name,
                        "qr_data": qr_data,
                        "track_id": track_id,
                        "bbox": result["person_bbox"],
                        "missing": alerted_items,
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

            if zone is not None and not self.zone_monitor.is_permitted(zone, worker_id):
                events["zone_breaches"].append({
                    "worker_id": worker_id,
                    "worker_name": worker_name,
                    "track_id": track_id,
                    "bbox": result["person_bbox"],
                    "zone_id": zone,
                })

        active_ids = [r.get("track_id") for r in ppe_results if r.get("track_id") is not None]
        self.worker_id.clear_stale(active_ids)

        self._compliance_cleanup_counter += 1
        if self._compliance_cleanup_counter >= _COMPLIANCE_CLEANUP_EVERY:
            self.compliance_tracker.cleanup_stale(current_ts=events["timestamp"])
            self._compliance_cleanup_counter = 0

        if self.wet_floor_detector is not None and self.wet_floor_detector.config.enabled:
            self._update_wet_floor(frame, events)

        return events

    def _update_wet_floor(self, frame, events):
        detections = self.wet_floor_detector.detect(frame)

        # Take the highest-confidence detection per frame for smoothing —
        # multi-region wet floor over time can be added later if needed.
        if not detections:
            self._wet_floor_history.append(None)
            if not any(self._wet_floor_history):
                self._wet_floor_first_seen_ts = None
            return

        best = max(detections, key=lambda d: d["confidence"])

        # Anchor IoU against the most recent non-empty entry so a single
        # dropped frame doesn't reset the streak.
        last = next((d for d in reversed(self._wet_floor_history) if d is not None), None)
        if last is None or _iou(best["bbox"], last["bbox"]) >= 0.5:
            self._wet_floor_history.append(best)
        else:
            self._wet_floor_history.clear()
            self._wet_floor_history.append(best)
            self._wet_floor_first_seen_ts = None

        if self._wet_floor_first_seen_ts is None:
            self._wet_floor_first_seen_ts = events["timestamp"]

        required = self.wet_floor_detector.config.consecutive_frames_required
        consecutive_count = sum(1 for d in self._wet_floor_history if d is not None)
        if consecutive_count >= required:
            events["wet_floor_events"].append({
                "bbox": best["bbox"],
                "confidence": best["confidence"],
                "area_pct": best["area_pct"],
                "consecutive_count": consecutive_count,
                "first_seen_ts": self._wet_floor_first_seen_ts,
            })

    def _detection_loop(self):
        while self.running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.001)
                continue

            events = self.process_frame(frame)

            if self.alert_client:
                # Smoothed alerts only — single-frame ppe_violations is for the
                # renderer, not for waking up the backend.
                for violation in events["confirmed_ppe_violations"]:
                    self.alert_client.send_ppe_alert(violation, frame)
                for zone_event in events["zone_breaches"]:
                    self.alert_client.send_zone_alert(zone_event, frame)
                for wf_event in events["wet_floor_events"]:
                    self.alert_client.send_wet_floor_alert(wf_event, frame)

            # Only update displayed events if we detected people,
            # otherwise keep showing previous results (avoids flickering
            # when detection momentarily misses a person).
            has_detections = (
                events["ppe_violations"] or
                events["compliant_workers"] or
                events["falls"]
            )
            with self._events_lock:
                if has_detections:
                    self._latest_events = events
                elif time.time() - self._latest_events["timestamp"] > 0.5:
                    self._latest_events = events

            self.event_logger.log_events(events)

    def run(self, display=True):
        self.running = True
        time.sleep(self.config.camera.startup_delay)

        print("Pipeline started. Press 'q' to stop.")

        fps = self.camera.stream.get(cv2.CAP_PROP_FPS) or 25
        frame_period = 1.0 / fps

        threading.Thread(target=self._detection_loop, daemon=True).start()

        while self.running:
            loop_start = time.perf_counter()

            frame = self.camera.read()
            if frame is None:
                time.sleep(0.001)
                continue

            with self._events_lock:
                events = self._latest_events

            if display:
                quit_requested = self.renderer.render_to_window(frame, events)
                if quit_requested:
                    break

            elapsed = time.perf_counter() - loop_start
            remaining = frame_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

        self.stop()

    def stop(self):
        self.running = False
        self.camera.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped.")
