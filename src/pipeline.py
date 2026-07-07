import cv2
import logging
import time
import threading
from collections import deque

from src.ppe_compliance_tracker import PPEComplianceTracker
from src.violation_confirmer import ViolationConfirmer

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
        # Time-based gate: only confirm a violation after it's been constant for
        # config.compliance.confirm_seconds (filters single-frame false pops).
        self.violation_confirmer = ViolationConfirmer(config.compliance.confirm_seconds)
        self._compliance_cleanup_counter = 0
        # Displayed verdict per track ("ok"/"violation" + last bbox/timestamp).
        # This is what keeps the UI stable while a worker moves: BoT-SORT churns
        # ids on fast motion (ReID is off) and each new id starts an empty
        # evidence window, so without memory the display would drop back to
        # Checking — or flash red off a few blurred frames — for a worker who
        # was confirmed compliant a moment ago. See _resolve_display.
        self._display_memory = {}
        self._active_track_ids = set()
        # Last box + time each track id was seen — the substrate for churn
        # bridging: when a brand-new id overlaps a recently-lost one, it IS
        # that worker, and everything keyed by track id (QR identity, display
        # verdict) is carried over. See _resolve_track_continuity.
        self._track_last_seen = {}

        wf_required = (
            wet_floor_detector.config.consecutive_frames_required
            if wet_floor_detector is not None else 5
        )
        self._wet_floor_history = deque(maxlen=wf_required)
        self._wet_floor_first_seen_ts = None

        self._detection_thread = None
        self._events_lock = threading.Lock()
        self._latest_events = {
            "ppe_violations": [],
            "confirmed_ppe_violations": [],
            "compliant_workers": [],
            "pending_workers": [],
            "falls": [],
            "zone_breaches": [],
            "wet_floor_events": [],
            "timestamp": 0,
        }

    def add_zone(self, zone_id, points):
        self.zone_monitor.add_zone(zone_id, points)

    def _resolve_track_continuity(self, track_id, bbox, now):
        """Bridge tracker id churn: BoT-SORT drops and re-acquires a moving
        worker under a fresh id (ReID is off; association is motion/IoU only),
        and everything keyed by track id — the QR identity and the displayed
        compliance verdict — would die with the old id. When an id is seen for
        the FIRST time, find the recently-lost track (not active this frame,
        seen within track_timeout_seconds) whose last box overlaps this one:
        that is the same worker, so carry their state over to the new id.
        Live tracks are excluded so a new box next to a live worker can't
        steal their identity; the IoU floor keeps a worker across the room
        from inheriting anything."""
        if track_id in self._track_last_seen:
            self._track_last_seen[track_id] = {"bbox": bbox, "ts": now}
            return

        timeout = self.config.compliance.track_timeout_seconds
        best_id, best_iou = None, 0.3
        for tid, seen in self._track_last_seen.items():
            if tid in self._active_track_ids or now - seen["ts"] > timeout:
                continue
            iou = _iou(bbox, seen["bbox"])
            if iou > best_iou:
                best_id, best_iou = tid, iou

        if best_id is not None:
            self.worker_id.transfer(best_id, track_id)
            if best_id in self._display_memory:
                self._display_memory[track_id] = self._display_memory.pop(best_id)
            del self._track_last_seen[best_id]

        self._track_last_seen[track_id] = {"bbox": bbox, "ts": now}

    def _resolve_display(self, track_id, bbox, now):
        """Displayed verdict for a tracked person: "violation", "ok", or None
        (still checking — rendered as nothing). Wraps get_display_state with a
        per-worker memory so a verdict, once earned, survives the two things
        that made the display flicker on moving workers: track-id churn (new
        id = empty window = Checking) and short runs of missed vest/helmet
        detections (3 blurred frames = red under the bare warm-up vote).
        Downgrading a confirmed-compliant worker to red therefore always
        requires full-window hysteresis evidence — the same bar a stable
        track has — while a genuinely new worker still turns red after
        display_min_evidence frames."""
        dstate = self.compliance_tracker.get_display_state(track_id)
        flagged = sorted(item for item, v in dstate.items() if v is True)
        pending = any(v is None for v in dstate.values())

        # Churn inheritance already happened in _resolve_track_continuity, so
        # a re-acquired worker's verdict is sitting under their new id.
        memory = self._display_memory.get(track_id)

        if flagged:
            window_full = all(
                self.compliance_tracker.get_window_summary(track_id, item)["size"]
                >= self.config.compliance.window_size
                for item in self.required_items
            )
            if memory is not None and memory["verdict"] == "ok" and not window_full:
                # Warm-up red on a worker already confirmed compliant: hold
                # green until the window fills and hysteresis itself flags it.
                verdict, missing = "ok", []
            else:
                verdict, missing = "violation", flagged
        elif not pending:
            verdict, missing = "ok", []
        elif memory is not None:
            verdict, missing = memory["verdict"], memory["missing"]
        else:
            return None, []

        self._display_memory[track_id] = {
            "verdict": verdict, "missing": missing, "bbox": bbox, "ts": now,
        }
        return verdict, missing

    def process_frame(self, frame):
        events = {
            "ppe_violations": [],
            "confirmed_ppe_violations": [],
            "compliant_workers": [],
            "pending_workers": [],
            "falls": [],
            "zone_breaches": [],
            "wet_floor_events": [],
            "timestamp": time.time(),
        }

        ppe_results = self.ppe_detector.detect(frame)
        self._active_track_ids = {
            r.get("track_id") for r in ppe_results if r.get("track_id") is not None
        }

        # Falls come straight from best.pt's 'fallen' class (surfaced by the PPE
        # detector), so they're independent of the upright-person boxes — a worker
        # on the ground no longer needs a 'person' box to be flagged.
        if self.fall_detector:
            fallen_dets = getattr(self.ppe_detector, "fallen_detections", [])
            for fr in self.fall_detector.detect(fallen_dets):
                # Best-effort identity: reuse the ID this track was scanned with
                # while upright (their QR can't be read on the ground).
                identity = self.worker_id.get_cached(fr.get("track_id"))
                events["falls"].append({
                    "worker_id": identity["worker_id"] if identity else None,
                    "worker_name": identity["worker_name"] if identity else None,
                    "track_id": fr.get("track_id"),
                    "bbox": fr["bbox"],
                    "confidence": fr.get("confidence"),
                    "severity": fr.get("severity"),
                    "still_seconds": fr.get("still_seconds"),
                    "alert": fr.get("alert", False),
                })

        for i, result in enumerate(ppe_results):
            x1, y1, x2, y2 = result["person_bbox"]
            track_id = result.get("track_id")
            # Must run before identify(): a churned id inherits its QR identity
            # here, so the cache hit below finds it without a rescan.
            if track_id is not None:
                self._resolve_track_continuity(
                    track_id, result["person_bbox"], events["timestamp"])
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

            # Feed the time-based confirmer every frame (compliant resets the
            # streak). Key by the most stable identity we have.
            is_violation = not result["compliant"]
            conf_key = (
                worker_id if worker_id is not None
                else f"track_{track_id}" if track_id is not None
                else f"loc_{(x1 + x2) // 2 // 50}_{(y1 + y2) // 2 // 50}"
            )
            confirmed = self.violation_confirmer.update(
                conf_key, is_violation, events["timestamp"],
            )

            # On-screen state uses the SMOOTHED per-track verdict plus a
            # per-worker memory (_resolve_display), never the raw frame —
            # best.pt's vest/helmet detection flickers at this camera's
            # distance and BoT-SORT churns track ids on fast motion. Checking
            # (verdict None) is internal only: such workers go to
            # pending_workers, which the renderer does not draw. With no track
            # to smooth on, fall back to the raw frame.
            display_pending = False
            if track_id is not None:
                verdict, remembered_missing = self._resolve_display(
                    track_id, result["person_bbox"], events["timestamp"])
                display_violation = verdict == "violation"
                display_missing = remembered_missing or result["missing_ppe"]
                display_pending = verdict is None
            else:
                display_violation = not result["compliant"]
                display_missing = result["missing_ppe"]

            if display_violation:
                events["ppe_violations"].append({
                    "worker_id": worker_id,
                    "worker_name": worker_name,
                    "qr_data": qr_data,
                    "track_id": track_id,
                    "bbox": result["person_bbox"],
                    "missing": display_missing,
                    "detected": result["detected_ppe"],
                    "zone": zone,
                })
            elif display_pending:
                events["pending_workers"].append({
                    "worker_id": worker_id,
                    "worker_name": worker_name,
                    "track_id": track_id,
                    "bbox": result["person_bbox"],
                })
            else:
                events["compliant_workers"].append({
                    "worker_id": worker_id,
                    "worker_name": worker_name,
                    "track_id": track_id,
                    "bbox": result["person_bbox"],
                })

            # Backend alert: sustained, time-confirmed violation — separate from
            # the on-screen state, using the raw signal + confirm_seconds gate.
            if is_violation and confirmed:
                events["confirmed_ppe_violations"].append({
                    "worker_id": worker_id,
                    "worker_name": worker_name,
                    "qr_data": qr_data,
                    "track_id": track_id,
                    "bbox": result["person_bbox"],
                    "missing": result["missing_ppe"],
                    "detected": result["detected_ppe"],
                    "zone": zone,
                })

            if zone is not None and not self.zone_monitor.is_permitted(zone, worker_id):
                events["zone_breaches"].append({
                    "worker_id": worker_id,
                    "worker_name": worker_name,
                    "track_id": track_id,
                    "bbox": result["person_bbox"],
                    "zone_id": zone,
                    "zone_backend_id": self.zone_monitor.backend_ids.get(zone, 0),
                })

        active_ids = [r.get("track_id") for r in ppe_results if r.get("track_id") is not None]
        self.worker_id.clear_stale(active_ids)

        self._compliance_cleanup_counter += 1
        if self._compliance_cleanup_counter >= _COMPLIANCE_CLEANUP_EVERY:
            self.compliance_tracker.cleanup_stale(current_ts=events["timestamp"])
            self.violation_confirmer.cleanup(events["timestamp"])
            # Same timeout as the tracker: state too old to inherit belongs
            # to a worker who actually left, not a churned id.
            timeout = self.config.compliance.track_timeout_seconds
            self._display_memory = {
                tid: m for tid, m in self._display_memory.items()
                if events["timestamp"] - m["ts"] <= timeout
            }
            self._track_last_seen = {
                tid: s for tid, s in self._track_last_seen.items()
                if events["timestamp"] - s["ts"] <= timeout
            }
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
                # Fall alerts are already cooldown-gated in the detector; only the
                # frame that flips `alert` True reaches the backend.
                for fall in events["falls"]:
                    if fall.get("alert"):
                        self.alert_client.send_fall_alert(fall, frame)

            # Only update displayed events if we detected people,
            # otherwise keep showing previous results (avoids flickering
            # when detection momentarily misses a person).
            has_detections = (
                events["ppe_violations"] or
                events["compliant_workers"] or
                events["pending_workers"] or
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

        self._detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._detection_thread.start()

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
        # Let the detection thread finish its current frame before we tear down
        # the camera and OpenCV windows — otherwise it can be mid-inference or
        # mid-imshow when the interpreter exits, which crashes the C++ runtime
        # ("terminate called without an active exception").
        if self._detection_thread is not None:
            self._detection_thread.join(timeout=5.0)
            self._detection_thread = None
        self.camera.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped.")
