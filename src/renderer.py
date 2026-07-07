import time

import cv2
from config.settings import DisplayConfig

# Never extrapolate a box further than this many seconds of motion — beyond
# it the estimate is a guess and the box would sail past the worker.
_MAX_PROJECT_SECONDS = 0.8


def project_bbox(bbox, velocity, age, frame_shape):
    """Shift a box along its track velocity by the age of the detection
    result. Detection runs a full inference cycle behind the displayed frame,
    so an unprojected box trails a moving worker by exactly that latency;
    riding the (EMA-smoothed) velocity forward closes most of the gap. The
    shift is capped in time and clamped to the frame."""
    if not velocity or age <= 0:
        return bbox
    dt = min(age, _MAX_PROJECT_SECONDS)
    dx, dy = int(velocity[0] * dt), int(velocity[1] * dt)
    if dx == 0 and dy == 0:
        return bbox
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, min(x1 + dx, w - bw))
    y1 = max(0, min(y1 + dy, h - bh))
    return [x1, y1, x1 + bw, y1 + bh]


class FrameRenderer:
    def __init__(self, zone_monitor, display_config=None):
        self.zone_monitor = zone_monitor
        self.config = display_config or DisplayConfig()

    def draw(self, frame, events):
        display = frame.copy()
        # How far behind the displayed frame these events are — worker boxes
        # are projected forward by this much along their track velocity.
        age = max(0.0, time.time() - events.get("timestamp", time.time()))

        for worker in events["compliant_workers"]:
            x1, y1, x2, y2 = project_bbox(
                worker["bbox"], worker.get("velocity"), age, frame.shape)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            wid = worker.get("worker_name") or worker.get("worker_id") or "?"
            cv2.putText(display, f"W:{wid} OK", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # events["pending_workers"] (Checking state) is intentionally NOT
        # drawn: it's internal evidence accumulation. Rendering it made every
        # track churn / entry flash a yellow box; a worker with no verdict yet
        # simply gets no annotation until one is earned.

        for violation in events["ppe_violations"]:
            x1, y1, x2, y2 = project_bbox(
                violation["bbox"], violation.get("velocity"), age, frame.shape)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"VIOLATION: {', '.join(violation['missing'])}"
            wid = violation.get("worker_name") or violation.get("worker_id") or "?"
            cv2.putText(display, f"W:{wid} {label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for fall_event in events["falls"]:
            x1, y1, x2, y2 = fall_event["bbox"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)
            severity = fall_event.get("severity") or ""
            label = f"FALL DETECTED [{severity}]" if severity else "FALL DETECTED"
            cv2.putText(display, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

        for wf in events.get("wet_floor_events", []):
            x1, y1, x2, y2 = wf["bbox"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 3)  # cyan = wet floor
            cv2.putText(display, "WET FLOOR", (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        for zone_event in events["zone_breaches"]:
            x1, y1, x2, y2 = zone_event["bbox"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(display, f"ZONE: {zone_event['zone_id']}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        for zone_id, polygon in self.zone_monitor.zones.items():
            cv2.polylines(display, [polygon], True, (0, 165, 255), 2)
            cv2.putText(display, zone_id, tuple(polygon[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        return display

    def render_to_window(self, frame, events):
        annotated = self.draw(frame, events)
        h, w = annotated.shape[:2]
        max_w = self.config.max_display_width
        if w > max_w:
            scale = max_w / w
            annotated = cv2.resize(annotated, (max_w, int(h * scale)))
        cv2.imshow(self.config.window_name, annotated)
        return cv2.waitKey(1) & 0xFF == ord('q')
