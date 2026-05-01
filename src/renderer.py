import cv2
from config.settings import DisplayConfig


class FrameRenderer:
    def __init__(self, zone_monitor, display_config=None, fps=25):
        self.zone_monitor = zone_monitor
        self.config = display_config or DisplayConfig()
        self._wait_ms = max(1, int(1000 / fps))

    def draw(self, frame, events):
        display = frame.copy()

        for worker in events["compliant_workers"]:
            x1, y1, x2, y2 = worker["bbox"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            wid = worker.get("worker_name") or worker.get("worker_id") or "?"
            cv2.putText(display, f"W:{wid} OK", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for violation in events["ppe_violations"]:
            x1, y1, x2, y2 = violation["bbox"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"VIOLATION: {', '.join(violation['missing'])}"
            wid = violation.get("worker_name") or violation.get("worker_id") or "?"
            cv2.putText(display, f"W:{wid} {label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for fall_event in events["falls"]:
            x1, y1, x2, y2 = fall_event["bbox"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(display, "FALL DETECTED", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

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
        return cv2.waitKey(self._wait_ms) & 0xFF == ord('q')
