"""Interactive restricted-zone editor.

Grabs one frame from the SAME camera the pipeline uses (CAMERA_RTSP_URL, else a
--image file), lets you click the corners of a restricted area, and writes the
polygon to data/zones.json. Capturing from the same source guarantees the points
are in the pipeline's native coordinates (e.g. 704x576 on the /1/2 substream), so
the zone lines up exactly at runtime.

Usage:
    CAMERA_RTSP_URL='rtsp://user:pass@host:554/1/2' python3 tools/define_zone.py
    python3 tools/define_zone.py --image path/to/frame.jpg      # define from a saved frame

Controls (in the window):
    left click   add a corner
    u            undo last corner
    c            clear all
    s / Enter    save polygon to data/zones.json (needs >= 3 corners)
    q / Esc      quit without saving
"""
import argparse
import json
import os

# Match main.py's RTSP transport so a substream opens reliably.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000"
)
import cv2  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ZONES_PATH = os.path.join(_ROOT, "data", "zones.json")


def grab_frame(image, source):
    if image:
        frame = cv2.imread(image)
        if frame is None:
            raise SystemExit(f"Could not read image: {image}")
        return frame
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera source: {source}")
    frame = None
    for _ in range(30):          # warm up — first RTSP frames are often junk
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    cap.release()
    if frame is None:
        raise SystemExit("No frame captured from camera.")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="define from a saved frame instead of the camera")
    ap.add_argument("--zone-id", default="restricted_1")
    ap.add_argument("--backend-id", type=int, default=0,
                    help="the server's integer id for this zone (sent as zoneId in alerts)")
    ap.add_argument("--append", action="store_true",
                    help="add to existing zones.json instead of replacing it")
    args = ap.parse_args()

    source = os.environ.get("CAMERA_RTSP_URL")
    if not args.image and not source:
        raise SystemExit("Set CAMERA_RTSP_URL or pass --image.")

    frame = grab_frame(args.image, source)
    h, w = frame.shape[:2]
    print(f"frame captured: {w}x{h}  (zone points will be saved in these coords)")

    points = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])

    win = f"Define zone '{args.zone_id}'  |  click corners, s=save, u=undo, c=clear, q=quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = frame.copy()
        for i, p in enumerate(points):
            cv2.circle(disp, tuple(p), 5, (0, 165, 255), -1)
            cv2.putText(disp, str(i + 1), (p[0] + 6, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(disp, tuple(points[i]), tuple(points[i + 1]), (0, 165, 255), 2)
        if len(points) >= 3:
            cv2.line(disp, tuple(points[-1]), tuple(points[0]), (0, 165, 255), 1)
        cv2.imshow(win, disp)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            print("cancelled — zones.json unchanged")
            break
        if key == ord("u") and points:
            points.pop()
        elif key == ord("c"):
            points.clear()
        elif key in (ord("s"), 13):
            if len(points) < 3:
                print("need at least 3 corners")
                continue
            zone = {"zone_id": args.zone_id, "backend_id": args.backend_id,
                    "points": points, "allowed_workers": []}
            existing = []
            if args.append and os.path.exists(_ZONES_PATH):
                existing = [z for z in json.load(open(_ZONES_PATH))
                            if z.get("zone_id") != args.zone_id]
            zones = existing + [zone]
            with open(_ZONES_PATH, "w") as f:
                json.dump(zones, f, indent=2)
            print(f"saved {len(zones)} zone(s) to {_ZONES_PATH}")
            print(json.dumps(zone))
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
