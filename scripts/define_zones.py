"""Interactive tool to define restricted zones by clicking on a video frame.

Usage:
    python scripts/define_zones.py

Controls:
    Left-click  — add a point to the current zone polygon
    Right-click — finish the current zone and start a new one
    'u'         — undo the last point
    's'         — save all zones and exit
    'q'         — quit without saving

Output:
    Prints zone definitions ready to paste into config/settings.py
    Also saves to data/zones.json for programmatic loading.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# RTSP needs TCP transport + a socket timeout, set BEFORE cv2 is imported
# (OpenCV reads this env var at module load). Mirrors src/main.py so the frame
# we draw zones on comes through the same path the pipeline uses.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000",
)

import json

import cv2
import numpy as np
from config.settings import PipelineConfig
from src.zone_client import ZoneClient

config = PipelineConfig()
source = config.camera.source
is_rtsp = isinstance(source, str) and source.lower().startswith("rtsp://")
print(f"Capturing a frame from: {source}")

cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG) if is_rtsp else cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"Cannot open {source}")
    sys.exit(1)

# Grab a frame to draw on. The first RTSP reads can be empty or a pre-keyframe
# (garbled green/gray), so warm up and keep the latest good frame.
frame = None
if is_rtsp:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(30):
        ret, f = cap.read()
        if ret and f is not None:
            frame = f
else:
    ret, f = cap.read()
    if ret:
        frame = f
cap.release()

if frame is None:
    print("Failed to read a frame")
    sys.exit(1)

zones = []
current_points = []
zone_counter = 1
pending_zone_points = None  # finished polygon awaiting the worker-ID prompt
save_requested = False

# Scale frame for display if too large
MAX_DISPLAY_WIDTH = 960
h_orig, w_orig = frame.shape[:2]
if w_orig > MAX_DISPLAY_WIDTH:
    scale = MAX_DISPLAY_WIDTH / w_orig
else:
    scale = 1.0
display_size = (int(w_orig * scale), int(h_orig * scale))


def draw_overlay():
    display = frame.copy()

    # Draw completed zones
    for zone in zones:
        pts = np.array(zone["points"], dtype=np.int32)
        cv2.polylines(display, [pts], True, (0, 165, 255), 2)
        cv2.fillPoly(display, [pts], (0, 165, 255, 40))
        # Blend for transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 165, 255))
        cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
        cv2.polylines(display, [pts], True, (0, 165, 255), 2)
        cv2.putText(display, zone["zone_id"], tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # Draw current polygon in progress
    if current_points:
        for pt in current_points:
            cv2.circle(display, tuple(pt), 5, (0, 255, 0), -1)
        if len(current_points) > 1:
            pts = np.array(current_points, dtype=np.int32)
            cv2.polylines(display, [pts], False, (0, 255, 0), 2)

    # Instructions
    cv2.putText(display, "L-click: add point | R-click: finish zone | 'u': undo | 's': save | 'q': quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display, f"Zone {zone_counter} — {len(current_points)} points",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Resize for display
    if scale != 1.0:
        display = cv2.resize(display, display_size)
    return display


def prompt_allowed_workers(zone_id):
    """Ask the operator which worker IDs may enter this zone.

    Blank input means an empty whitelist — ZoneMonitor treats that as
    'deny everyone', so the zone is fully restricted.
    """
    try:
        raw = input(
            f"  Allowed worker IDs for '{zone_id}' "
            f"(comma-separated, blank = none): "
        ).strip()
    except EOFError:
        raw = ""
    return [w.strip() for w in raw.split(",") if w.strip()]


def prompt_radius(zone_id):
    """Keep-out radius in pixels. 0 = plain area (foot must be inside).
    >0 turns the polygon into a machine keep-out: triggers when feet come
    within this many pixels of the polygon edge.
    """
    try:
        raw = input(
            f"  Keep-out radius in px for '{zone_id}' "
            f"(0 = restricted area, >0 = machine proximity): "
        ).strip()
    except EOFError:
        raw = ""
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def finalize_zone(points):
    # Record geometry only. Worker IDs / radius are collected after the GUI
    # closes, so we never block on input() while the heavy window is held open.
    global zone_counter
    zone_id = f"restricted_{zone_counter}"
    zones.append({
        "zone_id": zone_id,
        "points": points,
        "allowed_workers": [],
        "radius": 0,
    })
    print(f"Zone '{zone_id}' defined with {len(points)} points")
    zone_counter += 1


def save_zones(path="data/zones.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(zones, f, indent=2)
    return path


def mouse_callback(event, x, y, flags, param):
    global current_points, pending_zone_points

    # Map display coordinates back to original frame coordinates
    orig_x = int(x / scale)
    orig_y = int(y / scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([orig_x, orig_y])

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_points) >= 3:
            # Hand the finished polygon to the main loop. Prompting for worker
            # IDs with input() *here* would block inside the GUI event callback,
            # which crashes the HighGUI backend.
            pending_zone_points = current_points.copy()
            current_points = []
        else:
            print("Need at least 3 points to define a zone")


window_name = "Define Zones"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

print("Click on the frame to define zone polygons.")
print("Left-click to add points, right-click to finish a zone.")
print("Press 's' to save, 'q' to quit.\n")

while True:
    display = draw_overlay()
    cv2.imshow(window_name, display)
    key = cv2.waitKey(30) & 0xFF

    # A right-click finished a zone — record it (no blocking prompt while the
    # window is up; worker IDs are collected after the GUI closes).
    if pending_zone_points is not None:
        finalize_zone(pending_zone_points)
        pending_zone_points = None

    if key == ord('q'):
        print("Quit without saving.")
        break

    elif key == ord('u'):
        if current_points:
            current_points.pop()
            print("Undid last point")

    elif key == ord('s'):
        # Finish the in-progress zone if it has enough points, then leave the GUI.
        if len(current_points) >= 3:
            finalize_zone(current_points.copy())
            current_points = []

        if not zones:
            print("No zones defined. Nothing to save.")
            break

        save_requested = True
        break

# Tear the window down BEFORE prompting/saving. On a memory-starved machine,
# holding the heavy GUI open while blocked on input() invites the OOM killer.
cv2.destroyAllWindows()

if save_requested:
    # Persist geometry FIRST (empty permissions) so the polygons are never lost,
    # even if the worker-ID prompt below is interrupted.
    output_path = save_zones()
    print(f"\nSaved {len(zones)} zone(s) to {output_path}")

    # Window is closed and the process is lean now — safe to prompt.
    for z in zones:
        z["allowed_workers"] = prompt_allowed_workers(z["zone_id"])
        z["radius"] = prompt_radius(z["zone_id"])
    save_zones()

    # Push to the server. Local save already done, so a failed upload loses nothing.
    zone_client = ZoneClient(config.zone.endpoint, enabled=config.zone.upload_enabled)
    if zone_client.upload_zones(zones):
        print(f"Uploaded {len(zones)} zone(s) to {config.zone.endpoint}")
    else:
        print(f"Zone upload to {config.zone.endpoint} did not complete (see log). "
              f"Zones are saved locally in {output_path}.")

    # Print config snippet
    print("\n--- Paste into config/settings.py ZoneConfig ---\n")
    print("@dataclass")
    print("class ZoneConfig:")
    print("    zones: List[ZoneDefinition] = field(default_factory=lambda: [")
    for z in zones:
        print(f'        ZoneDefinition(zone_id="{z["zone_id"]}", '
              f'points={z["points"]}, allowed_workers={z["allowed_workers"]}, '
              f'radius={z.get("radius", 0)}),')
    print("    ])")
    print("    alert_enabled: bool = True")
    print("    cooldown_seconds: float = 30.0")
