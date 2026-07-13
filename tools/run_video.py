"""Headless batch runner: feed a video file through the full SafetyPipeline
and write an annotated copy plus a JSON event summary.

Mirrors the live pipeline (same detectors, smoothing, zones) with three
differences: no display window, no backend alerts, and time.time() is
replaced by a video clock (frame_idx / fps) so every temporal gate —
confirm_seconds, fall severity, box holds — behaves as it would at real-time
playback regardless of how slow offline inference runs.

Usage:
    python3 tools/run_video.py VIDEO [VIDEO ...] [--out-dir DIR]
                               [--every N] [--max-width W]
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- Video clock: must be installed before any module captures time.time ---
_real_time = time.time
_video_clock = {"now": _real_time()}


def _fake_time():
    return _video_clock["now"]


time.time = _fake_time

import cv2  # noqa: E402

from config.settings import PipelineConfig  # noqa: E402
from src.ppe_detector import PPEDetector  # noqa: E402
from src.fall_detector import FallDetector  # noqa: E402
from src.wet_floor_detector import WetFloorDetector  # noqa: E402
from src.worker_id import WorkerIdentifier  # noqa: E402
from src.zone_monitor import ZoneMonitor  # noqa: E402
from src.renderer import FrameRenderer  # noqa: E402
from src.event_logger import EventLogger  # noqa: E402
from src.pipeline import SafetyPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("run_video")


def build_pipeline(config):
    ppe_detector = PPEDetector(
        config.ppe.model_path,
        confidence=config.ppe.confidence,
        required_ppe=config.ppe.required_ppe,
        overlap_threshold=config.ppe.overlap_threshold,
        class_confidences=config.ppe.class_confidences,
        tracker_config=config.ppe.tracker_config,
        imgsz=config.ppe.imgsz,
        ppe_uncertain_floor=config.ppe.ppe_uncertain_floor,
    )
    fall_detector = FallDetector(config.fall) if config.fall.enabled else None
    wet_floor_detector = (
        WetFloorDetector(config.wet_floor) if config.wet_floor.enabled else None
    )
    zone_monitor = ZoneMonitor()
    if config.zone.enabled:
        zones_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "zones.json")
        if os.path.exists(zones_path):
            with open(zones_path) as f:
                zones_data = json.load(f)
            for zone in zones_data:
                zone_monitor.add_zone(
                    zone["zone_id"], zone["points"],
                    zone.get("allowed_workers"), radius=zone.get("radius", 0),
                    backend_id=zone.get("backend_id", 0),
                    frame_size=zone.get("frame_size"),
                )
            logger.info("Loaded %d zone(s)", len(zones_data))

    renderer = FrameRenderer(zone_monitor, config.display)
    pipeline = SafetyPipeline(
        config=config,
        camera=None,  # process_frame never touches the camera
        ppe_detector=ppe_detector,
        fall_detector=fall_detector,
        worker_identifier=WorkerIdentifier(),
        zone_monitor=zone_monitor,
        renderer=renderer,
        event_logger=EventLogger(),
        alert_client=None,  # offline run: nothing POSTs to the backend
        wet_floor_detector=wet_floor_detector,
    )
    return pipeline, renderer


def process_video(path, out_dir, every, max_width):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    config = PipelineConfig()
    pipeline, renderer = build_pipeline(config)

    base = os.path.splitext(os.path.basename(path))[0]
    out_video = os.path.join(out_dir, f"{base}_annotated.mp4")
    writer = None

    base_ts = _real_time()
    events = None
    summary = {
        "video": os.path.basename(path),
        "fps": fps,
        "frames": 0,
        "processed_frames": 0,
        "violation_frames": 0,
        "missing_item_frames": {},
        "confirmed_violations": 0,
        "compliant_frames": 0,
        "identified_workers": set(),
        "fall_frames": 0,
        "fall_alerts": 0,
        "max_fall_severity": None,
        "max_fall_still_seconds": 0.0,
        "wet_floor_frames": 0,
        "zone_breach_frames": 0,
    }
    severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

    frame_idx = 0
    wall_start = _real_time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Advance the video clock BEFORE processing so every temporal gate
        # sees playback time.
        _video_clock["now"] = base_ts + frame_idx / fps

        if frame_idx % every == 0:
            events = pipeline.process_frame(frame)
            summary["processed_frames"] += 1

            if events["ppe_violations"]:
                summary["violation_frames"] += 1
                for v in events["ppe_violations"]:
                    for item in v.get("missing", []):
                        summary["missing_item_frames"][item] = (
                            summary["missing_item_frames"].get(item, 0) + 1)
            summary["confirmed_violations"] += len(
                events["confirmed_ppe_violations"])
            if events["compliant_workers"]:
                summary["compliant_frames"] += 1
            for group in ("compliant_workers", "ppe_violations", "falls"):
                for e in events[group]:
                    if e.get("worker_name") or e.get("worker_id"):
                        summary["identified_workers"].add(
                            e.get("worker_name") or e.get("worker_id"))
            if events["falls"]:
                summary["fall_frames"] += 1
                for fall in events["falls"]:
                    if fall.get("alert"):
                        summary["fall_alerts"] += 1
                    sev = fall.get("severity")
                    if sev and severity_rank.get(sev, 0) > severity_rank.get(
                            summary["max_fall_severity"] or "", 0):
                        summary["max_fall_severity"] = sev
                    summary["max_fall_still_seconds"] = max(
                        summary["max_fall_still_seconds"],
                        fall.get("still_seconds") or 0.0)
            if events["wet_floor_events"]:
                summary["wet_floor_frames"] += 1
            if events["zone_breaches"]:
                summary["zone_breach_frames"] += 1

        annotated = renderer.draw(frame, events) if events else frame
        h, w = annotated.shape[:2]
        if w > max_width:
            scale = max_width / w
            annotated = cv2.resize(annotated, (max_width, int(h * scale)))
        if writer is None:
            oh, ow = annotated.shape[:2]
            writer = cv2.VideoWriter(
                out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 200 == 0:
            elapsed = _real_time() - wall_start
            logger.info("%s: %d/%d frames (%.1f s elapsed)",
                        base, frame_idx, total, elapsed)

    cap.release()
    if writer is not None:
        writer.release()

    summary["frames"] = frame_idx
    summary["identified_workers"] = sorted(summary["identified_workers"])
    summary["wall_seconds"] = round(_real_time() - wall_start, 1)
    out_json = os.path.join(out_dir, f"{base}_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote %s and %s", out_video, out_json)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out-dir", default="data/sample_videos")
    ap.add_argument("--every", type=int, default=None,
                    help="process every Nth frame (default: PPEConfig.process_every_n)")
    ap.add_argument("--max-width", type=int, default=1440)
    args = ap.parse_args()

    every = args.every or PipelineConfig().ppe.process_every_n
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for path in args.videos:
        logger.info("=== %s ===", path)
        results.append(process_video(path, args.out_dir, every, args.max_width))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
