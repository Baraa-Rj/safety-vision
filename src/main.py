import os

# Force FFmpeg to use TCP for RTSP (UDP silently drops frames on busy networks)
# and apply a 5s socket timeout so dead connections don't hang the read loop.
# Must be set BEFORE cv2 is imported — OpenCV reads this env var at module load.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000",
)

import json
import logging
from config.settings import PipelineConfig
from src.camera import CameraStream
from src.ppe_detector import PPEDetector
from src.fall_detector import FallDetector
from src.wet_floor_detector import WetFloorDetector
from src.worker_id import WorkerIdentifier
from src.zone_monitor import ZoneMonitor
from src.renderer import FrameRenderer
from src.event_logger import EventLogger
from src.alert_client import AlertClient
from src.pipeline import SafetyPipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")

config = PipelineConfig()

camera = CameraStream(
    config.camera.source,
    reconnect_after_seconds=config.camera.reconnect_after_seconds,
    open_max_attempts=config.camera.open_max_attempts,
    open_retry_backoff=config.camera.open_retry_backoff,
)
ppe_detector = PPEDetector(
    config.ppe.model_path,
    confidence=config.ppe.confidence,
    required_ppe=config.ppe.required_ppe,
    overlap_threshold=config.ppe.overlap_threshold,
    class_confidences=config.ppe.class_confidences,
    tracker_config=config.ppe.tracker_config,
)
# Detection-based fall detection: gates the 'fallen' class emitted by the PPE
# detector (models/best.pt). Toggle with FallDetectionConfig.enabled.
fall_detector = FallDetector(config.fall) if config.fall.enabled else None

# Wet floor detection: enable by setting WetFloorConfig.enabled = True
# AND providing models/wet_floor.pt. The detector is constructed unconditionally
# so the path is wired; it returns no detections (and only logs once) until
# both conditions are met.
wet_floor_detector = WetFloorDetector(config.wet_floor) if config.wet_floor.enabled else None

worker_id = WorkerIdentifier()
zone_monitor = ZoneMonitor()

# Zone monitoring is disabled for now (ZoneConfig.enabled). With no zones loaded,
# the monitor reports no breaches and the renderer draws no polygons — the rest
# of the pipeline is unaffected. Flip ZoneConfig.enabled = True to restore it.
if config.zone.enabled:
    zones_path = os.path.join(os.path.dirname(__file__), "..", "data", "zones.json")
    if os.path.exists(zones_path):
        with open(zones_path) as f:
            zones_data = json.load(f)
        for zone in zones_data:
            zone_monitor.add_zone(
                zone["zone_id"], zone["points"],
                zone.get("allowed_workers"), radius=zone.get("radius", 0),
            )
        logging.info(f"Loaded {len(zones_data)} zone(s) from {zones_path}")
    else:
        logging.warning(f"No zones file found at {zones_path}")
else:
    logging.info("Zone monitoring disabled (ZoneConfig.enabled = False)")

renderer = FrameRenderer(zone_monitor, config.display)
event_logger = EventLogger()

alert_client = AlertClient(
    config.alert.endpoint,
    enabled=config.alert.enabled,
    cooldown_seconds=config.alert.cooldown_seconds,
    zone_endpoint=config.alert.zone_endpoint,
    wet_floor_endpoint=config.alert.wet_floor_endpoint,
    fall_endpoint=config.alert.fall_endpoint,
    fall_unidentified_user_id=config.alert.fall_unidentified_user_id,
) if config.alert.enabled else None

pipeline = SafetyPipeline(
    config=config,
    camera=camera,
    ppe_detector=ppe_detector,
    fall_detector=fall_detector,
    worker_identifier=worker_id,
    zone_monitor=zone_monitor,
    renderer=renderer,
    event_logger=event_logger,
    alert_client=alert_client,
    wet_floor_detector=wet_floor_detector,
)

pipeline.run(display=True)
