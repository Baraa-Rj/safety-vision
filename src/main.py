import json
import logging
import os
from config.settings import PipelineConfig
from src.camera import CameraStream
from src.ppe_detector import PPEDetector
from src.fall_detector import FallDetector
from src.worker_id import WorkerIdentifier
from src.zone_monitor import ZoneMonitor
from src.renderer import FrameRenderer
from src.event_logger import EventLogger
from src.alert_client import AlertClient
from src.pipeline import SafetyPipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")

config = PipelineConfig()

camera = CameraStream(config.camera.source)
ppe_detector = PPEDetector(
    config.ppe.model_path,
    confidence=config.ppe.confidence,
    required_ppe=config.ppe.required_ppe,
    overlap_threshold=config.ppe.overlap_threshold,
)
fall_detector = None
worker_id = WorkerIdentifier()
zone_monitor = ZoneMonitor()

zones_path = os.path.join(os.path.dirname(__file__), "..", "data", "zones.json")
if os.path.exists(zones_path):
    with open(zones_path) as f:
        zones_data = json.load(f)
    for zone in zones_data:
        zone_monitor.add_zone(zone["zone_id"], zone["points"], zone.get("allowed_workers"))
    logging.info(f"Loaded {len(zones_data)} zone(s) from {zones_path}")
else:
    logging.warning(f"No zones file found at {zones_path}")

renderer = FrameRenderer(zone_monitor, config.display)
event_logger = EventLogger()

pipeline = SafetyPipeline(
    config=config,
    camera=camera,
    ppe_detector=ppe_detector,
    fall_detector=fall_detector,
    worker_identifier=worker_id,
    zone_monitor=zone_monitor,
    renderer=renderer,
    event_logger=event_logger,
    alert_client=None,
)

pipeline.run(display=True)
