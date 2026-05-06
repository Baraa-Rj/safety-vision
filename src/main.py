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
fall_detector = FallDetector(
    config.fall.classifier_model_path,
    confidence=config.fall.confidence,
) if config.fall.enabled else None
worker_id = WorkerIdentifier()
zone_monitor = ZoneMonitor()
renderer = FrameRenderer(zone_monitor, config.display)
event_logger = EventLogger()
alert_client = AlertClient(
    config.alert.endpoint,
    enabled=config.alert.enabled,
    cooldown_seconds=config.alert.cooldown_seconds,
)

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
)

zones_file = "data/zones.json"
if os.path.exists(zones_file):
    with open(zones_file) as f:
        zones_data = json.load(f)
    for z in zones_data:
        pipeline.add_zone(z["zone_id"], z["points"])
    logging.info("Loaded %d zone(s) from %s", len(zones_data), zones_file)
else:
    for zone_def in config.zone.zones:
        pipeline.add_zone(zone_def.zone_id, zone_def.points)

pipeline.run(display=True)
