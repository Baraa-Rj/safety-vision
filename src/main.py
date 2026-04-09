import logging
from config.settings import PipelineConfig
from src.camera import CameraStream
from src.ppe_detector import PPEDetector
from src.fall_detector import FallDetector
from src.worker_id import WorkerIdentifier
from src.zone_monitor import ZoneMonitor
from src.renderer import FrameRenderer
from src.event_logger import EventLogger
from src.pipeline import SafetyPipeline

logging.basicConfig(level=logging.WARNING, format="%(message)s")

config = PipelineConfig()

camera = CameraStream(config.camera.source)
ppe_detector = PPEDetector(
    config.ppe.model_path,
    confidence=config.ppe.confidence,
    required_ppe=config.ppe.required_ppe,
    overlap_threshold=config.ppe.overlap_threshold,
)
fall_detector = FallDetector(
    config.fall.model_path,
    window_size=config.fall.window_size,
    fall_speed_threshold=config.fall.fall_speed_threshold,
    hip_drop_ratio=config.fall.hip_drop_ratio,
)
worker_id = WorkerIdentifier()
zone_monitor = ZoneMonitor()
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
)
pipeline.run(display=True)
