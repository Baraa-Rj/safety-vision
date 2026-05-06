from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class CameraConfig:
    source: str = "data/sample_videos/output.mp4"
    startup_delay: float = 1.0


@dataclass
class PPEConfig:
    model_path: str = "models/best.pt"
    confidence: float = 0.35
    required_ppe: set = field(default_factory=lambda: {"helmet", "vest"})
    overlap_threshold: float = 0.5
    process_every_n: int = 3


@dataclass
class FallDetectionConfig:
    classifier_model_path: str = "models/fall_classifier/weights/best.pt"
    confidence: float = 0.7
    enabled: bool = False


@dataclass
class DisplayConfig:
    max_display_width: int = 960
    window_name: str = "Safety Vision"


@dataclass
class AlertConfig:
    endpoint: str = "http://203.0.113.10:8080/api/ppe-alerts"
    enabled: bool = False
    cooldown_seconds: float = 30.0


@dataclass
class ZoneDefinition:
    zone_id: str = ""
    points: List[List[int]] = field(default_factory=list)


@dataclass
class ZoneConfig:
    zones: List[ZoneDefinition] = field(default_factory=list)
    alert_enabled: bool = True
    cooldown_seconds: float = 30.0


@dataclass
class PipelineConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    ppe: PPEConfig = field(default_factory=PPEConfig)
    fall: FallDetectionConfig = field(default_factory=FallDetectionConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    zone: ZoneConfig = field(default_factory=ZoneConfig)
