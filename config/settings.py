from dataclasses import dataclass, field


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


@dataclass
class FallDetectionConfig:
    model_path: str = "models/yolo26s-pose.pt"
    window_size: int = 15
    fall_speed_threshold: int = 15
    hip_drop_ratio: float = 0.25
    detection_interval: int = 2


@dataclass
class DisplayConfig:
    max_display_width: int = 960
    window_name: str = "Safety Vision"


@dataclass
class PipelineConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    ppe: PPEConfig = field(default_factory=PPEConfig)
    fall: FallDetectionConfig = field(default_factory=FallDetectionConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
