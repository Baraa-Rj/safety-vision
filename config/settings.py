import os
from dataclasses import dataclass, field
from typing import List, Tuple

_DEFAULT_CAMERA_SOURCE = "data/sample_videos/full-vest-with-QR.mp4"


def _camera_source():
    # CAMERA_RTSP_URL keeps live-camera credentials out of git.
    return os.environ.get("CAMERA_RTSP_URL") or _DEFAULT_CAMERA_SOURCE


@dataclass
class CameraConfig:
    source: str = field(default_factory=_camera_source)
    startup_delay: float = 3.0  # RTSP handshake can take a few seconds on first connect
    reconnect_after_seconds: float = 3.0
    open_max_attempts: int = 5
    open_retry_backoff: float = 2.0


@dataclass
class PPEConfig:
    model_path: str = "models/best.pt"
    confidence: float = 0.35                    # floor for any class without an entry in class_confidences
    # Per-class confidence overrides. Inference runs at min(values) so YOLO doesn't
    # drop low-confidence vests before we filter; each detection is then gated by
    # its own class threshold.
    class_confidences: dict = field(default_factory=lambda: {
        "vest": 0.20,
        "helmet": 0.35,
        "person": 0.35,
    })
    required_ppe: set = field(default_factory=lambda: {"helmet", "vest"})
    overlap_threshold: float = 0.5
    process_every_n: int = 3


@dataclass
class FallDetectionConfig:
    classifier_model_path: str = "models/fall_classifier/weights/best.pt"
    confidence: float = 0.7
    enabled: bool = False


@dataclass
class ComplianceConfig:
    window_size: int = 15
    missing_to_alert: int = 10
    present_to_clear: int = 3
    track_timeout_seconds: float = 5.0
    uncertain_lower: float = 0.25
    uncertain_upper: float = 0.40


@dataclass
class WetFloorConfig:
    enabled: bool = False                       # off until model exists
    model_path: str = "models/wet_floor.pt"
    confidence_threshold: float = 0.5
    min_area_pct: float = 1.0                   # ignore boxes <1% of frame
    consecutive_frames_required: int = 5        # temporal smoothing


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
    allowed_workers: List[str] = field(default_factory=list)


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
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    wet_floor: WetFloorConfig = field(default_factory=WetFloorConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    zone: ZoneConfig = field(default_factory=ZoneConfig)
