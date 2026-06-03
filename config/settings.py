import os
from dataclasses import dataclass, field
from typing import List, Tuple

_DEFAULT_CAMERA_SOURCE = "data/sample_videos/full-vest-with-QR.mp4"


def _camera_source():
    # CAMERA_RTSP_URL keeps live-camera credentials out of git.
    return os.environ.get("CAMERA_RTSP_URL") or _DEFAULT_CAMERA_SOURCE


_DEFAULT_ZONES_ENDPOINT = "http://203.0.113.10:8080/api/zones"


def _zones_endpoint():
    # ZONES_ENDPOINT lets deployments point zone uploads elsewhere without a code change.
    return os.environ.get("ZONES_ENDPOINT") or _DEFAULT_ZONES_ENDPOINT


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
    # Custom BoT-SORT config: larger track_buffer + ReID so a worker keeps the
    # same track_id (and thus their cached QR identity) across movement.
    tracker_config: str = "config/botsort_persistent.yaml"


@dataclass
class FallDetectionConfig:
    # Pose-based detector (no trained fall model needed).
    pose_model_path: str = "models/yolo26s-pose.pt"
    person_conf: float = 0.40               # pose person-detection confidence
    min_keypoint_conf: float = 0.30         # ignore keypoints below this
    torso_angle_threshold: float = 50.0     # deg from vertical; >= this = lying
    aspect_ratio_threshold: float = 1.2     # fallback: kp-box width/height >= this
    consecutive_frames: int = 5             # sustained frames before flagging a fall
    match_iou: float = 0.3                  # IoU to match a pose to a PPE person box
    enabled: bool = False


@dataclass
class ComplianceConfig:
    window_size: int = 15
    missing_to_alert: int = 10
    present_to_clear: int = 3
    track_timeout_seconds: float = 5.0
    uncertain_lower: float = 0.25
    uncertain_upper: float = 0.40
    # A violation must be constant for this long before it's sent to the backend
    # — filters single-frame false pops.
    confirm_seconds: float = 15.0


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
    endpoint: str = "http://203.0.113.10:8080/api/ppe-alerts/create"
    enabled: bool = True
    cooldown_seconds: float = 60.0          # 1 min per worker — don't alert every frame
    # Zone / wet-floor alerts POST a different payload shape; leave blank to
    # disable them so they never hit the PPE endpoint with the wrong body.
    zone_endpoint: str = ""
    wet_floor_endpoint: str = ""


@dataclass
class ZoneDefinition:
    zone_id: str = ""
    points: List[List[int]] = field(default_factory=list)
    allowed_workers: List[str] = field(default_factory=list)
    radius: int = 0  # keep-out distance in px around the polygon (0 = containment)


@dataclass
class ZoneConfig:
    enabled: bool = False                # zone monitoring off for now; flip to re-enable
    zones: List[ZoneDefinition] = field(default_factory=list)
    alert_enabled: bool = True
    cooldown_seconds: float = 30.0
    endpoint: str = field(default_factory=_zones_endpoint)
    upload_enabled: bool = True          # POST zones to the server when they're created


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
