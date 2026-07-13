import os
from dataclasses import dataclass, field
from typing import List, Tuple

_DEFAULT_CAMERA_SOURCE = "data/sample_videos/0712.mp4"


def _camera_source():
    # CAMERA_RTSP_URL keeps live-camera credentials out of git.
    return os.environ.get("CAMERA_RTSP_URL") or _DEFAULT_CAMERA_SOURCE


# Backend base URL. The real deployment address is kept out of source — set
# BACKEND_URL in your environment (.env). Defaults to localhost so a fresh
# clone never points at, or leaks, a production server.
_DEFAULT_BACKEND_URL = "http://localhost:8080"


def _backend_base():
    return (os.environ.get("BACKEND_URL") or _DEFAULT_BACKEND_URL).rstrip("/")


def _ppe_endpoint():
    return os.environ.get("PPE_ALERTS_ENDPOINT") or f"{_backend_base()}/api/ppe-alerts/create"


def _zones_endpoint():
    # ZONES_ENDPOINT overrides the derived URL for split deployments.
    return os.environ.get("ZONES_ENDPOINT") or f"{_backend_base()}/api/zones"


def _fall_endpoint():
    return os.environ.get("FALL_ALERTS_ENDPOINT") or f"{_backend_base()}/api/fall-alerts/create"


def _wet_endpoint():
    # Confirmed against the live OpenAPI spec: POST /api/wet-alert/create.
    return os.environ.get("WET_ALERTS_ENDPOINT") or f"{_backend_base()}/api/wet-alert/create"


def _zone_alert_endpoint():
    # Breach alerts (distinct from _zones_endpoint, which uploads zone shapes).
    # Confirmed against the live OpenAPI spec: POST /api/zone-alerts/create.
    return os.environ.get("ZONE_ALERTS_ENDPOINT") or f"{_backend_base()}/api/zone-alerts/create"


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
        "vest": 0.35,
        "helmet": 0.35,
        "person": 0.35,
        "fallen": 0.40,
    })
    required_ppe: set = field(default_factory=lambda: {"helmet", "vest"})
    overlap_threshold: float = 0.5
    # Keep vest/helmet sightings down to this confidence and report them as
    # ppe_confidences (uncertain evidence) — must match uncertain_lower in
    # ComplianceConfig.
    ppe_uncertain_floor: float = 0.25
    # Inference resolution. 640 (YOLO's native) resolves small/distant PPE and
    # partially-occluded vests far better than 480; the cost is ~1.8x slower
    # inference, acceptable on the process_every_n cadence.
    imgsz: int = 640
    process_every_n: int = 3
    # Custom BoT-SORT config: larger track_buffer + ReID so a worker keeps the
    # same track_id (and thus their cached QR identity) across movement.
    tracker_config: str = "config/botsort_persistent.yaml"


@dataclass
class FallDetectionConfig:
    # Detection-based: best.pt has a dedicated 'fallen' class, so a worker on the
    # ground is detected directly (see PPEConfig.model_path). This is just the
    # temporal gate over those detections — no separate model or crop step.
    fallen_conf: float = 0.40               # min detection confidence for a fallen box
    consecutive_frames: int = 5             # sustained fallen frames before flagging
    min_size: int = 40                      # skip fallen boxes smaller than this (px)
    # A fallen person lies horizontal: real fallen boxes are wider than tall
    # (measured h/w <= ~0.9). Reject tall/narrow boxes as false positives.
    max_aspect_ratio: float = 1.5           # drop fallen box if height/width exceeds this
    # Severity (triage priority) thresholds. Severity escalates the longer a
    # worker stays down and the stiller they are; a backend alert is sent on the
    # first confirmation and again each time the tier rises (LOW->MEDIUM->HIGH).
    # No backend alert until a fall has persisted this long — filters brief
    # false positives (a stumble or quick crouch). The fall is still detected and
    # drawn on screen immediately; only the alert (POST) waits.
    alert_delay_seconds: float = 8.0
    still_motion_px: int = 15               # centroid move below this = "still"
    medium_seconds: float = 5.0             # on the ground this long -> MEDIUM
    high_still_seconds: float = 20.0        # motionless this long -> HIGH (urgent)
    # No fallen detection for this long ends the event (worker recovered). Longer
    # than the streak gap so a brief detector dropout doesn't reset the severity
    # clock — a motionless worker still escalates through flicker.
    recovery_seconds: float = 3.0
    enabled: bool = True


@dataclass
class ComplianceConfig:
    # On-screen red appears once an item is missing in missing_to_alert of the
    # last window_size processed frames (the window must fill first). Tuned to
    # flag a real violation in ~1s while ignoring single-frame detection flicker:
    # a true violator is missing ~all frames, a compliant worker only a few.
    # Smaller window / lower ratio = faster red but twitchier.
    window_size: int = 8
    missing_to_alert: int = 5
    present_to_clear: int = 2
    # A full window with this many present frames marks the item "established"
    # (reliably worn). Established items only flip red on a fully-missing
    # window: the helmet model's misses come in multi-second bursts on a worn
    # helmet (measured: it emits either conf >= 0.35 or nothing — the
    # 0.20-0.35 band is 0.5% of frames), and 5-of-8 flapped alert/clear on a
    # compliant worker. A removed helmet still reaches all-missing within one
    # window, so a genuine violation is delayed by 3 frames, not suppressed.
    present_to_establish: int = 5
    # Minimum processed frames agreeing before a NEW track gets any on-screen
    # verdict at all; below this it renders as yellow "checking". Prevents both
    # failure modes seen live: defaulting green until the window fills (a
    # no-PPE worker shown OK) and flashing red off a single blurry entry frame
    # or a transient phantom person box (a compliant worker shown VIOLATION).
    display_min_evidence: int = 3
    track_timeout_seconds: float = 5.0
    # Confidence bands for tracker observations. Aligned with the detector:
    # >= 0.35 (the PPE class threshold) counts as present, < 0.25 (the
    # detector's ppe_uncertain_floor, below which sightings aren't reported)
    # as missing, in between as uncertain — evidence against "missing" that
    # doesn't yet prove "present".
    uncertain_lower: float = 0.25
    uncertain_upper: float = 0.35
    # A violation must be constant for this long before it's sent to the backend
    # — filters single-frame false pops.
    confirm_seconds: float = 8.0


@dataclass
class WetFloorConfig:
    enabled: bool = True                        # wet_floor.pt trained (YOLO26-seg, single class)
    model_path: str = "models/wet_floor.pt"
    confidence_threshold: float = 0.5
    min_area_pct: float = 0.25                  # ignore boxes <0.25% of frame (real papers run 0.4-1.4%)
    consecutive_frames_required: int = 5        # temporal smoothing
    # Run the seg model only every Nth processed frame. A spill is a static
    # hazard, and this model runs in the same loop as PPE detection — skipping
    # it most frames cuts detection-loop latency, which is what keeps the
    # worker boxes fresh on CPU. Confirmed events are re-emitted on skipped
    # frames so the wet-floor box doesn't blink.
    process_every_n: int = 3


@dataclass
class DisplayConfig:
    max_display_width: int = 960
    window_name: str = "Safety Vision"
    # A tracked worker whose detection drops out keeps their last box (and
    # verdict) on screen for this long — bridges blur/pose/occlusion dropouts
    # so rectangles don't flicker off while the worker is clearly still there.
    # Short enough that a worker who genuinely leaves doesn't haunt the frame,
    # and that a fast walker doesn't leave a trail of expired positions.
    box_hold_seconds: float = 1.0
    # Processed-frame floor for the hold: on a CPU the detection loop can run
    # at ~1 frame/s, making 1.0s of wall clock a single frame of tolerance —
    # one dropout blanked the worker. The hold survives while EITHER window
    # (seconds or processed frames) is still open.
    box_hold_frames: int = 3


@dataclass
class AlertConfig:
    endpoint: str = field(default_factory=_ppe_endpoint)
    enabled: bool = True
    # Backoff base per alert key: first alert immediate (violations are
    # already confirm_seconds-confirmed upstream), then repeats at 2x, 3x,
    # 4x... this gap (16s, 24s, 32s, ...) so an ongoing violation notifies
    # with decreasing frequency instead of a fixed-rate stream.
    cooldown_seconds: float = 8.0
    # Zone breach alerts have a dedicated endpoint + payload
    # ({imgImage, userId, zoneId, message}), so this is derived (not blank).
    zone_endpoint: str = field(default_factory=_zone_alert_endpoint)
    # Breaching workers are often not QR-identified, but the endpoint requires a
    # valid userId FK (a blank one returns 500). So anonymous breaches are sent
    # under this sentinel — a real user row — which defaults to the shared
    # "2678a" anonymous id and is overridable via ZONE_UNIDENTIFIED_USER_ID.
    zone_unidentified_user_id: str = field(
        default_factory=lambda: os.environ.get("ZONE_UNIDENTIFIED_USER_ID") or "2678a")
    # Wet floor has a dedicated endpoint + payload ({description, imgImage}), so
    # this is derived (not blank) — same rationale as fall_endpoint below.
    wet_floor_endpoint: str = field(default_factory=_wet_endpoint)
    # Wet-floor cadence: first alert once the spill has persisted this long,
    # then backoff repeats at 2x, 3x... the base (20s, 30s, 40s, ...).
    wet_floor_first_alert_seconds: float = 10.0
    wet_floor_cooldown_seconds: float = 10.0
    # Falls have a dedicated endpoint + payload, so this is derived (not blank).
    fall_endpoint: str = field(default_factory=_fall_endpoint)
    # The fall endpoint requires a userId FK. A fallen worker often can't be
    # identified, so anonymous falls are sent under this sentinel user id (a
    # real row that must exist in the backend). A fall is a critical event —
    # it must NEVER be silently skipped — so this defaults to the same shared
    # anonymous row the zone alerts use rather than blank.
    fall_unidentified_user_id: str = field(
        default_factory=lambda: os.environ.get("FALL_UNIDENTIFIED_USER_ID") or "2678a")


@dataclass
class ZoneDefinition:
    zone_id: str = ""
    points: List[List[int]] = field(default_factory=list)
    allowed_workers: List[str] = field(default_factory=list)
    radius: int = 0  # keep-out distance in px around the polygon (0 = containment)


def _zones_enabled():
    # ZONES_ENABLED=0 turns zone monitoring off for a single run without
    # touching zones.json — used by the demo launcher so the restricted-zone
    # polygon only appears in the zone-breach scenario.
    return os.environ.get("ZONES_ENABLED", "1").lower() not in ("0", "false", "no")


@dataclass
class ZoneConfig:
    enabled: bool = field(default_factory=_zones_enabled)  # loads data/zones.json
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
