# Safety-Vision — Project State

## Purpose
Real-time computer vision system for industrial/construction safety monitoring. Detects PPE compliance (helmets, vests), identifies workers via QR badges, monitors restricted zones with per-worker permissions, detects falls, and sends alerts to a remote backend.

## Tech Stack
- **Python 3.10**, OpenCV, Ultralytics (YOLO26 weights)
- **Detection:** single YOLO model with a dedicated `fallen` class — PPE and fall detection share one detector, no separate crop/classifier step
- **Identification:** OpenCV QR code detector
- **Backend integration:** HTTP POST alerts with base64-encoded frames
- **Tests:** pytest

## Architecture

```
src/main.py                — entry point, wires everything together
src/pipeline.py            — orchestrator, threaded detection + render loop
src/camera.py              — threaded video capture (file or webcam)
src/ppe_detector.py        — YOLO-based PPE + person detection with tracking
src/fall_detector.py       — temporal gate over the detector's 'fallen' class, with severity scoring
src/wet_floor_detector.py  — YOLO-seg spill detection, temporally smoothed
src/worker_id.py           — QR code scanning with per-track caching
src/zone_monitor.py        — polygon zones with per-worker permission whitelist
src/renderer.py            — OpenCV overlay (boxes, labels, zone polygons)
src/event_logger.py        — stderr logging with cooldowns
src/alert_client.py        — HTTP alerts to the backend

config/settings.py         — dataclass-based config for all subsystems
data/zones.json            — zone polygon definitions + allowed_workers
data/sample_videos/        — test video
data/qr_badges/            — pre-generated QR badge images
models/best.pt             — PPE + fall detection model (not in git)
models/wet_floor.pt        — wet floor segmentation model (not in git)

scripts/                   — utilities: zone definition, QR generation, training
tests/                     — pytest suite
```

## Status by subsystem

| Subsystem | Status | Notes |
|-----------|--------|-------|
| PPE detection (helmet/vest) | Working | Temporal smoothing via `PPEComplianceTracker`; alert gated on sustained violation |
| Fall detection | Working | Enabled by default (`FallDetectionConfig.enabled`); severity escalates LOW→MEDIUM→HIGH |
| Wet floor detection | Working | Enabled by default (`WetFloorConfig.enabled`); requires `models/wet_floor.pt` |
| Worker QR identification | Working | Caches per track_id for 30s |
| Zone monitoring | Code complete, disabled by default | Polygon-based, foot-position check + per-worker whitelist; enable via `ZoneConfig.enabled` |
| Alert client | Working | Enabled by default; per-event-type endpoint and cooldown |
| Live display with overlays | Working | Color-coded: green=OK, red=violation, orange=zone |
| Threaded pipeline | Working | Detection runs async from rendering |

## What's Missing

- **Worker name lookup** — `worker_name` is always `None`. QR codes contain raw IDs; needs a lookup table or JSON-encoded QR payload to extract names.
- **No video recording** of annotated output
- **Single camera only** — no multi-camera support
- **No metrics/dashboard** for historical tracking (a separate frontend consumes the alert feed)

## Key Data Structures

### Zone definition (`data/zones.json`)
```json
[{
  "zone_id": "restricted_1",
  "points": [[145,316], [356,358], [188,554], [41,485], [142,316]],
  "allowed_workers": ["<worker-id>", "<worker-id>"]
}]
```

### Pipeline event dict (returned from `process_frame`)
```python
{
  "ppe_violations": [{worker_id, worker_name, track_id, bbox, missing, detected, zone}],
  "compliant_workers": [{worker_id, worker_name, track_id, bbox}],
  "falls": [{worker_id, worker_name, track_id, bbox}],
  "zone_breaches": [{worker_id, worker_name, track_id, bbox, zone_id}],
  "wet_floor_events": [{bbox, confidence, area_pct, consecutive_count, first_seen_ts}],
  "timestamp": float,
}
```

### Worker ID result
```python
{"qr_data": str, "worker_id": str, "worker_name": None}
```

## Configuration (`config/settings.py`)

Deployment-specific values (backend URL, RTSP credentials) come from the environment
— see `.env.example`. Nothing deployment-specific is hardcoded in this file.

## Test Coverage

| File | Coverage |
|------|----------|
| `test_ppe.py` | Overlap logic + integration |
| `test_pipeline.py` | Frame processing |
| `test_worker_id.py` | QR detection |
| `test_zone_monitor.py` | Containment, permissions, pipeline integration |
| `test_renderer.py` | Shape, drawing |
| `test_camera.py` | Source validation |
| `test_fall_detector.py` | Severity gating |
| `test_wet_floor_detector.py` | Detection smoothing |
| `test_ppe_compliance_tracker.py` | Hysteresis window |
| `test_alert_client.py` | Cooldown, per-event endpoints |

## How to Run
```bash
python3 -m src.main
```

## How to Test
```bash
pytest
```
