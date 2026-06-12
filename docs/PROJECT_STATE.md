# Safety-Vision — Project State

## Purpose
Real-time computer vision system for industrial/construction safety monitoring. Detects PPE compliance (helmets, vests), identifies workers via QR badges, monitors restricted zones with per-worker permissions, and sends alerts to a remote backend.

## Tech Stack
- **Python 3.10**, OpenCV, Ultralytics (YOLO26 weights)
- **Detection:** YOLO models for PPE detection and (planned) fall classification
- **Identification:** OpenCV QR code detector
- **Backend integration:** HTTP POST alerts with base64-encoded frames
- **Tests:** pytest

## Architecture

```
src/main.py                — entry point, wires everything together
src/pipeline.py            — orchestrator, threaded detection + render loop
src/camera.py              — threaded video capture (file or webcam)
src/ppe_detector.py        — YOLO-based PPE + person detection with tracking
src/fall_detector.py       — YOLO classifier for fall detection (disabled)
src/worker_id.py           — QR code scanning with per-track caching
src/zone_monitor.py        — polygon zones with per-worker permission whitelist
src/renderer.py            — OpenCV overlay (boxes, labels, zone polygons)
src/event_logger.py        — stderr logging with cooldowns
src/alert_client.py        — HTTP alerts (disabled in main.py)

config/settings.py         — dataclass-based config for all subsystems
data/zones.json            — zone polygon definitions + allowed_workers
data/sample_videos/        — test video
data/qr_badges/            — pre-generated QR badge images
models/best.pt             — PPE detection model (not in git)

scripts/                   — utilities: zone definition, QR generation, training
tests/                     — pytest suite (~25 tests)
```

## What's Working

| Subsystem | Status | Notes |
|-----------|--------|-------|
| PPE detection (helmet/vest) | Working with temporal smoothing | YOLO tracking, configurable threshold; alert events gated by `PPEComplianceTracker` |
| Worker QR identification | Working | Caches per track_id for 30s |
| Zone monitoring | Working | Polygon-based, foot-position check |
| **Zone permissions (NEW)** | Working | Per-worker whitelist via `allowed_workers` |
| Live display with overlays | Working | Color-coded: green=OK, red=violation, orange=zone |
| Event logging | Working | 15s cooldown per event type |
| Threaded pipeline | Working | Detection runs async from rendering |

## What's Built but Disabled

| Subsystem | Status | What's needed |
|-----------|--------|---------------|
| Alert client | Code complete, `alert_client=None` in main.py | Instantiate `AlertClient`, set `AlertConfig.enabled=True` |
| Fall detection | Code complete, `fall_detector=None` in main.py | Train classifier model (no training data yet) |
| Wet floor detection | Code complete, `WetFloorConfig.enabled=False` | Awaiting trained model. Code path is ready — set `WetFloorConfig.enabled=True` and provide `models/wet_floor.pt` |

## What's Missing

- **Worker name lookup** — `worker_name` is always `None`. QR codes contain raw IDs (e.g. `"0cb42"`); needs a lookup table or JSON-encoded QR payload to extract names.
- **No tests** for `AlertClient` and `FallDetector`
- **No video recording** of annotated output
- **Single camera only** — no multi-camera support
- **No metrics/dashboard** for historical tracking
- **Wet floor detection model** — scaffolding shipped; needs training data + `models/wet_floor.pt`

## Recent Work

**Temporal smoothing layer for PPE compliance** (just completed):
- Added `PPEComplianceTracker` (`src/ppe_compliance_tracker.py`) — per-(track_id, item) rolling window with hysteresis.
- Single-frame detection noise no longer causes missing-PPE alerts; alerts require sustained absence over a configurable rolling window with hysteresis on clearing.
- Pipeline now emits both `ppe_violations` (instantaneous, used by renderer for per-frame red boxes) and `confirmed_ppe_violations` (smoothed, used by alert client and event_logger).
- Untracked detections (`track_id is None`) bypass the smoother — safety-first fallback.
- New `ComplianceConfig` in `config/settings.py`. 9 new unit tests in `tests/test_ppe_compliance_tracker.py`.

**Wet floor detection scaffolding** (earlier):
- New `WetFloorDetector` in `src/wet_floor_detector.py` (lazy model load, single warning when model missing, returns `[{bbox, confidence, area_pct}]`)
- `WetFloorConfig` added to `config/settings.py` (disabled by default)
- Pipeline runs detection alongside PPE/fall and emits `wet_floor_events` after N consecutive frames with IoU≥0.5 against the last detection
- Frame extraction CLI `scripts/extract_wet_floor_frames.py` (perceptual-hash dedup via `imagehash`)
- `data/wet_floor/` directory skeleton (raw_videos, frames_session1, curated/{positives,negatives,holdout_test}, public) tracked via `.gitkeep`
- New tests in `tests/test_wet_floor_detector.py` (5 tests, all mocked — no model file needed)

**Zone-based worker permissions** (earlier):
- Each zone in `data/zones.json` now has an `allowed_workers: ["id1", "id2"]` list
- `ZoneMonitor.is_permitted(zone_id, worker_id)` gates breach events in `pipeline.py`
- Permitted workers in a zone → no breach event
- Unidentified workers (no QR scanned) → always trigger breach (safety-first)
- 5 new tests in `tests/test_zone_monitor.py`

**Zone loading from JSON** (earlier):
- `main.py` loads `data/zones.json` on startup and populates `ZoneMonitor` with polygons + permissions

## Key Data Structures

### Zone definition (`data/zones.json`)
```json
[{
  "zone_id": "restricted_1",
  "points": [[145,316], [356,358], [188,554], [41,485], [142,316]],
  "allowed_workers": ["0cb42", "98c39"]
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

- `CameraConfig.source` — default `"data/sample_videos/full-vest-with-QR.mp4"`
- `PPEConfig` — model path, confidence 0.35, required PPE = `{"helmet", "vest"}`
- `FallDetectionConfig.enabled = False`
- `AlertConfig.enabled = False`, endpoint `http://203.0.113.10:8080/api/ppe-alerts`
- `ZoneConfig.cooldown_seconds = 30.0`

## Test Coverage

| File | Tests | Coverage |
|------|-------|----------|
| `test_ppe.py` | 6 | Overlap logic + integration |
| `test_pipeline.py` | 3 | Frame processing |
| `test_worker_id.py` | 4 | QR detection |
| `test_zone_monitor.py` | 11 | Containment, permissions, pipeline integration |
| `test_renderer.py` | 2 | Shape, drawing |
| `test_camera.py` | 2 | Source validation |
| `test_alert_client.py` | — | **Missing** |
| `test_fall_detector.py` | — | **Missing** |

## Known Worker IDs (from `scripts/generate_qr_badges.py`)
- `0cb42` — worker_a
- `98c39` — worker_b
- `c52a4` — Worker D
- `2678a` — worker_c
- `7c2ea` — Worker E

## Next Steps Under Discussion

**Wet floor detection** — proposed approach:
- Train a YOLOv8 detection model for a new `wet_floor` class
- Follow existing `PPEDetector` pattern: new `WetFloorDetector` class with `detect(frame)` returning bounding boxes
- Integrate into pipeline alongside PPE/fall detection
- Add wet-floor events to the alert flow
- **Blocker:** needs labeled dataset (500-1000+ images of wet/dry floors with bounding boxes). No data collected yet.

## How to Run
```bash
python3 src/main.py
```

## How to Test
```bash
python3 -m pytest tests/ -v
```
