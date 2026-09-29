# Safety Vision

Real-time worksite safety monitoring with computer vision. A single camera feed is
analysed for PPE compliance, worker identity, restricted-zone entry, and falls, with
violations pushed to a backend over HTTP.

## Features

- **PPE detection** — a YOLO detector locates each worker and their helmet / vest; a
  missing item is flagged only after a sustained, time-confirmed violation, so
  single-frame false pops are filtered out.
- **Worker identification** — QR badge reading maps a tracked person to a worker ID/name,
  cached across frames via a persistent tracker so identity survives movement.
- **Zone monitoring** — polygon geofences raise a breach when an unauthorised worker
  enters (disabled by default; enable in `config/settings.py`).
- **Fall detection** — the detector has a dedicated `fallen` class, so a worker on the
  ground is detected directly (no crop, no separate classifier). A fall is reported only
  after `fallen` is sustained for several consecutive frames on the same track, with
  box-geometry checks and severity that escalates (LOW → MEDIUM → HIGH) while the worker
  stays down.
- **Alerts** — PPE, zone, and fall events POST to a backend. Identified workers send only
  their ID; unidentified workers attach the frame for review. Each event type has its own
  endpoint and is cooldown-gated so the backend isn't spammed.

## Architecture

```
camera ─▶ PPEDetector ─┬─▶ WorkerIdentifier (QR)
   │                   ├─▶ FallDetector (fallen-class gate)
   │                   ├─▶ ZoneMonitor
   │                   └─▶ ComplianceTracker / ViolationConfirmer
   ▼
SafetyPipeline ─▶ FrameRenderer (display)
               └▶ AlertClient ─▶ backend (HTTP)
```

Capture and detection run on separate threads: display stays smooth at camera FPS while
inference runs as fast as the CPU allows, and the renderer holds the last known events
when a frame momentarily has no detections.

| Layer | Modules |
|---|---|
| Entry point | `src/main.py` |
| Orchestration | `src/pipeline.py` |
| Detection | `src/ppe_detector.py`, `src/fall_detector.py`, `src/wet_floor_detector.py` |
| Identity / zones | `src/worker_id.py`, `src/zone_monitor.py`, `src/zone_client.py` |
| Smoothing | `src/ppe_compliance_tracker.py`, `src/violation_confirmer.py` |
| Output | `src/renderer.py`, `src/event_logger.py`, `src/alert_client.py` |
| Config | `config/settings.py` |

## Setup

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
cp .env.example .env        # then edit .env with your values
```

### Models

Model weights are not tracked in git. Place them in `models/`:

- `models/best.pt` — detector (helmet, person, vest, fallen); drives both PPE compliance
  and fall detection

### Configuration

Secrets and deployment-specific values come from the environment (see `.env.example`);
everything else lives in `config/settings.py`.

| Variable | Purpose | Default |
|---|---|---|
| `CAMERA_RTSP_URL` | Live RTSP stream — keeps credentials out of git | bundled sample video |
| `BACKEND_URL` | Base URL; PPE & zone endpoints derive from it | `http://localhost:8080` |
| `PPE_ALERTS_ENDPOINT` | Override the PPE alert URL | derived from `BACKEND_URL` |
| `ZONES_ENDPOINT` | Override the zone upload URL | derived from `BACKEND_URL` |

The camera layer auto-detects RTSP URLs and applies TCP transport, a 1-frame buffer to
stay at the live edge, and auto-reconnect if reads stall beyond
`CameraConfig.reconnect_after_seconds`.

## Run

From the repository root:

```bash
python3 -m src.main
```

Press `q` to stop. With no `CAMERA_RTSP_URL` set, it runs against the sample video.

### Generate QR badges

Edit the `WORKERS` list in the script, then:

```bash
python3 scripts/generate_qr_badges.py
```

QR codes encode `{"id": "W001", "name": "Alice"}`; print from `data/qr_badges/` and attach
to helmets or vests.

## Tests

```bash
pytest                    # unit tests; integration tests are deselected by default
pytest -m integration     # needs the sample video and models/best.pt (see below)
```

The integration tests read the default sample video, `data/sample_videos/0712.mp4`, and
the detector weights, `models/best.pt`. Neither is in git; copy them into place first.

## Repository layout

```
config/   pipeline configuration and tracker settings
src/       pipeline, detectors, alerting, rendering
scripts/   dataset prep, training, and utility tools
tests/     unit tests
docs/      task spec and project notes
data/      videos, frames, datasets (gitignored)
models/    model weights (gitignored)
```

## License

MIT — see [LICENSE](LICENSE).
