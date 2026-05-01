# Safety Vision

Real-time PPE compliance monitoring system using computer vision. Detects workers missing safety equipment (helmets, vests), identifies them via QR codes, and sends alerts to a remote server.

## Features

- **PPE Detection** — YOLO-based detection of helmets and vests on workers
- **QR Worker Identification** — Reads QR codes on badges to identify workers by name and ID
- **Zone Monitoring** — Geofencing to detect workers in restricted areas
- **Real-time Alerts** — Sends violation alerts (with frame + worker info) to a remote server via HTTP POST
- **Fall Detection** — YOLO classifier for fallen/standing poses (disabled by default)

## Setup

### Requirements

- Python 3.10+
- OpenCV 4.8+
- Webcam or video file

### Install

```bash
pip install -r requirements.txt
```

### Models

Place YOLO model weights in the `models/` directory:

- `models/best.pt` — PPE detection model (helmet, vest, person)
- `models/fall_classifier/weights/best.pt` — Fall classification model (optional)

### Video Source

Place your video in `data/sample_videos/` or update the source in `config/settings.py`.

## Usage

### Run the pipeline

```bash
python3 src/main.py
```

Press `q` to stop.

### Generate QR badges for workers

Edit the `WORKERS` list in the script, then run:

```bash
python3 scripts/generate_qr_badges.py
```

QR codes encode JSON: `{"id": "W001", "name": "Alice"}`

Print the generated images from `data/qr_badges/` and attach to worker helmets or vests.

### Run tests

```bash
python3 -m pytest tests/ -v
```

## Configuration

All settings are in `config/settings.py`:

| Config | Key fields |
|--------|------------|
| `CameraConfig` | `source`, `startup_delay` |
| `PPEConfig` | `model_path`, `confidence`, `required_ppe`, `overlap_threshold` |
| `FallDetectionConfig` | `classifier_model_path`, `confidence`, `enabled` |
| `DisplayConfig` | `max_display_width`, `window_name` |
| `AlertConfig` | `endpoint`, `enabled`, `cooldown_seconds` |

## Alert Payload

When a PPE violation is detected, a POST request is sent to the configured endpoint:

```json
{
  "userId": "W042",
  "workerName": "John Doe",
  "qrData": "{\"id\": \"W042\", \"name\": \"John Doe\"}",
  "imgImage": "<base64 jpg>",
  "missingItem": "vest",
  "message": "Worker John Doe missing vest"
}
```

## Project Structure

```
├── config/
│   └── settings.py          # All configuration dataclasses
├── src/
│   ├── main.py              # Entry point
│   ├── pipeline.py          # Main pipeline orchestration
│   ├── ppe_detector.py      # YOLO PPE detection
│   ├── fall_detector.py     # YOLO fall classification
│   ├── worker_id.py         # QR code worker identification
│   ├── zone_monitor.py      # Geofence zone monitoring
│   ├── alert_client.py      # HTTP alert client
│   ├── event_logger.py      # Stderr event logging
│   ├── renderer.py          # OpenCV frame rendering
│   └── camera.py            # Threaded camera capture
├── scripts/
│   ├── generate_qr_badges.py
│   ├── extract_crops.py
│   ├── train_classifier.py
│   └── test_alert.py
├── tests/
├── models/                  # YOLO weights (not tracked)
├── data/                    # Videos, images (not tracked)
├── requirements.txt
└── pytest.ini
```
