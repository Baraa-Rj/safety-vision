# Fall detection — task brief

Sub-task: add fall detection to the Python AI pipeline. The detector and training
code already exist under `fall_detection/`. The remaining job is to collect data,
run, evaluate, and integrate — not to redesign.

## Approach (decided — do not revert)
- **Two separate models.** `best.pt` (existing PPE detector — 3 classes
  `helmet`/`person`/`vest`, Ultralytics Detect head) is used **only** as the
  person-detection + tracking front-end. It is **not** retrained for falls.
- The fall decision is a **separate small temporal model** (`FallTCN`, a dilated
  1D-CNN) that classifies a ~1.5 s window of per-track motion features as
  `fall` / `no_fall`.
- **Temporal, not single-frame.** A single-frame fall detector/classifier learns
  posture appearance and fires on anyone sitting or lying down. Rejected. Do not
  "simplify" back to a YOLO fall class or a `-cls` model.
- **Why feature-sequence over pose:** single fixed overhead/oblique camera; pose
  keypoints degrade at that distance (same reason gloves/glasses were deferred).
  Bbox-motion features are distance-robust and data-efficient.

## Files (`fall_detection/`)
- `features.py` — 6 per-frame motion features (cy_norm, aspect, h_norm,
  v_centroid, v_bottom, disp) + `TrackBuffer`. Shared by extraction and inference
  so features are identical on both sides.
- `extract_dataset.py` — runs `best.pt` tracking over a video folder, slides
  windows over each person track, labels from `labels.csv`, writes `dataset.npz`
  grouped by video.
- `model.py` — `FallTCN` + checkpoint save/load (bundles normalization stats,
  window length, decision threshold).
- `train.py` — scene-level (by-video) split, standardize from train only,
  class-weighted training, threshold pick at FPR ≤ 0.07, prints held-out
  sensitivity/FPR, saves `fall_model.pt`.
- `fall_detector.py` — `FallDetector`: best.pt tracking → per-track window →
  FallTCN → confirm gate → emits `DetectionResult` to
  `AlertManager.processDetection`.
- `labels.example.csv`, `README.md`.

## Data flow
best.pt person box (per frame) → ByteTrack track id → `TrackBuffer` (window) →
standardize → `FallTCN` → prob → threshold + N-consecutive confirm gate +
per-track cooldown → `DetectionResult(FALL_DETECTED)` → `AlertManager`.

## Status
Code written, compiles. **Not yet run** — no fall footage collected yet.

## Ordered next steps
1. **Collect staged footage** from the deployed camera (same session as wet
   floor): fall types (forward/back/side/slump/slip) + hard negatives (bending,
   crouch, sit, kneel, lie-down, squat). Vary distance from camera.
   ~30–50 falls + a larger negative pool.
2. **Sanity-check `best.pt` person recall on fallen/horizontal bodies.** If weak,
   add fallen-person frames and lightly retrain `best.pt` — person detection
   only, do not add fall classes.
3. **Annotate** fall frame ranges in `labels.csv` (see example).
4. **Build dataset:**
   `python extract_dataset.py --videos ./videos --weights best.pt --labels labels.csv --out dataset.npz --window <≈1.5s in frames>`
5. **Train:**
   `python train.py --data dataset.npz --out fall_model.pt --fpr-target 0.07`
   Confirm held-out sensitivity ≥ 0.85, FPR ≤ 0.07 (SR2.4).
6. **Integrate** `FallDetector` into the live pipeline with the real
   `AlertManager`; map tracker id → QR worker id where the PPE path does
   (`-1` = UNIDENTIFIED).
7. **Tune** confirm / cooldown / threshold against measured FPR.

## Constraints / gotchas (respect)
- Resolve the `person` class **by name** from `model.names`; never hardcode an
  index.
- **Never random-split windows** — split by video. Windows from one clip are
  correlated; random splits leak and inflate metrics (same issue as the PPE
  pipeline).
- Standardization stats come from **train only**.
- **SR2.4:** sensitivity ≥ 85%, FPR ≤ 7% at 1080p. Define the FPR denominator
  explicitly (false alarms per hour of normal activity) — the SR leaves it open.
- **SR2.1** ≥ 30 FPS and **SR2.2** alert ≤ 5 s (location, snapshot, live-feed
  link). The ~2 s confirm window fits inside the 5 s budget; FallTCN is tiny and
  runs alongside detection.
- `DetectionResult` is a stand-in; swap for the pipeline's real class. Field
  names already match the class diagram (type, workerId, zoneId, confidence,
  timestamp).
- Known weak spot to own (not hide): a fall directly toward/away from the camera
  produces little vertical pixel motion and may be missed; the stillness signal
  still catches most because the worker stays down.

## Conventions
YOLO26 (Ultralytics), PyTorch, ByteTrack. Inline comments only where non-obvious.
Prioritize working over clean; do not refactor or over-engineer unprompted.
