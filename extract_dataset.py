"""Build the windowed fall-detection dataset from staged videos.

Runs best.pt person detection + tracking over every video, slides a fixed
window over each person track, labels each window from labels.csv, and writes
X / y / groups to one .npz. `groups` is the source video name so train.py can
split by scene instead of by window (avoids leakage from correlated frames).

labels.csv format (header required):
    video,fall_start,fall_end,track_id
  - video:      filename only, e.g. fall_clip_03.mp4
  - fall_start: frame index where the fall transition begins
  - fall_end:   frame index where the fall transition ends
  - track_id:   optional; if set, only that track is labeled fall. Leave blank
                when the clip has a single staged faller.
Videos absent from labels.csv are treated as all-negative.

Usage:
    python extract_dataset.py --videos ./videos --weights best.pt \
        --labels labels.csv --out dataset.npz --window 45 --stride 8
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from ultralytics import YOLO

from features import TrackBuffer, N_FEATURES


def load_labels(path):
    """Returns {video: [(start, end, track_id_or_None), ...]}."""
    spans = defaultdict(list)
    if not path or not os.path.exists(path):
        return spans
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            tid = row.get("track_id", "").strip()
            spans[row["video"].strip()].append(
                (int(row["fall_start"]), int(row["fall_end"]),
                 int(tid) if tid else None)
            )
    return spans


def window_label(video_spans, track_id, center_frame):
    for start, end, tid in video_spans:
        if tid is not None and tid != track_id:
            continue
        if start <= center_frame <= end:
            return 1
    return 0


def resolve_person_class(model):
    for k, v in model.names.items():
        if str(v).lower() == "person":
            return int(k)
    raise ValueError(f"No 'person' class in model.names: {model.names}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--out", default="dataset.npz")
    ap.add_argument("--window", type=int, default=45)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-gap", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    model = YOLO(args.weights)
    person_cls = resolve_person_class(model)

    vids = sorted(
        f for f in os.listdir(args.videos)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".dav"))
    )
    if not vids:
        raise SystemExit(f"No videos found in {args.videos}")

    X, y, groups = [], [], []

    for vid in vids:
        spans = labels.get(vid, [])
        buffers, last_emit = {}, {}
        results = model.track(
            source=os.path.join(args.videos, vid),
            stream=True, persist=True, conf=args.conf,
            tracker=args.tracker, classes=[person_cls], verbose=False,
        )
        for frame_idx, r in enumerate(results):
            frame_h = r.orig_shape[0]
            if r.boxes is None or r.boxes.id is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            for box, tid in zip(xyxy, ids):
                buf = buffers.setdefault(tid, TrackBuffer(args.window, args.max_gap))
                buf.update(tuple(box), frame_idx, frame_h)
                if not buf.ready():
                    continue
                if frame_idx - last_emit.get(tid, -10**9) < args.stride:
                    continue
                last_emit[tid] = frame_idx
                center = frame_idx - args.window // 2
                X.append(buf.array())
                y.append(window_label(spans, tid, center))
                groups.append(vid)

        pos = sum(1 for g, lbl in zip(groups, y) if g == vid and lbl == 1)
        tot = sum(1 for g in groups if g == vid)
        print(f"{vid}: {tot} windows ({pos} fall)")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    assert X.shape[1:] == (args.window, N_FEATURES), X.shape
    np.savez_compressed(args.out, X=X, y=y, groups=groups,
                        window=args.window, n_features=N_FEATURES)
    print(f"\nSaved {len(y)} windows ({int(y.sum())} fall / {len(y) - int(y.sum())} neg) "
          f"across {len(set(groups))} videos -> {args.out}")


if __name__ == "__main__":
    main()
