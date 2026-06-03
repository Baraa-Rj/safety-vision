"""Extract frames from the PPE training videos for Roboflow annotation.

Samples a few frames per second from each video and deduplicates near-identical
frames with perceptual hashing, so you don't waste annotation effort on frames
that barely changed. Output is a single flat folder ready to zip and upload to
Roboflow.

Default inputs are the two new training clips:
    data/sample_videos/training.mp4
    data/sample_videos/training1.mp4

Usage:
    PYTHONPATH=. python3 scripts/extract_training_frames.py
    # or point at specific videos / tune sampling:
    PYTHONPATH=. python3 scripts/extract_training_frames.py \\
        --videos data/sample_videos/training.mp4 data/sample_videos/training1.mp4 \\
        --output-dir data/training_frames --per-second 2 --hash-threshold 4

Then:
    1. Zip the output:   cd data && zip -r training_frames.zip training_frames
    2. Upload training_frames.zip to a Roboflow project, annotate
       helmet / person / vest  (label LIME vests as the SAME 'vest' class).
    3. Export YOLO format from Roboflow, then prepare + upload to Kaggle
       (see scripts/prepare_dataset.py and scripts/train_ppe_kaggle.py).
"""

import argparse
import os
import sys

import cv2
import imagehash
from PIL import Image

DEFAULT_VIDEOS = [
    "data/sample_videos/training.mp4",
    "data/sample_videos/training1.mp4",
]


def parse_args():
    p = argparse.ArgumentParser(description="Extract PPE training frames for annotation.")
    p.add_argument("--videos", nargs="+", default=DEFAULT_VIDEOS,
                   help="Input video paths (default: the two training clips)")
    p.add_argument("--output-dir", default="data/training_frames",
                   help="Directory for extracted frames (default data/training_frames)")
    p.add_argument("--per-second", type=float, default=1.0,
                   help="Frames to keep per second of video, before dedup (default 1)")
    p.add_argument("--hash-threshold", type=int, default=4,
                   help="Hamming distance below which frames are dropped as duplicates "
                        "(default 4; set 0 or use --no-dedup to keep everything)")
    p.add_argument("--no-dedup", action="store_true",
                   help="Disable perceptual-hash deduplication")
    p.add_argument("--clean", action="store_true",
                   help="Delete existing frames in the output dir before extracting")
    return p.parse_args()


def extract_one(video_path, output_dir, per_second, hash_threshold, dedup, seen_hashes):
    """Extract frames from a single video. Returns (read, sampled, kept)."""
    name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  cannot open {video_path}, skipping", file=sys.stderr)
        return 0, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps / per_second)) if per_second > 0 else 1

    read = sampled = kept = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        read += 1

        if idx % step != 0:
            idx += 1
            continue
        idx += 1
        sampled += 1

        if dedup:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            phash = imagehash.phash(Image.fromarray(rgb))
            if any((phash - h) <= hash_threshold for h in seen_hashes):
                continue
            seen_hashes.append(phash)

        # frame index in the filename keeps ordering and avoids collisions
        out_path = os.path.join(output_dir, f"{name}_{(idx - 1):06d}.jpg")
        cv2.imwrite(out_path, frame)
        kept += 1

    cap.release()
    print(f"  {name}: fps={fps:.1f} step={step}  sampled={sampled} kept={kept}")
    return read, sampled, kept


def main():
    args = parse_args()

    missing = [v for v in args.videos if not os.path.isfile(v)]
    if missing:
        for v in missing:
            print(f"Error: video not found at {v}", file=sys.stderr)
        return 1

    if args.clean and os.path.isdir(args.output_dir):
        for f in os.listdir(args.output_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                os.remove(os.path.join(args.output_dir, f))
        print(f"Cleaned existing frames in {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    dedup = not args.no_dedup
    # Shared hash list across videos so duplicates spanning both clips are dropped.
    seen_hashes = []
    totals = [0, 0, 0]

    print(f"Extracting to {args.output_dir} "
          f"(per_second={args.per_second}, dedup={'on' if dedup else 'off'})")
    for video in args.videos:
        r, s, k = extract_one(video, args.output_dir, args.per_second,
                              args.hash_threshold, dedup, seen_hashes)
        totals = [totals[0] + r, totals[1] + s, totals[2] + k]

    print(f"\nTotal: read={totals[0]} sampled={totals[1]} kept={totals[2]}")
    print(f"Frames in: {args.output_dir}")
    print(f"\nNext: zip and upload to Roboflow:")
    print(f"  cd {os.path.dirname(args.output_dir) or '.'} && "
          f"zip -r {os.path.basename(args.output_dir)}.zip {os.path.basename(args.output_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
