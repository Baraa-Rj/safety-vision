"""Extract candidate wet-floor frames from a video for annotation.

Samples every Nth frame, deduplicates with perceptual hashing, writes JPGs.

Usage:
    python3 scripts/extract_wet_floor_frames.py \\
        --video data/wet_floor/raw_videos/session1.mp4 \\
        --output-dir data/wet_floor/frames_session1
"""

import argparse
import os
import sys

import cv2
import imagehash
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Extract wet floor candidate frames.")
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument("--output-dir", required=True, help="Directory for extracted frames")
    p.add_argument("--sample-every", type=int, default=5,
                   help="Sample one frame every N frames (default 5)")
    p.add_argument("--hash-threshold", type=int, default=4,
                   help="Hamming distance below which frames are considered duplicates (default 4)")
    p.add_argument("--prefix", type=str, default="wf",
                   help="Filename prefix for extracted frames (default 'wf')")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: video not found at {args.video}", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: cannot open video {args.video}", file=sys.stderr)
        return 1

    total_read = 0
    sampled = 0
    kept = 0
    seen_hashes = []
    output_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_read += 1

        if (total_read - 1) % args.sample_every != 0:
            continue
        sampled += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        phash = imagehash.phash(Image.fromarray(rgb))

        is_dup = any((phash - h) <= args.hash_threshold for h in seen_hashes)
        if is_dup:
            continue
        seen_hashes.append(phash)

        out_path = os.path.join(args.output_dir, f"{args.prefix}_{output_idx:04d}.jpg")
        cv2.imwrite(out_path, frame)
        kept += 1
        output_idx += 1

    cap.release()

    print(f"Total frames read:    {total_read}")
    print(f"Frames sampled:       {sampled}")
    print(f"Frames kept (dedup):  {kept}")
    print(f"Output directory:     {args.output_dir}")

    if kept == 0:
        print("Error: no frames extracted.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
