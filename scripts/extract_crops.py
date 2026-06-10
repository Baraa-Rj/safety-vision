"""Extract person crops from fall videos for the fall classifier.

Runs the person detector over each video and saves person crops to
data/crops/unlabeled/. You then MANUALLY sort them into fallen/ and standing/.

Usage:
    PYTHONPATH=. python3 scripts/extract_crops.py --videos data/fall_videos/*.mp4
    # or default to the configured camera source:
    PYTHONPATH=. python3 scripts/extract_crops.py

After running, sort data/crops/unlabeled/ into:
    data/crops/train/fallen/   data/crops/train/standing/
    data/crops/val/fallen/     data/crops/val/standing/   (~20% held out)
Then train: scripts/train_classifier_kaggle.py (on Kaggle GPU).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2

from config.settings import PipelineConfig
from src.ppe_detector import PPEDetector


def parse_args():
    p = argparse.ArgumentParser(description="Extract person crops for the fall classifier.")
    p.add_argument("--videos", nargs="+", default=None,
                   help="Video files (default: the configured camera source)")
    p.add_argument("--output-dir", default="data/crops/unlabeled")
    p.add_argument("--sample-every", type=int, default=5,
                   help="Take one frame every N frames (default 5)")
    p.add_argument("--min-size", type=int, default=40,
                   help="Skip crops smaller than this many px on a side")
    return p.parse_args()


def main():
    args = parse_args()
    config = PipelineConfig()
    videos = args.videos or [config.camera.source]

    detector = PPEDetector(
        config.ppe.model_path, confidence=config.ppe.confidence,
        required_ppe=config.ppe.required_ppe,
        overlap_threshold=config.ppe.overlap_threshold,
        class_confidences=config.ppe.class_confidences,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    total = 0

    for video_path in videos:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open {video_path}, skipping", file=sys.stderr)
            continue
        name = os.path.splitext(os.path.basename(video_path))[0]
        frame_idx = saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % args.sample_every != 0:
                continue
            for i, r in enumerate(detector.detect(frame)):
                x1, y1, x2, y2 = r["person_bbox"]
                if (x2 - x1) < args.min_size or (y2 - y1) < args.min_size:
                    continue
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue
                out = os.path.join(args.output_dir, f"{name}_{frame_idx:05d}_p{i}.jpg")
                cv2.imwrite(out, crop)
                saved += 1
        cap.release()
        print(f"{name}: {saved} crops")
        total += saved

    print(f"\nDone. {total} crops in {args.output_dir}/")
    print("Next: sort into data/crops/{train,val}/{fallen,standing}/, then train.")


if __name__ == "__main__":
    main()
