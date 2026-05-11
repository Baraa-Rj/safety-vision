"""Extract frames from all training videos for annotation.

Usage:
    PYTHONPATH=. python3 scripts/extract_frames.py

Clears data/frames/, then extracts 1 frame per second from each video
in /home/dark/Downloads/videos/.
"""

import glob
import os
import shutil
import sys

import cv2

VIDEOS_DIR = "/home/dark/Downloads/videos"
OUTPUT_DIR = "data/frames"

# Clear existing frames
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

videos = sorted(glob.glob(os.path.join(VIDEOS_DIR, "*.mp4")))
if not videos:
    print(f"No .mp4 files found in {VIDEOS_DIR}")
    sys.exit(1)

total_count = 0

for video_path in videos:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open {video_path}, skipping")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(fps))
    count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            filename = os.path.join(OUTPUT_DIR, f"{video_name}_{frame_idx:05d}.jpg")
            cv2.imwrite(filename, frame)
            count += 1

        frame_idx += 1

    cap.release()
    print(f"{video_name}: {count} frames extracted")
    total_count += count

print(f"\nTotal: {total_count} frames saved to {OUTPUT_DIR}/")
