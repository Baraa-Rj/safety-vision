"""Scrub fall videos and mark fall start/end frames -> labels.csv.

No frame dumping, no Roboflow. Step through each clip, press 's' at the frame
the fall begins and 'e' at the frame it ends; a row is appended to labels.csv.
Mark multiple falls per clip by repeating s/e. Clips you never mark stay
all-negative automatically.

Usage:
    PYTHONPATH=. python3 scripts/label_falls.py --videos data/fall_videos/*.mp4
    PYTHONPATH=. python3 scripts/label_falls.py --videos data/fall_videos/fall_01.mp4 --labels labels.csv

Controls (focus the video window):
    d / a      next / previous frame
    f / b      forward / back 10 frames
    space      play / pause
    s          mark FALL START at current frame
    e          mark FALL END at current frame -> writes the row
    u          undo the pending start
    n          next video
    q          quit (saves)
"""

import argparse
import csv
import glob
import os

import cv2

MAX_W = 1280


def parse_args():
    p = argparse.ArgumentParser(description="Mark fall start/end frames into labels.csv")
    p.add_argument("--videos", nargs="+", required=True, help="Video files (globs ok)")
    p.add_argument("--labels", default="labels.csv")
    return p.parse_args()


def append_row(path, video, start, end):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["video", "fall_start", "fall_end", "track_id"])
        w.writerow([video, start, end, ""])


def label_video(path, labels_path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"cannot open {path}, skipping")
        return True
    name = os.path.basename(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    win = "label_falls"
    cv2.namedWindow(win)

    idx, playing, pending_start = 0, False, None
    while True:
        idx = max(0, min(idx, total - 1)) if total else max(0, idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            idx = max(0, idx - 1)
            playing = False
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break

        disp = frame.copy()
        h, w = disp.shape[:2]
        if w > MAX_W:
            disp = cv2.resize(disp, (MAX_W, int(h * MAX_W / w)))
        ps = f"START={pending_start}" if pending_start is not None else "START=-"
        cv2.putText(disp, f"{name}  frame {idx}/{max(total-1,0)}  fps~{fps:.0f}  {ps}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(disp, "d/a step  f/b x10  space play  s start  e end  u undo  n next  q quit",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(win, disp)

        key = cv2.waitKey(int(1000 / fps) if playing else 0) & 0xFF
        if playing and key == 255:          # no key during playback
            idx += 1
            if total and idx >= total - 1:
                playing = False
            continue

        if key == ord('q'):
            cv2.destroyWindow(win)
            cap.release()
            return False                     # signal: stop entirely
        elif key == ord('n'):
            break
        elif key == ord(' '):
            playing = not playing
        elif key == ord('d'):
            idx += 1
        elif key == ord('a'):
            idx -= 1
        elif key == ord('f'):
            idx += 10
        elif key == ord('b'):
            idx -= 10
        elif key == ord('s'):
            pending_start = idx
            print(f"  {name}: start={idx}")
        elif key == ord('u'):
            pending_start = None
            print("  pending start cleared")
        elif key == ord('e'):
            if pending_start is None:
                print("  press 's' to set a start first")
            elif idx < pending_start:
                print("  end is before start — ignored")
            else:
                append_row(labels_path, name, pending_start, idx)
                print(f"  WROTE: {name},{pending_start},{idx}")
                pending_start = None

    cv2.destroyWindow(win)
    cap.release()
    return True


def main():
    args = parse_args()
    videos = []
    for v in args.videos:
        videos.extend(sorted(glob.glob(v)))
    if not videos:
        raise SystemExit("No videos matched.")
    print(f"Labeling {len(videos)} video(s) -> {args.labels}\n")
    for v in videos:
        if not label_video(v, args.labels):
            break
    print(f"\nDone. labels.csv: {args.labels}")


if __name__ == "__main__":
    main()
