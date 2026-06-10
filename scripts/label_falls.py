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


def parse_args():
    p = argparse.ArgumentParser(description="Mark fall start/end frames into labels.csv")
    p.add_argument("--videos", nargs="+", required=True, help="Video files (globs ok)")
    p.add_argument("--labels", default="labels.csv")
    p.add_argument("--max-width", type=int, default=900,
                   help="Display width in px; the frame is scaled to fit (default 900)")
    return p.parse_args()


def append_row(path, video, start, end):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["video", "fall_start", "fall_end", "track_id"])
        w.writerow([video, start, end, ""])


def label_video(path, labels_path, max_w):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"cannot open {path}, skipping")
        return True
    name = os.path.basename(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    win = "label_falls"
    cv2.namedWindow(win)   # AUTOSIZE: window fits the scaled frame exactly

    # Seek bar. The callback fires while OpenCV processes window events during
    # waitKey; `guard` suppresses the echo when we set the position ourselves.
    nav = {"target": None, "guard": False}

    def on_seek(pos):
        if not nav["guard"]:
            nav["target"] = pos

    if total:
        cv2.createTrackbar("seek", win, 0, max(total - 1, 1), on_seek)

    idx, playing, pending_start = 0, False, None
    while True:
        idx = max(0, min(idx, total - 1)) if total else max(0, idx)
        if total:
            nav["guard"] = True
            cv2.setTrackbarPos("seek", win, idx)
            nav["guard"] = False
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
        if w > max_w:
            disp = cv2.resize(disp, (max_w, int(h * max_w / w)))
        ps = f"START={pending_start}" if pending_start is not None else "START=-"
        cv2.putText(disp, f"{name}  frame {idx}/{max(total-1,0)}  fps~{fps:.0f}  {ps}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(disp, "drag bar to seek | a/d 1  b/f 10  ,/. 50  [/] 100  space play  s start  e end  u undo  n next  q quit",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(win, disp)

        key = cv2.waitKey(int(1000 / fps) if playing else 0) & 0xFF

        if nav["target"] is not None:        # user dragged the seek bar
            idx = nav["target"]
            nav["target"] = None
            playing = False
            continue

        if playing and key == 255:           # no key during playback
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
        elif key == ord('.'):
            idx += 50
        elif key == ord(','):
            idx -= 50
        elif key == ord(']'):
            idx += 100
        elif key == ord('['):
            idx -= 100
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
        if not label_video(v, args.labels, args.max_width):
            break
    print(f"\nDone. labels.csv: {args.labels}")


if __name__ == "__main__":
    main()
