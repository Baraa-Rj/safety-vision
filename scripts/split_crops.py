"""Split sorted fall crops into a YOLO-cls train/val layout, BY VIDEO.

You sort the extracted crops into just two folders:
    data/crops/sorted/fallen/      data/crops/sorted/standing/
This script then builds:
    data/crops/train/{fallen,standing}/   data/crops/val/{fallen,standing}/

Split is by source video (the `videoN_` filename prefix), never by random
frame. Frames from one clip are near-duplicates, so a random split would put
near-identical images in train and val and inflate the score. Holding out whole
videos gives a val number you can trust. Each class must come from >= 2 videos
so one can be held out; if a class has only one source video the script stops
and tells you (record/relabel another scene for that class).

Usage:
    PYTHONPATH=. python3 scripts/split_crops.py
    PYTHONPATH=. python3 scripts/split_crops.py --src data/crops/sorted --out data/crops --val-videos 1
"""
import argparse
import os
import re
import shutil
from collections import defaultdict

CLASSES = ("fallen", "standing")
VID_RE = re.compile(r"^(video\d+)_")


def video_of(fname):
    m = VID_RE.match(fname)
    return m.group(1) if m else "unknown"


def collect(src, cls):
    d = os.path.join(src, cls)
    if not os.path.isdir(d):
        raise SystemExit(f"missing folder: {d}")
    by_vid = defaultdict(list)
    for f in os.listdir(d):
        if f.lower().endswith((".jpg", ".png")):
            by_vid[video_of(f)].append(os.path.join(d, f))
    if not by_vid:
        raise SystemExit(f"no images in {d}")
    return by_vid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/crops/sorted")
    ap.add_argument("--out", default="data/crops")
    ap.add_argument("--val-videos", type=int, default=1,
                    help="How many source videos per class to hold out for val")
    ap.add_argument("--val-vids", default="",
                    help="Comma list of video ids held out GLOBALLY for val "
                         "(e.g. video5). Applied across all classes so no video "
                         "spans train and val. Overrides --val-videos.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ratio", type=float, default=0.0,
                    help="Cap TRAIN majority class at this multiple of the train "
                         "minority class (e.g. 3 = 3:1). Val is left untouched so "
                         "its class balance reflects reality. 0 = no capping.")
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    forced_val = {v.strip() for v in args.val_vids.split(",") if v.strip()}
    plan = {}  # (split, cls) -> list of file paths
    for cls in CLASSES:
        by_vid = collect(args.src, cls)
        vids = sorted(by_vid)
        if len(vids) < 2:
            raise SystemExit(
                f"class '{cls}' has crops from only {len(vids)} video(s) "
                f"({vids}). Need >= 2 so one can be held out for val. Add crops "
                f"from another scene for '{cls}'.")
        if forced_val:
            val_vids = forced_val & set(vids)
            if not val_vids or val_vids == set(vids):
                raise SystemExit(
                    f"class '{cls}' videos {vids} vs val-vids {sorted(forced_val)}: "
                    f"holdout must keep >=1 video on each side for every class.")
        else:
            rng.shuffle(vids)
            n_val = min(args.val_videos, len(vids) - 1)   # keep >=1 video in train
            val_vids = set(vids[:n_val])
        for v in vids:
            split = "val" if v in val_vids else "train"
            plan.setdefault((split, cls), []).extend(by_vid[v])
        print(f"{cls}: val videos={sorted(val_vids)}  train videos="
              f"{sorted(set(vids) - val_vids)}")

    # Balance the TRAIN split only: cap the majority class at max_ratio x the
    # minority, sampling per source video so scene variety (crouch/bend) is kept.
    if args.max_ratio > 0:
        train_counts = {c: len(plan.get(("train", c), [])) for c in CLASSES}
        minority = min(train_counts.values())
        cap = int(round(args.max_ratio * minority))
        for cls in CLASSES:
            files = plan.get(("train", cls), [])
            if len(files) <= cap:
                continue
            by_vid = defaultdict(list)
            for f in files:
                by_vid[video_of(os.path.basename(f))].append(f)
            keep = []
            for v, fs in by_vid.items():                 # proportional per video
                k = max(1, round(cap * len(fs) / len(files)))
                keep.extend(rng.sample(fs, min(k, len(fs))))
            plan[("train", cls)] = keep
            print(f"train/{cls}: capped {len(files)} -> {len(keep)} (<= {cap})")

    for (split, cls), files in plan.items():
        dst = os.path.join(args.out, split, cls)
        os.makedirs(dst, exist_ok=True)
        for f in files:
            shutil.copy2(f, os.path.join(dst, os.path.basename(f)))

    print("\nCounts:")
    for split in ("train", "val"):
        for cls in CLASSES:
            n = len(plan.get((split, cls), []))
            print(f"  {split}/{cls}: {n}")
    print(f"\nLayout ready under {args.out}/  -> train_classifier_kaggle.py")


if __name__ == "__main__":
    main()
