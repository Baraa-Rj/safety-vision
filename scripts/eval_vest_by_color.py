"""Per-color vest recall check.

Answers the only question that matters after a retrain: does the model
actually fire on LIME vests, not just orange? Standard mAP averages colors
together and hides a per-color blind spot — this measures each color directly.

Build a holdout set by hand (images NOT used in training), one folder per
color, each image containing a clearly visible vest of that color:

    data/vest_holdout/orange/*.jpg
    data/vest_holdout/lime/*.jpg

This is a PRESENCE recall (no ground-truth boxes needed): for each image it
asks "did the model detect any vest?". It tells you whether the model sees a
color at all — which is exactly the failure you're chasing. It does not measure
box quality; use `yolo val` against the labeled test split for mAP.

Usage:
    PYTHONPATH=. python3 scripts/eval_vest_by_color.py \
        --model models/best.pt --holdout data/vest_holdout --conf 0.20
"""

import argparse
import os

from ultralytics import YOLO

VEST_CLASS = "vest"
_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/best.pt")
    ap.add_argument("--holdout", required=True,
                    help="dir with one subfolder per vest color")
    ap.add_argument("--conf", type=float, default=0.20,
                    help="vest confidence floor (match config.ppe.class_confidences)")
    args = ap.parse_args()

    model = YOLO(args.model)
    names = model.names

    print(f"\nmodel={args.model}  conf>={args.conf}\n")
    print(f"{'color':<12}{'images':>8}{'detected':>10}{'recall':>9}")
    print("-" * 39)

    overall_imgs = overall_hits = 0
    for color in sorted(os.listdir(args.holdout)):
        cdir = os.path.join(args.holdout, color)
        if not os.path.isdir(cdir):
            continue
        imgs = [f for f in os.listdir(cdir) if f.lower().endswith(_EXTS)]
        if not imgs:
            continue

        hits = 0
        for fn in imgs:
            res = model(os.path.join(cdir, fn), conf=args.conf, verbose=False)[0]
            detected = {names[int(b.cls[0])] for b in res.boxes}
            if VEST_CLASS in detected:
                hits += 1

        recall = hits / len(imgs)
        flag = "  <-- BLIND SPOT" if recall < 0.8 else ""
        print(f"{color:<12}{len(imgs):>8}{hits:>10}{recall:>9.2f}{flag}")
        overall_imgs += len(imgs)
        overall_hits += hits

    if overall_imgs:
        print("-" * 39)
        print(f"{'ALL':<12}{overall_imgs:>8}{overall_hits:>10}"
              f"{overall_hits / overall_imgs:>9.2f}")


if __name__ == "__main__":
    main()
