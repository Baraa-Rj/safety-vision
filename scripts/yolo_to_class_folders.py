"""Convert a Roboflow YOLO detection export of single-person crops into
classification folders (one class per image = its largest box's class).

Each crop holds one subject person; if a label file has several boxes (a
bystander clipped the edge), the largest-area box decides the image's class.
Empty/missing label files are skipped (unlabeled crop).

Usage:
    PYTHONPATH=. python3 scripts/yolo_to_class_folders.py \
        --export data/crops/_roboflow --out data/crops/sorted
"""
import argparse
import glob
import os
import shutil

import yaml


def class_of(label_path):
    best_cls, best_area = None, -1.0
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            cls = int(float(p[0]))
            area = float(p[3]) * float(p[4])   # w * h (normalized)
            if area > best_area:
                best_area, best_cls = area, cls
    return best_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="data/crops/_roboflow")
    ap.add_argument("--out", default="data/crops/sorted")
    args = ap.parse_args()

    with open(os.path.join(args.export, "data.yaml")) as f:
        names = yaml.safe_load(f)["names"]
    for n in names:
        os.makedirs(os.path.join(args.out, n), exist_ok=True)

    img_dir = os.path.join(args.export, "train", "images")
    lbl_dir = os.path.join(args.export, "train", "labels")
    copied = {n: 0 for n in names}
    skipped = 0
    for img in glob.glob(os.path.join(img_dir, "*")):
        base = os.path.splitext(os.path.basename(img))[0]
        lbl = os.path.join(lbl_dir, base + ".txt")
        cls = class_of(lbl) if os.path.exists(lbl) else None
        if cls is None:
            skipped += 1
            continue
        name = names[cls]
        shutil.copy2(img, os.path.join(args.out, name, os.path.basename(img)))
        copied[name] += 1

    for n in names:
        print(f"{n}: {copied[n]}")
    print(f"skipped (unlabeled): {skipped}")


if __name__ == "__main__":
    main()
