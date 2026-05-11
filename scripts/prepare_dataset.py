"""Convert polygon labels to bounding boxes and split into train/val/test.

Usage:
    PYTHONPATH=. python3 scripts/prepare_dataset.py

Reads from data/dataset/train/ (Roboflow export), converts segmentation
polygons to YOLO bounding-box format, and splits into 70/20/10.
"""

import os
import random
import shutil

SRC_DIR = "data/dataset/train"
OUT_DIR = "data/ppe_dataset"
SPLIT_RATIOS = (0.7, 0.2, 0.1)  # train, val, test
SEED = 42


def polygon_to_bbox(tokens):
    """Convert polygon coordinate tokens to x_center, y_center, w, h."""
    coords = [float(t) for t in tokens]
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return x_center, y_center, width, height


def convert_label(src_path, dst_path):
    """Read a polygon-format label file, write bounding-box format."""
    lines = []
    with open(src_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = parts[0]
            xc, yc, w, h = polygon_to_bbox(parts[1:])
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    with open(dst_path, "w") as f:
        f.writelines(lines)


def main():
    img_dir = os.path.join(SRC_DIR, "images")
    lbl_dir = os.path.join(SRC_DIR, "labels")

    images = sorted(os.listdir(img_dir))
    random.seed(SEED)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])

    splits = {
        "train": images[:n_train],
        "valid": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    # Clean output
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    for split_name, file_list in splits.items():
        img_out = os.path.join(OUT_DIR, split_name, "images")
        lbl_out = os.path.join(OUT_DIR, split_name, "labels")
        os.makedirs(img_out)
        os.makedirs(lbl_out)

        for img_name in file_list:
            # Copy image
            shutil.copy2(os.path.join(img_dir, img_name), os.path.join(img_out, img_name))

            # Convert and copy label
            lbl_name = os.path.splitext(img_name)[0] + ".txt"
            src_lbl = os.path.join(lbl_dir, lbl_name)
            dst_lbl = os.path.join(lbl_out, lbl_name)
            if os.path.exists(src_lbl):
                convert_label(src_lbl, dst_lbl)
            else:
                # Empty label = negative example (no objects)
                open(dst_lbl, "w").close()

        print(f"{split_name}: {len(file_list)} images")

    # Write dataset.yaml
    yaml_path = os.path.join(OUT_DIR, "dataset.yaml")
    abs_out = os.path.abspath(OUT_DIR)
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_out}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("test: test/images\n")
        f.write("\n")
        f.write("nc: 3\n")
        f.write("names: ['helmet', 'person', 'vest']\n")

    print(f"\nDataset ready at {OUT_DIR}/")
    print(f"YAML config: {yaml_path}")


if __name__ == "__main__":
    main()
