"""Generate QR code badge images for workers (A4, 300 DPI).

Usage:
    python scripts/generate_qr_badges.py

Edit the WORKERS list below to match your workforce.
Output goes to data/qr_badges/
"""

import os
import cv2
import numpy as np

# A4 at 300 DPI
PAGE_WIDTH = 2480
PAGE_HEIGHT = 3508
QR_SIZE = int(PAGE_WIDTH * 0.8)  # 80% of page width

WORKERS = [
    {"id": "0cb42", "label": "worker-1"},
    {"id": "98c39", "label": "worker-2"},
    {"id": "c52a4", "label": "worker-3"},
]

OUTPUT_DIR = "data/qr_badges"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    encoder = cv2.QRCodeEncoder.create()

    for worker in WORKERS:
        qr_img = encoder.encode(worker["id"])

        # Scale QR to large size with sharp edges
        qr_large = cv2.resize(qr_img, (QR_SIZE, QR_SIZE), interpolation=cv2.INTER_NEAREST)

        # Create white A4 canvas and center the QR
        page = np.ones((PAGE_HEIGHT, PAGE_WIDTH), dtype=np.uint8) * 255
        x_offset = (PAGE_WIDTH - QR_SIZE) // 2
        y_offset = (PAGE_HEIGHT - QR_SIZE) // 2
        page[y_offset:y_offset + QR_SIZE, x_offset:x_offset + QR_SIZE] = qr_large

        path = os.path.join(OUTPUT_DIR, f"qr_{worker['label']}.png")
        cv2.imwrite(path, page)
        print(f"Generated {path}  ({worker['label']})")

    print(f"\nDone. {len(WORKERS)} badges saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
