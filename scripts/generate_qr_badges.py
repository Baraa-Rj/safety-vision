"""Generate QR code badge images for workers.

Usage:
    python scripts/generate_qr_badges.py

Edit the WORKERS list below to match your workforce.
Output goes to data/qr_badges/
"""

import json
import os
import cv2

WORKERS = [
    {"id": "W001", "name": "Alice"}
]

OUTPUT_DIR = "data/qr_badges"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    encoder = cv2.QRCodeEncoder.create()

    for worker in WORKERS:
        data = json.dumps(worker)
        qr_img = encoder.encode(data)
        path = os.path.join(OUTPUT_DIR, f"qr_{worker['id']}.png")
        cv2.imwrite(path, qr_img)
        print(f"Generated {path}  ({worker['name']})")

    print(f"\nDone. {len(WORKERS)} badges saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
