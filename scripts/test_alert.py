"""Send a test PPE alert to the server to verify the endpoint works.

Usage:
    python3 scripts/test_alert.py
"""

import base64
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import requests
from config.settings import PipelineConfig

config = PipelineConfig()

# Read a frame from the video to use as the test image
cap = cv2.VideoCapture(config.camera.source)
if not cap.isOpened():
    print(f"Cannot open {config.camera.source}")
    sys.exit(1)

ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read a frame")
    sys.exit(1)

# Encode frame to base64
_, buffer = cv2.imencode(".jpg", frame)
img_b64 = base64.b64encode(buffer).decode("utf-8")

endpoint = config.alert.endpoint

# Test 1: Known worker
payload_known = {
    "userId": "42",
    "imgImage": img_b64,
    "missingItem": "helmet, vest",
    "message": "Worker 42 missing helmet, vest",
}

print(f"POST {endpoint}")
print(f"Image size: {len(img_b64)} chars (base64)")
print()

print("--- Test 1: Known worker ---")
try:
    resp = requests.post(endpoint, json=payload_known, timeout=10)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text}")
except Exception as e:
    print(f"Error: {e}")

print()

# Test 2: Unknown worker
payload_unknown = {
    "imgImage": img_b64,
    "missingItem": "vest",
    "message": "Unknown worker missing vest",
}

print("--- Test 2: Unknown worker ---")
try:
    resp = requests.post(endpoint, json=payload_unknown, timeout=10)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text}")
except Exception as e:
    print(f"Error: {e}")
