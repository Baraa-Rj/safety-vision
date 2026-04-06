# test_worker_id.py
import cv2
import numpy as np
from src.worker_id import WorkerIdentifier


def generate_marker(marker_id, size=200, border=50):
    """Generate an ArUco marker image with white border."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
    # Add white border for detection
    padded = cv2.copyMakeBorder(marker, border, border, border, border,
                                cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


identifier = WorkerIdentifier()

# Test 1: Detect marker ID 42
marker_img = generate_marker(42)
result = identifier.identify(marker_img)
print(f"Test 1 - Expected: 42, Got: {result}, {'PASS' if result == 42 else 'FAIL'}")

# Test 2: Detect marker ID 7
marker_img = generate_marker(7)
result = identifier.identify(marker_img)
print(f"Test 2 - Expected: 7, Got: {result}, {'PASS' if result == 7 else 'FAIL'}")

# Test 3: No marker (blank image)
blank = np.zeros((200, 200, 3), dtype=np.uint8)
result = identifier.identify(blank)
print(f"Test 3 - Expected: None, Got: {result}, {'PASS' if result is None else 'FAIL'}")
