import json
import cv2


class WorkerIdentifier:
    def __init__(self):
        self.detector = cv2.QRCodeDetector()

    def identify(self, person_crop):
        """
        Takes a cropped image of a person.
        Returns dict with qr_data, worker_name, worker_id — or None if no QR found.
        """
        if person_crop is None or person_crop.size == 0:
            return None

        data, _, _ = self.detector.detectAndDecode(person_crop)
        if not data:
            return None

        try:
            parsed = json.loads(data)
            return {
                "qr_data": data,
                "worker_name": parsed.get("name"),
                "worker_id": parsed.get("id"),
            }
        except (json.JSONDecodeError, TypeError):
            return {
                "qr_data": data,
                "worker_name": data,
                "worker_id": None,
            }
