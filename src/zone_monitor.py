import cv2
import numpy as np


class ZoneMonitor:
    def __init__(self):
        self.zones = {}
        self.permissions = {}

    def add_zone(self, zone_id, points, allowed_workers=None):
        """
        points: list of [x, y] coordinates defining the polygon
        allowed_workers: list of worker_id strings permitted in this zone
        """
        self.zones[zone_id] = np.array(points, dtype=np.int32)
        self.permissions[zone_id] = set(allowed_workers) if allowed_workers else set()

    def is_permitted(self, zone_id, worker_id):
        if worker_id is None:
            return False
        return worker_id in self.permissions.get(zone_id, set())

    def check_person(self, person_bbox):
        """
        Check if a person is inside any zone.
        Use bottom-center of bbox as the person's position (their feet).
        Returns zone_id if inside a zone, None if not.
        """
        x1, _, x2, y2 = person_bbox
        foot = (int((x1 + x2) / 2), int(y2))

        for zone_id, polygon in self.zones.items():
            if cv2.pointPolygonTest(polygon, foot, measureDist=False) >= 0:
                return zone_id

        return None