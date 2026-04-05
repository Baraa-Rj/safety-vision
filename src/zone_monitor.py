import cv2
import numpy as np


class ZoneMonitor:
    def __init__(self):
        self.zones = {}

    def add_zone(self, zone_id, points):
        """
        points: list of [x, y] coordinates defining the polygon
        """
        self.zones[zone_id] = np.array(points, dtype=np.int32)

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