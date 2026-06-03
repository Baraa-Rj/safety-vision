import cv2
import numpy as np


class ZoneMonitor:
    def __init__(self, band_frac=0.30):
        self.zones = {}
        self.permissions = {}
        self.radii = {}
        # Fraction of the bbox height (from the bottom) used as the lower-body
        # reference band. Sampling this band instead of a single foot point keeps
        # detection working when feet are truncated by the frame edge or occluded
        # by an object — the lowest *visible* body part is still used.
        self.band_frac = band_frac

    def add_zone(self, zone_id, points, allowed_workers=None, radius=0):
        """
        points: list of [x, y] coordinates defining the polygon
        allowed_workers: list of worker_id strings permitted in this zone
        radius: keep-out distance in pixels around the polygon.
            0  -> plain containment: a foot triggers only when inside the polygon
                  (restricted area behaviour, unchanged).
            >0 -> proximity: a foot triggers when inside OR within `radius` pixels
                  of the polygon edge. Use this for a restricted machine — draw a
                  polygon around its footprint and set the keep-out distance.
        """
        self.zones[zone_id] = np.array(points, dtype=np.int32)
        self.permissions[zone_id] = set(allowed_workers) if allowed_workers else set()
        self.radii[zone_id] = float(radius)

    def is_permitted(self, zone_id, worker_id):
        if worker_id is None:
            return False
        return worker_id in self.permissions.get(zone_id, set())

    def _reference_points(self, person_bbox):
        """Sample a 3x3 grid over the lower band of the bbox.

        Returns points so the closest one to a zone can be used — robust to the
        feet being out of frame or occluded, where a single bottom-center point
        would land on the knees/waist and read as outside the zone.
        """
        x1, y1, x2, y2 = person_bbox
        h = max(0, y2 - y1)
        y_top = int(y2 - self.band_frac * h)
        xs = (int(x1), int((x1 + x2) / 2), int(x2))
        ys = (y_top, int((y_top + y2) / 2), int(y2))
        return [(x, y) for y in ys for x in xs]

    def check_person(self, person_bbox):
        """
        Check if a person is inside (or, for a zone with a radius, near) any zone.
        Uses the lower-body band of the bbox (not a single foot point) so the
        check survives truncated/occluded feet. Returns the first matching
        zone_id, or None.
        """
        points = self._reference_points(person_bbox)

        for zone_id, polygon in self.zones.items():
            radius = self.radii.get(zone_id, 0.0)
            # Signed distance: >0 inside, 0 on edge, <0 outside (px to nearest
            # edge). Take the band point closest to (or deepest inside) the zone.
            best = max(
                cv2.pointPolygonTest(polygon, p, measureDist=True) for p in points
            )
            if best >= -radius:
                return zone_id

        return None
