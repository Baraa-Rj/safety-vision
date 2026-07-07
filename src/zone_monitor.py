import cv2
import numpy as np


class ZoneMonitor:
    def __init__(self, band_frac=0.30):
        self.zones = {}
        self.permissions = {}
        self.radii = {}
        self.backend_ids = {}   # zone_id (str) -> backend integer id for alerts
        # Originals as saved in zones.json, kept so polygons can be rescaled
        # when the runtime frame size differs from the one they were drawn on.
        self._ref_points = {}
        self._ref_sizes = {}    # zone_id -> (w, h) the zone was defined on, or None
        self._ref_radii = {}
        self._frame_size = None
        # Fraction of the bbox height (from the bottom) used as the lower-body
        # reference band. Sampling this band instead of a single foot point keeps
        # detection working when feet are truncated by the frame edge or occluded
        # by an object — the lowest *visible* body part is still used.
        self.band_frac = band_frac

    def add_zone(self, zone_id, points, allowed_workers=None, radius=0, backend_id=0,
                 frame_size=None):
        """
        points: list of [x, y] coordinates defining the polygon
        allowed_workers: list of worker_id strings permitted in this zone
        radius: keep-out distance in pixels around the polygon.
            0  -> plain containment: a foot triggers only when inside the polygon
                  (restricted area behaviour, unchanged).
            >0 -> proximity: a foot triggers when inside OR within `radius` pixels
                  of the polygon edge. Use this for a restricted machine — draw a
                  polygon around its footprint and set the keep-out distance.
        backend_id: the server's integer id for this zone, sent as `zoneId` in
            breach alerts (the backend keys on an int, not the string zone_id).
        frame_size: (w, h) of the frame the points were drawn on. When the
            pipeline later reports its actual frame size (set_frame_size), the
            polygon is rescaled so a zone defined on the 704x576 substream still
            lands correctly on the 2880x1620 main stream. None = points are
            already in runtime coordinates and are never rescaled.
        """
        self.zones[zone_id] = np.array(points, dtype=np.int32)
        self.permissions[zone_id] = set(allowed_workers) if allowed_workers else set()
        self.radii[zone_id] = float(radius)
        self.backend_ids[zone_id] = int(backend_id)
        self._ref_points[zone_id] = [list(p) for p in points]
        self._ref_sizes[zone_id] = tuple(frame_size) if frame_size else None
        self._ref_radii[zone_id] = float(radius)
        if frame_size and self._frame_size:
            self._scale_zone(zone_id, *self._frame_size)

    def set_frame_size(self, width, height):
        """Rescale zones to the frame size actually being processed.

        Called by the pipeline on every frame; a no-op unless the size changed.
        Zones added without a frame_size are left untouched.
        """
        if (width, height) == self._frame_size:
            return
        self._frame_size = (width, height)
        for zone_id in self.zones:
            self._scale_zone(zone_id, width, height)

    def _scale_zone(self, zone_id, width, height):
        ref = self._ref_sizes.get(zone_id)
        if not ref:
            return
        sx, sy = width / ref[0], height / ref[1]
        self.zones[zone_id] = np.array(
            [[x * sx, y * sy] for x, y in self._ref_points[zone_id]],
            dtype=np.int32)
        # Keep-out distance is isotropic; scale by the mean stretch.
        self.radii[zone_id] = self._ref_radii[zone_id] * (sx + sy) / 2.0

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
