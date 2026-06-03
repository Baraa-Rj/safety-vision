import logging

import requests

logger = logging.getLogger("safety_vision")


class ZoneClient:
    """Pushes zone definitions to the backend.

    Used when zones are (re)defined so the server reflects what the cameras
    enforce. Network failures are swallowed and logged — the caller is expected
    to have already persisted zones locally, so a server outage must not lose
    the operator's work.
    """

    def __init__(self, endpoint, enabled=True, timeout=10.0):
        self.endpoint = endpoint
        self.enabled = enabled
        self.timeout = timeout

    def upload_zones(self, zones):
        """POST the full zone list. Returns True on success, False otherwise.

        zones: list of dicts as stored in data/zones.json, e.g.
               [{"zone_id": str, "points": [[x, y], ...], "allowed_workers": [...]}]
        """
        if not self.enabled:
            logger.info("Zone upload disabled; skipping server sync.")
            return False

        try:
            payload = {"zones": zones, "count": len(zones)}
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            logger.info("[ZONES UPLOADED] %d zone(s) -> %s", len(zones), self.endpoint)
            return True
        except Exception as e:
            logger.error("[ZONE UPLOAD FAILED] %s", e)
            return False
