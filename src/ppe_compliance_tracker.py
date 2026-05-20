import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class _ItemState:
    window: deque = field(default_factory=deque)
    alerted: bool = False


@dataclass
class _TrackState:
    items: dict = field(default_factory=dict)
    last_update: float = 0.0


class PPEComplianceTracker:
    """Per-track temporal smoothing for PPE compliance.

    Frame-by-frame PPE detection is noisy (motion blur, jitter, brief occlusion).
    This class converts a stream of per-frame observations into stable alert/clear
    events using a rolling window with hysteresis.
    """

    def __init__(self, config, required_items):
        self.config = config
        self.required_items = frozenset(required_items)
        self._tracks: dict = {}

    @property
    def tracked_count(self) -> int:
        return len(self._tracks)

    def _classify(self, conf):
        # conf is None → item not detected this frame → "missing"
        if conf is None or conf < self.config.uncertain_lower:
            return "missing"
        if conf < self.config.uncertain_upper:
            return "uncertain"
        return "present"

    def _ensure_item_state(self, track_state, item):
        if item not in track_state.items:
            track_state.items[item] = _ItemState(
                window=deque(maxlen=self.config.window_size),
            )
        return track_state.items[item]

    def update(self, track_id, observations, timestamp: Optional[float] = None):
        now = timestamp if timestamp is not None else time.time()
        track_state = self._tracks.setdefault(track_id, _TrackState())
        track_state.last_update = now

        results = {}
        for item in self.required_items:
            item_state = self._ensure_item_state(track_state, item)
            label = self._classify(observations.get(item))
            item_state.window.append(label)

            transition = None
            # Decisions only fire after the window has filled — avoids reacting
            # to the first few frames before we have enough signal.
            if len(item_state.window) >= self.config.window_size:
                missing = sum(1 for x in item_state.window if x == "missing")
                present = sum(1 for x in item_state.window if x == "present")
                if not item_state.alerted and missing >= self.config.missing_to_alert:
                    item_state.alerted = True
                    transition = "alert"
                    logger.debug(
                        "track=%s item=%s ALERT (missing=%d/%d window=%s)",
                        track_id, item, missing, self.config.window_size,
                        list(item_state.window),
                    )
                elif item_state.alerted and present >= self.config.present_to_clear:
                    item_state.alerted = False
                    transition = "clear"
                    logger.debug(
                        "track=%s item=%s CLEAR (present=%d/%d window=%s)",
                        track_id, item, present, self.config.window_size,
                        list(item_state.window),
                    )
            results[item] = transition
        return results

    def get_state(self, track_id):
        track_state = self._tracks.get(track_id)
        if track_state is None:
            return {item: False for item in self.required_items}
        return {
            item: track_state.items.get(item, _ItemState()).alerted
            for item in self.required_items
        }

    def get_window_summary(self, track_id, item):
        track_state = self._tracks.get(track_id)
        if track_state is None or item not in track_state.items:
            return {"present": 0, "missing": 0, "uncertain": 0, "size": 0}
        window = track_state.items[item].window
        return {
            "present": sum(1 for x in window if x == "present"),
            "missing": sum(1 for x in window if x == "missing"),
            "uncertain": sum(1 for x in window if x == "uncertain"),
            "size": len(window),
        }

    def cleanup_stale(self, current_ts: Optional[float] = None) -> int:
        now = current_ts if current_ts is not None else time.time()
        stale = [
            tid for tid, st in self._tracks.items()
            if now - st.last_update > self.config.track_timeout_seconds
        ]
        for tid in stale:
            del self._tracks[tid]
        return len(stale)
