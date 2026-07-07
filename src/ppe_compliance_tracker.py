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
    # True once a full window held strong present evidence — the item is
    # reliably worn, so detection-dropout bursts get a higher alert bar.
    established: bool = False


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
                if not item_state.alerted and present >= self.config.present_to_establish:
                    item_state.established = True
                # An established item (a full window once showed it reliably
                # worn) only alerts on PURE absence: the model drops worn
                # helmets in multi-second bursts (measured: when it sees one
                # at all, conf >= 0.35; misses emit nothing), so a partial
                # window of misses is detector noise, not a removed helmet.
                # A truly removed item reaches all-missing within one window
                # anyway — the alert arrives 3 frames later, not never.
                # Fresh tracks (never established) keep the 5-of-8 bar.
                alert_bar = (
                    len(item_state.window) if item_state.established
                    else self.config.missing_to_alert
                )
                if not item_state.alerted and missing >= alert_bar:
                    item_state.alerted = True
                    item_state.established = False
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

    def get_display_state(self, track_id):
        """Per-item on-screen verdict: True = draw as missing, False = present,
        None = not enough evidence for any verdict yet (render as "checking").

        `get_state` is silent until the window fills, which is right for
        backend alerts but fails open on screen: a worker entering the frame
        with no PPE at all would be drawn green until window_size frames
        accumulate — and again after every track-id churn. A bare first-frame
        vote fails the other way: one blurry entry frame or a transient
        phantom person box flashes VIOLATION. So during warm-up:
        - RED needs display_min_evidence frames of PURE absence — the model
          never saw the item at all, not even below threshold. A single
          present or uncertain sighting blocks the fast red, because entry
          frames (half-visible worker, motion blur) are exactly where the
          model under-detects and a false red is the costlier error.
        - GREEN needs display_min_evidence present frames and a majority.
        Anything less stays None. Once the window has filled, the hysteresis
        verdict takes over — a mixed-evidence worker gets THAT verdict, just
        a few frames later.
        """
        track_state = self._tracks.get(track_id)
        state = {}
        for item in self.required_items:
            item_state = track_state.items.get(item) if track_state else None
            if item_state is None or not item_state.window:
                state[item] = None
            elif len(item_state.window) >= self.config.window_size:
                state[item] = item_state.alerted
            else:
                missing = sum(1 for x in item_state.window if x == "missing")
                present = sum(1 for x in item_state.window if x == "present")
                min_ev = self.config.display_min_evidence
                if missing >= min_ev and missing == len(item_state.window):
                    state[item] = True
                elif present >= min_ev and present > missing:
                    state[item] = False
                else:
                    state[item] = None
        return state

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
