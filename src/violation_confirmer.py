class ViolationConfirmer:
    """Time-based 'sustained violation' gate.

    A single-frame PPE detection flicker should not produce a server alert. This
    confirms a violation only after it has been CONSTANT for `confirm_seconds`,
    keyed by the most stable identity available (worker_id > track_id > location).

    Call update() once per person per frame with their current violation state;
    it returns True only once the violation has held continuously for the window.
    A compliant observation resets that identity's timer.
    """

    def __init__(self, confirm_seconds, stale_after=5.0):
        self.confirm_seconds = confirm_seconds
        self.stale_after = stale_after
        self._since = {}       # key -> timestamp the current violation started
        self._last_seen = {}   # key -> last update timestamp (for cleanup)

    def update(self, key, is_violating, now):
        self._last_seen[key] = now
        if not is_violating:
            # Worker is compliant this frame — reset their streak.
            self._since.pop(key, None)
            return False
        start = self._since.setdefault(key, now)
        return (now - start) >= self.confirm_seconds

    def cleanup(self, now):
        """Drop timers for identities not seen recently (left the frame)."""
        stale = [k for k, t in self._last_seen.items() if now - t > self.stale_after]
        for k in stale:
            self._since.pop(k, None)
            self._last_seen.pop(k, None)
        return len(stale)
