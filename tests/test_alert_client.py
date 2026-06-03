import numpy as np

from src.alert_client import AlertClient

FRAME = np.zeros((32, 32, 3), dtype=np.uint8)


def _client(**kw):
    """AlertClient with the background executor replaced by a synchronous
    recorder so we can assert exactly which alerts would be sent."""
    c = AlertClient("http://x/api/ppe-alerts/create", enabled=True, **kw)
    sent = []
    c._executor.submit = lambda fn, *a, **k: sent.append(a)
    return c, sent


def _violation(worker_id="W1", track_id=1):
    return {"worker_id": worker_id, "track_id": track_id,
            "missing": ["vest"], "bbox": [0, 0, 10, 10]}


def test_repeat_violation_suppressed_within_cooldown():
    c, sent = _client(cooldown_seconds=60.0)
    c.send_ppe_alert(_violation(), FRAME)   # first → sent
    c.send_ppe_alert(_violation(), FRAME)   # same worker, immediately → suppressed
    c.send_ppe_alert(_violation(), FRAME)
    assert len(sent) == 1


def test_different_workers_not_suppressed():
    c, sent = _client(cooldown_seconds=60.0)
    c.send_ppe_alert(_violation(worker_id="W1", track_id=1), FRAME)
    c.send_ppe_alert(_violation(worker_id="W2", track_id=2), FRAME)
    assert len(sent) == 2


def test_alert_resent_after_cooldown_elapses():
    c, sent = _client(cooldown_seconds=60.0)
    c.send_ppe_alert(_violation(), FRAME)
    # Backdate the recorded time so the cooldown window has "passed".
    key = next(iter(c._last_alert_time))
    c._last_alert_time[key] -= 61.0
    c.send_ppe_alert(_violation(), FRAME)
    assert len(sent) == 2


def test_disabled_client_sends_nothing():
    c, sent = _client()
    c.enabled = False
    c.send_ppe_alert(_violation(), FRAME)
    assert sent == []
