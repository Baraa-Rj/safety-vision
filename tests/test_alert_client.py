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


# --- fall alerts ---

def _capture_post(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json

        class _R:
            def raise_for_status(self):
                pass

        return _R()

    monkeypatch.setattr("src.alert_client.requests.post", fake_post)
    return captured


def test_fall_alert_payload_shape_identified(monkeypatch):
    captured = _capture_post(monkeypatch)
    c = AlertClient("http://x/ppe", enabled=True,
                    fall_endpoint="http://x/api/fall-alerts/create")
    c._post_fall_alert(
        {"worker_id": "W42", "worker_name": None, "severity": "HIGH"}, FRAME)

    p = captured["json"]
    assert set(p.keys()) == {"userId", "severity", "imgImage", "message"}
    assert p["userId"] == "W42"
    assert p["severity"] == "HIGH"
    assert p["imgImage"]                      # frame always attached
    assert "HIGH" in p["message"]
    assert captured["url"] == "http://x/api/fall-alerts/create"


def test_fall_alert_unidentified_uses_fallback_user_id(monkeypatch):
    captured = _capture_post(monkeypatch)
    c = AlertClient("http://x/ppe", enabled=True,
                    fall_endpoint="http://x/api/fall-alerts/create",
                    fall_unidentified_user_id="UNKNOWN-1")
    c._post_fall_alert({"worker_id": None, "severity": "LOW"}, FRAME)

    p = captured["json"]
    assert p["userId"] == "UNKNOWN-1"     # sentinel FK for anonymous falls
    assert p["imgImage"]


def test_fall_alert_skipped_when_unidentified_and_no_fallback(monkeypatch):
    captured = _capture_post(monkeypatch)
    c = AlertClient("http://x/ppe", enabled=True,
                    fall_endpoint="http://x/api/fall-alerts/create")
    c._post_fall_alert({"worker_id": None, "severity": "LOW"}, FRAME)

    assert captured == {}                 # no POST attempted (would 500)


def test_send_fall_alert_noop_without_endpoint():
    c, sent = _client()                       # no fall_endpoint configured
    c.send_fall_alert({"severity": "LOW", "alert": True}, FRAME)
    assert sent == []


# --- wet floor alerts ---

def _wf_event():
    return {"bbox": [0, 0, 10, 10], "confidence": 0.95,
            "area_pct": 1.4, "consecutive_count": 5, "first_seen_ts": 1.0}


def test_wet_floor_alert_payload_shape(monkeypatch):
    captured = _capture_post(monkeypatch)
    c = AlertClient("http://x/ppe", enabled=True,
                    wet_floor_endpoint="http://x/api/wet/alert/create")
    c._post_wet_floor_alert(_wf_event(), FRAME)

    p = captured["json"]
    assert set(p.keys()) == {"description", "imgImage"}   # exact backend contract
    assert p["imgImage"]                                  # frame always attached
    assert "Wet floor" in p["description"]
    assert captured["url"] == "http://x/api/wet/alert/create"


def test_send_wet_floor_alert_noop_without_endpoint():
    c, sent = _client()                       # no wet_floor_endpoint configured
    c.send_wet_floor_alert(_wf_event(), FRAME)
    assert sent == []
