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


def test_alert_resent_after_backoff_gap_elapses():
    # After k alerts the next needs a gap of cooldown_seconds * (k + 1):
    # the second alert requires 2x the base.
    c, sent = _client(cooldown_seconds=60.0)
    c.send_ppe_alert(_violation(), FRAME)
    key = next(iter(c._last_alert_time))
    c._last_alert_time[key] -= 61.0          # one base elapsed: not enough now
    c.send_ppe_alert(_violation(), FRAME)
    assert len(sent) == 1
    c._last_alert_time[key] -= 60.0          # two bases elapsed in total
    c.send_ppe_alert(_violation(), FRAME)
    assert len(sent) == 2


def test_backoff_intervals_increase_arithmetically():
    # 5s base: first immediate, then gaps of 10, 15, 20 seconds.
    c, sent = _client(cooldown_seconds=5.0)
    now = 1000.0
    assert c._allow("W1", now) is True                 # first alert
    assert c._allow("W1", now + 9) is False
    assert c._allow("W1", now + 10) is True            # +10s
    assert c._allow("W1", now + 10 + 14) is False
    assert c._allow("W1", now + 10 + 15) is True       # +15s
    assert c._allow("W1", now + 25 + 19) is False
    assert c._allow("W1", now + 25 + 20) is True       # +20s


def test_backoff_counter_resets_after_long_silence():
    c, sent = _client(cooldown_seconds=5.0)
    now = 1000.0
    for _ in range(3):
        assert c._allow("W1", now) is True
        now += 100                                     # generous gaps: 3 alerts
    now += c._BACKOFF_RESET_SECONDS                    # violation episode over
    assert c._allow("W1", now) is True                 # fresh episode
    assert c._allow("W1", now + 9) is False            # back to the 2x-base gap
    assert c._allow("W1", now + 10) is True


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
            status_code = 201
            text = '{"alertId": "test"}'

            @property
            def url(self):
                return url

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


# --- zone alerts ---

def _zone_event(worker_id="W7", backend_id=3):
    return {"worker_id": worker_id, "worker_name": None, "track_id": 1,
            "bbox": [0, 0, 10, 10], "zone_id": "restricted_1",
            "zone_backend_id": backend_id}


def test_zone_alert_payload_shape_identified(monkeypatch):
    captured = _capture_post(monkeypatch)
    c = AlertClient("http://x/ppe", enabled=True,
                    zone_endpoint="http://x/api/zone/alert/create")
    c._post_zone_alert(_zone_event(worker_id="W7", backend_id=3), FRAME)

    p = captured["json"]
    assert set(p.keys()) == {"imgImage", "userId", "zoneId", "message"}
    assert p["userId"] == "W7"
    assert p["zoneId"] == 3 and isinstance(p["zoneId"], int)
    assert p["imgImage"]                       # zone alerts always attach the frame
    assert captured["url"] == "http://x/api/zone/alert/create"


def test_zone_alert_unidentified_uses_sentinel(monkeypatch):
    captured = _capture_post(monkeypatch)
    c = AlertClient("http://x/ppe", enabled=True,
                    zone_endpoint="http://x/api/zone/alert/create",
                    zone_unidentified_user_id="UNKNOWN-1")
    c._post_zone_alert(_zone_event(worker_id=None), FRAME)
    assert captured["json"]["userId"] == "UNKNOWN-1"


def test_zone_alert_unidentified_no_sentinel_sends_blank_user(monkeypatch):
    # Zone breaches always send, identified or not; with no worker and no
    # sentinel, userId is blank but the alert (with image) still goes out.
    captured = _capture_post(monkeypatch)
    c = AlertClient("http://x/ppe", enabled=True,
                    zone_endpoint="http://x/api/zone/alert/create")
    c._post_zone_alert(_zone_event(worker_id=None), FRAME)
    assert captured["json"]["userId"] == ""
    assert captured["json"]["imgImage"]


def test_send_zone_alert_noop_without_endpoint():
    c, sent = _client()                       # no zone_endpoint configured
    c.send_zone_alert(_zone_event(), FRAME)
    assert sent == []


def test_fall_sentinel_defaults_to_shared_anonymous_row():
    # A fall must never be skipped for lack of identity: the config supplies
    # the shared anonymous user row by default (overridable via env).
    from config.settings import PipelineConfig
    assert PipelineConfig().alert.fall_unidentified_user_id == "2678a"


def _wet_event(first_seen_ts=None):
    e = {"bbox": [100, 100, 200, 160], "confidence": 0.93, "area_pct": 0.3,
         "consecutive_count": 5}
    if first_seen_ts is not None:
        e["first_seen_ts"] = first_seen_ts
    return e


def test_wet_floor_first_alert_waits_for_persistence():
    import time as _time
    c, sent = _client(wet_floor_first_alert_seconds=10.0)
    c.wet_floor_endpoint = "http://x/api/wet-alert/create"
    c.send_wet_floor_alert(_wet_event(first_seen_ts=_time.time() - 3), FRAME)
    assert sent == []                                     # only 3s old: wait
    c.send_wet_floor_alert(_wet_event(first_seen_ts=_time.time() - 11), FRAME)
    assert len(sent) == 1                                 # persisted 10s: send


def test_wet_floor_backoff_uses_wet_base():
    import time as _time
    c, sent = _client(cooldown_seconds=8.0, wet_floor_first_alert_seconds=10.0,
                      wet_floor_cooldown_seconds=10.0)
    c.wet_floor_endpoint = "http://x/api/wet-alert/create"
    old = _time.time() - 60
    c.send_wet_floor_alert(_wet_event(first_seen_ts=old), FRAME)
    assert len(sent) == 1
    key = next(iter(c._last_alert_time))
    c._last_alert_time[key] -= 19.0                       # 19s elapsed < 2x10s
    c.send_wet_floor_alert(_wet_event(first_seen_ts=old), FRAME)
    assert len(sent) == 1
    c._last_alert_time[key] -= 1.0                        # 20s elapsed
    c.send_wet_floor_alert(_wet_event(first_seen_ts=old), FRAME)
    assert len(sent) == 2


def test_jittered_wet_bboxes_share_one_backoff_key():
    # The exact boxes from the live triple-alert: centroid crossed two 50px
    # grid lines, but all three are the same spill and must share one key.
    import time as _time
    c, sent = _client(wet_floor_first_alert_seconds=10.0,
                      wet_floor_cooldown_seconds=10.0)
    c.wet_floor_endpoint = "http://x/api/wet-alert/create"
    old = _time.time() - 60
    for bbox in ([1418, 1434, 1779, 1514],
                 [1421, 1433, 1779, 1513],
                 [1521, 1433, 1785, 1513]):
        e = _wet_event(first_seen_ts=old)
        e["bbox"] = bbox
        c.send_wet_floor_alert(e, FRAME)
    assert len(sent) == 1
    assert len(c._wet_spills) == 1


def test_distinct_spills_get_distinct_keys():
    import time as _time
    c, sent = _client(wet_floor_first_alert_seconds=10.0,
                      wet_floor_cooldown_seconds=10.0)
    c.wet_floor_endpoint = "http://x/api/wet-alert/create"
    old = _time.time() - 60
    for bbox in ([100, 100, 300, 200], [2000, 1200, 2300, 1400]):
        e = _wet_event(first_seen_ts=old)
        e["bbox"] = bbox
        c.send_wet_floor_alert(e, FRAME)
    assert len(sent) == 2
    assert len(c._wet_spills) == 2


def test_ppe_alert_identified_carries_image_and_user_id(monkeypatch):
    # Every PPE alert attaches the frame as visual evidence; identified
    # workers additionally carry their userId.
    captured = _capture_post(monkeypatch)
    c = AlertClient("http://x/api/ppe-alerts/create", enabled=True)
    c._post_alert(_violation(worker_id="W42"), FRAME)

    p = captured["json"]
    assert set(p.keys()) == {"missingItem", "message", "imgImage", "userId"}
    assert p["userId"] == "W42"
    assert p["imgImage"]


def test_ppe_alert_unidentified_carries_image_without_user_id(monkeypatch):
    captured = _capture_post(monkeypatch)
    c = AlertClient("http://x/api/ppe-alerts/create", enabled=True)
    c._post_alert(_violation(worker_id=None), FRAME)

    p = captured["json"]
    assert set(p.keys()) == {"missingItem", "message", "imgImage"}
    assert p["imgImage"]
