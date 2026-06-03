from src.violation_confirmer import ViolationConfirmer


def test_not_confirmed_before_window():
    c = ViolationConfirmer(confirm_seconds=15.0)
    assert c.update("W1", True, now=100.0) is False     # streak starts
    assert c.update("W1", True, now=110.0) is False      # 10s < 15s
    assert c.update("W1", True, now=114.9) is False


def test_confirmed_after_window():
    c = ViolationConfirmer(confirm_seconds=15.0)
    c.update("W1", True, now=100.0)
    assert c.update("W1", True, now=115.0) is True       # exactly 15s
    assert c.update("W1", True, now=120.0) is True


def test_compliant_frame_resets_streak():
    c = ViolationConfirmer(confirm_seconds=15.0)
    c.update("W1", True, now=100.0)
    c.update("W1", True, now=110.0)
    c.update("W1", False, now=111.0)                     # became compliant
    assert c.update("W1", True, now=120.0) is False       # streak restarted at 120
    assert c.update("W1", True, now=135.0) is True


def test_keys_are_independent():
    c = ViolationConfirmer(confirm_seconds=15.0)
    c.update("W1", True, now=100.0)
    c.update("W2", True, now=110.0)
    assert c.update("W1", True, now=116.0) is True        # W1: 16s
    assert c.update("W2", True, now=116.0) is False        # W2: 6s


def test_cleanup_drops_absent_keys():
    c = ViolationConfirmer(confirm_seconds=15.0, stale_after=5.0)
    c.update("W1", True, now=100.0)
    assert c.cleanup(now=104.0) == 0                       # still recent
    assert c.cleanup(now=106.0) == 1                       # not seen for >5s
