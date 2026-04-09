import pytest
from src.zone_monitor import ZoneMonitor


@pytest.fixture
def monitor():
    zm = ZoneMonitor()
    zm.add_zone("zone_a", [[100, 100], [300, 100], [300, 300], [100, 300]])
    zm.add_zone("zone_b", [[400, 400], [600, 400], [600, 600], [400, 600]])
    return zm


def test_person_inside_zone_a(monitor):
    assert monitor.check_person([150, 100, 250, 250]) == "zone_a"


def test_person_inside_zone_b(monitor):
    assert monitor.check_person([450, 400, 550, 550]) == "zone_b"


def test_person_outside_all_zones(monitor):
    assert monitor.check_person([0, 0, 100, 50]) is None


def test_person_on_boundary(monitor):
    assert monitor.check_person([50, 100, 150, 200]) == "zone_a"


def test_no_zones_defined():
    empty_monitor = ZoneMonitor()
    assert empty_monitor.check_person([150, 100, 250, 250]) is None
