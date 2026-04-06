# test_zone_monitor.py
from src.zone_monitor import ZoneMonitor


monitor = ZoneMonitor()
monitor.add_zone("zone_a", [[100, 100], [300, 100], [300, 300], [100, 300]])
monitor.add_zone("zone_b", [[400, 400], [600, 400], [600, 600], [400, 600]])

# foot = bottom-center of bbox
# Test 1: Person fully inside zone_a
# bbox center-x=200, y2=250 → foot (200, 250) inside zone_a
result = monitor.check_person([150, 100, 250, 250])
print(f"Test 1 - Expected: zone_a, Got: {result}, {'PASS' if result == 'zone_a' else 'FAIL'}")

# Test 2: Person fully inside zone_b
# bbox center-x=500, y2=550 → foot (500, 550) inside zone_b
result = monitor.check_person([450, 400, 550, 550])
print(f"Test 2 - Expected: zone_b, Got: {result}, {'PASS' if result == 'zone_b' else 'FAIL'}")

# Test 3: Person outside all zones
# foot (50, 50) outside both polygons
result = monitor.check_person([0, 0, 100, 50])
print(f"Test 3 - Expected: None,   Got: {result}, {'PASS' if result is None else 'FAIL'}")

# Test 4: Person on the boundary of zone_a (pointPolygonTest returns 0 on edge)
# foot (100, 200) is on the left edge of zone_a
result = monitor.check_person([50, 100, 150, 200])
print(f"Test 4 - Expected: zone_a, Got: {result}, {'PASS' if result == 'zone_a' else 'FAIL'}")

# Test 5: No zones defined
empty_monitor = ZoneMonitor()
result = empty_monitor.check_person([150, 100, 250, 250])
print(f"Test 5 - Expected: None,   Got: {result}, {'PASS' if result is None else 'FAIL'}")
