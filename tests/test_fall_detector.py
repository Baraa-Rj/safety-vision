from config.settings import FallDetectionConfig
from src.fall_detector import FallDetector


def _det(**overrides):
    cfg = FallDetectionConfig(**overrides)
    # model_path points nowhere — posture/gate logic doesn't need the model.
    return FallDetector("does_not_exist.pt", cfg)


def _kpts(positions, conf=1.0):
    """Build a 17x3 COCO keypoint list; only `positions` get confidence."""
    k = [[0.0, 0.0, 0.0] for _ in range(17)]
    for i, (x, y) in positions.items():
        k[i] = [float(x), float(y), conf]
    return k


# --- posture: torso angle ---

def test_standing_is_not_fallen():
    d = _det()
    # shoulders above hips, same x -> vertical torso
    kp = _kpts({5: (100, 100), 6: (120, 100), 11: (100, 200), 12: (120, 200)})
    assert d._is_fallen_posture(kp) is False


def test_lying_is_fallen():
    d = _det()
    # shoulders and hips at same height, far apart in x -> horizontal torso
    kp = _kpts({5: (100, 150), 6: (110, 150), 11: (250, 155), 12: (260, 155)})
    assert d._is_fallen_posture(kp) is True


# --- posture: aspect-ratio fallback when torso keypoints missing ---

def test_aspect_fallback_horizontal_is_fallen():
    d = _det()
    # no shoulders/hips; visible points spread wide, short vertically
    kp = _kpts({0: (100, 100), 9: (200, 95), 15: (300, 110)})
    assert d._is_fallen_posture(kp) is True


def test_aspect_fallback_vertical_is_not_fallen():
    d = _det()
    kp = _kpts({0: (100, 100), 9: (105, 250), 15: (110, 400)})
    assert d._is_fallen_posture(kp) is False


# --- temporal gate ---

def test_tracked_needs_sustained_frames():
    d = _det(consecutive_frames=5)
    for _ in range(4):
        assert d._confirm(track_id=1, is_fallen=True) is False
    assert d._confirm(track_id=1, is_fallen=True) is True   # 5th consecutive


def test_one_upright_frame_resets_gate():
    d = _det(consecutive_frames=5)
    for _ in range(5):
        d._confirm(1, True)               # now confirmed
    d._confirm(1, False)                   # a single upright frame
    assert d._confirm(1, True) is False    # streak broken, not yet re-confirmed


def test_untracked_bypasses_gate():
    d = _det()
    assert d._confirm(track_id=None, is_fallen=True) is True
    assert d._confirm(track_id=None, is_fallen=False) is False
