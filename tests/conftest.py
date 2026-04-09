import numpy as np
import pytest


@pytest.fixture
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)
