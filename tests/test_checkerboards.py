import numpy as np
import yaml

from mjecv.calibration import CheckerboardTarget, find_checkerboard_corners
from mjecv.io import imread


def load_chess1():
    """Load OpenCV test data."""
    image = imread("data/chess1.png")
    corners = yaml.load(open("data/chess_corners1.dat"), Loader=yaml.SafeLoader)[
        "corners"
    ]
    shape = (corners["cols"], corners["rows"])
    expected = np.array(corners["data"]).reshape((-1, 2))
    return image, shape, expected


def test_find_checkerboard_corners():
    image, shape, expected = load_chess1()

    actual = find_checkerboard_corners(image, shape)
    assert np.allclose(actual, expected, atol=0.1)


def test_CheckerboardTarget_detect():
    image, shape, expected = load_chess1()
    target = CheckerboardTarget(shape, 1)

    actual = target.detect(image)
    assert np.allclose(actual, expected, atol=0.1)
