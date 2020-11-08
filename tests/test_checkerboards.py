from pathlib import Path

import numpy as np
import yaml

from mjecv.calibration import (
    CheckerboardTarget,
    draw_checkerboard,
    find_checkerboard_corners,
)
from mjecv.io import imread


def load_chess1():
    """Load OpenCV test data."""
    data_dir = Path(__file__).parent / "data"
    image = imread(data_dir / "chess1.png")
    corners = yaml.load(open(data_dir / "chess_corners1.dat"), Loader=yaml.SafeLoader)[
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


def test_CheckerboardTarget_object_points():
    target = CheckerboardTarget((3, 2), square_width=0.2, size_or_square_height=0.1)
    expected = [
        (-0.2, -0.05, 0),
        (0, -0.05, 0),
        (0.2, -0.05, 0),
        (-0.2, 0.05, 0),
        (0, 0.05, 0),
        (0.2, 0.05, 0),
    ]
    assert np.allclose(target.object_points, expected)


def test_draw_checkerboard():
    target = CheckerboardTarget((3, 2), 1)
    im = draw_checkerboard(target)
    # fmt: off
    expected = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
    ]) * 255
    # fmt: on
    assert np.all(im == expected)


def test_draw_checkerboard_bg():
    target = CheckerboardTarget((3, 2), 2)
    im = draw_checkerboard(target, background=2, background_colour=100)
    expected = np.array(
        [
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            [100, 100, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 100, 100],
            [100, 100, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 100, 100],
            [100, 100, 255, 255, 000, 000, 255, 255, 000, 000, 255, 255, 100, 100],
            [100, 100, 255, 255, 000, 000, 255, 255, 000, 000, 255, 255, 100, 100],
            [100, 100, 255, 255, 255, 255, 000, 000, 255, 255, 255, 255, 100, 100],
            [100, 100, 255, 255, 255, 255, 000, 000, 255, 255, 255, 255, 100, 100],
            [100, 100, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 100, 100],
            [100, 100, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 100, 100],
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        ]
    )
    assert np.all(im == expected)
