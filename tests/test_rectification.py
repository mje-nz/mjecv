from pathlib import Path

import numpy as np

from mjecv.calibration import CameraIntrinsics, ImageExtent, PinholeRadTanIntrinsics
from mjecv.io import imread


DATA_DIR = Path(__file__).parent / "data"


def test_undistort_static_radtan():
    image = imread(DATA_DIR / "static-cam0-98.png")
    expected = imread(DATA_DIR / "static-cam0-98-undistorted-radtan.png")
    intrinsics_file = DATA_DIR / "camchain-static-radtan.yaml"
    cam = CameraIntrinsics.from_kalibr_yaml(intrinsics_file, "cam0")
    actual = cam.get_undistorter(extent=ImageExtent.OnlyValid).undistort_image(image)
    assert np.allclose(actual, expected)


def test_undistort_static_fisheye():
    image = imread(DATA_DIR / "static-cam0-98.png")
    expected = imread(DATA_DIR / "static-cam0-98-undistorted-equi.png")
    cam = CameraIntrinsics.from_kalibr_yaml(DATA_DIR / "camchain-static-equi.yaml", "cam0")
    actual = cam.get_undistorter(extent=ImageExtent.OnlyValid).undistort_image(image)
    assert np.allclose(actual, expected)


def test_pinhole_radtan_project():
    cam = PinholeRadTanIntrinsics((10, 10, 100, 100), (0, 0, 0, 0), 200, 200)
    # Basic
    assert np.all(cam.project_points((0, 0, 0), (0, 0, 0), (0, 0, 0)) == (100, 100))
    # Different shapes
    assert np.all(cam.project_points([[0, 0, 0]], (0, 0, 0), (0, 0, 0)) == [[100, 100]])
    assert np.all(
        cam.project_points([[[0, 0, 0]]], (0, 0, 0), (0, 0, 0)) == [[[100, 100]]]
    )
    # Different dtypes
    assert np.all(
        cam.project_points(np.array([0, 0, 0], dtype=np.float32), (0, 0, 0), (0, 0, 0))
        == (100, 100)
    )
    assert np.all(
        cam.project_points(np.array([0, 0, 0], dtype=np.uint64), (0, 0, 0), (0, 0, 0))
        == (100, 100)
    )
    # Another point
    assert np.all(cam.project_points((1, 2, 0), (0, 0, 0), (0, 0, 1)) == (110, 120))


# TODO: more tests
