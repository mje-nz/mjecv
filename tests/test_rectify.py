import numpy as np

from mjecv.calibration import CameraIntrinsics, ImageExtent
from mjecv.io import imread


def test_undistort_static_radtan():
    image = imread("data/static-cam0-98.png")
    expected = imread("data/static-cam0-98-undistorted-radtan.png")
    cam = CameraIntrinsics.from_kalibr_yaml("data/camchain-static-radtan.yaml", "cam0")
    actual = cam.get_undistorter(extent=ImageExtent.OnlyValid).undistort_image(image)
    assert np.allclose(actual, expected)


def test_undistort_static_fisheye():
    image = imread("data/static-cam0-98.png")
    expected = imread("data/static-cam0-98-undistorted-equi.png")
    cam = CameraIntrinsics.from_kalibr_yaml("data/camchain-static-equi.yaml", "cam0")
    actual = cam.get_undistorter(extent=ImageExtent.OnlyValid).undistort_image(image)
    assert np.allclose(actual, expected)


# TODO: more tests
