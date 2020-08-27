import numpy as np
import pytest

from mjecv.calibration import (
    CheckerboardTarget,
    PinholeNoneIntrinsics,
    PinholeRadTanIntrinsics,
)
from mjecv.geometry import solve_pnp


def test_solve_pnp_simple():
    f = 10
    p = 100
    intrinsic_matrix = np.array([[f, 0, p], [0, f, p], [0, 0, 1]])
    # cam = PinholeRadTanIntrinsics((f, f, p, p), (0, 0, 0, 0), 200, 200)
    points = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0], [2, 0, 0]])
    world_points = np.copy(points)
    world_points[:, 2] += 2
    image_points = p + f * world_points[:, :2] / world_points[:, 2].reshape(-1, 1)
    rvec, tvec = solve_pnp(points, image_points, intrinsic_matrix, tol=0.1)
    assert np.allclose(rvec, (0, 0, 0))
    assert np.allclose(tvec, (0, 0, 2))


poses = (
    ((0, 0, 0), (0, 0, 1)),
    ((0.5, 0, 0), (0, 0, 1)),
    ((0, 0.5, 0), (0, 0, 1)),
    ((0, 0, 0.5), (0, 0, 1)),
    ((0.5, 0.5, 0), (0, 0, 1)),
    ((0.5, 0, 0.5), (0, 0, 1)),
    ((0, 0.5, 0.5), (0, 0, 1)),
    ((0.5, 0.5, 0.5), (0, 0, 1)),
)


@pytest.mark.parametrize(
    "cam",
    (
        PinholeNoneIntrinsics((10, 10, 100, 100), shape=(200, 200)),
        PinholeRadTanIntrinsics(
            (10, 10, 100, 100), (1e-3, 1e-4, 1e-5, 1e-6), shape=(200, 200)
        ),
    ),
)
@pytest.mark.parametrize("expected_rvec,expected_tvec", poses)
def test_solve_pnp_pinhole(cam, expected_rvec, expected_tvec):
    np.random.seed(0)
    points = np.c_[np.random.uniform(-2, 2, (8, 2)), np.ones((8, 1))]
    image_points = cam.project_points(points, expected_rvec, expected_tvec)

    rvec, tvec = cam.solve_pnp(points, image_points, tol=0.1)
    assert np.allclose(rvec, expected_rvec, atol=1e-6)
    assert np.allclose(tvec, expected_tvec, atol=1e-6)


@pytest.mark.parametrize("expected_rvec,expected_tvec", poses)
def test_solvepnp_checkerboard(expected_rvec, expected_tvec):
    cam = PinholeNoneIntrinsics((10, 10, 100, 100), shape=(200, 200))
    target = CheckerboardTarget((8, 5), 0.1)
    image_points = cam.project_points(
        target.object_points, expected_rvec, expected_tvec
    )
    rvec, tvec = cam.solve_pnp(target.object_points, image_points, tol=0.1)
    assert np.allclose(rvec, expected_rvec, atol=1e-10)
    assert np.allclose(tvec, expected_tvec, atol=1e-10)


@pytest.mark.parametrize("expected_rvec,expected_tvec", poses)
def test_solvepnp_checkerboard_cm3(expected_rvec, expected_tvec):
    # Intrinsics from Chameleon3 in slider experiment
    cam = PinholeRadTanIntrinsics(
        [1727.2, 1730.4, 638.2, 482.2],
        [-9.97e-02, 1.8e-01, -7e-04, -1e-04],
        1280,
        1024,
    )
    # Checkerboard configuration from slider experiment
    # TODO: add check for points outside image, which is what happens when this
    #       checkerboard is bigger
    target = CheckerboardTarget((8, 5), 0.035)
    image_points = cam.project_points(
        target.object_points, expected_rvec, expected_tvec
    )

    rvec, tvec = cam.solve_pnp(target.object_points, image_points, tol=0.1,)
    assert np.allclose(rvec, expected_rvec, atol=1e-9)
    assert np.allclose(tvec, expected_tvec, atol=1e-9)


@pytest.mark.parametrize(
    "bad_point",
    ((-1, 100), (100, -1), (200, 100), (100, 210), (np.nan, np.nan), (np.inf, np.inf)),
)
def test_solvepnp_bad_points(bad_point):
    cam = PinholeNoneIntrinsics((10, 10, 100, 100), shape=(200, 210))
    target = CheckerboardTarget((8, 5), 0.1)
    # TODO: similar checks in project points
    image_points = cam.project_points(target.object_points, (0, 0, 0), (0, 0, 0))
    image_points[0] = bad_point
    with pytest.raises(ValueError):
        cam.solve_pnp(
            target.object_points, image_points,
        )
