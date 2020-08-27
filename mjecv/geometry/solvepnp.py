from typing import Tuple

import cv2
import numpy as np

from ..util import as_float, require

__all__ = ["solve_pnp"]


def _solve_pnp_generic(
    object_points: np.ndarray,
    image_points: np.ndarray,
    intrinsic_matrix: np.ndarray,
    distortion_coeffs: np.ndarray = None,
    prior: np.ndarray = None,
    method=None,
):
    """Find an object pose from 3D-2D point correspondences.

    The return value is a transformation consisting of a fixed-axis ZYX rotation
    followed by a translation.  It is not always correct, as cv2.solvePnP has
    a tendency to silently fail -- only in corner cases as far as I know however.

    N.B. OpenCV requires the points to be Nx3 and Nx2. This function will
    silently flip them around if they're 3xN or 2xN, but if they're 3x3 or 2x2
    they must be one row per point.

    Arguments:
        object_points: vector of 3D points in object coordinates
        image_points: vector of corresponding 2D points in image coordinates
        intrinsic_matrix: 3x3 camera intrinsic matrix
    Returns: rotation vector, (x, y, z) translation in metres
    """
    # Check arguments
    require(object_points is not None, "Object points are required")
    object_points = as_float(object_points)
    require(np.all(np.isfinite(object_points)), "Object points must be finite")
    if object_points.shape[1] != 3:
        require(object_points.shape[0] == 3, "Object points must be 3xN or Nx3")
        object_points = object_points.transpose()

    require(image_points is not None, "Image points are required")
    image_points = as_float(image_points)
    require(np.all(np.isfinite(image_points)), "Image points must be finite")
    if image_points.shape[1] != 2:
        require(image_points.shape[0] == 2, "Image points must be 2xN or Nx2")
        image_points = image_points.transpose()

    require(intrinsic_matrix is not None, "Intrinsic matrix is required")
    intrinsic_matrix = as_float(intrinsic_matrix)
    if distortion_coeffs is not None:
        distortion_coeffs = as_float(distortion_coeffs)

    prior_rvec = None
    prior_tvec = None
    if prior:
        require(cv2.SOLVEPNP_ITERATIVE, "Prior only used with SOLVEPNP_ITERATIVE")
        prior_rvec, prior_tvec = prior

    solutions, rvecs, tvecs, reprojection_errors = cv2.solvePnPGeneric(
        object_points,
        image_points,
        intrinsic_matrix,
        distortion_coeffs,
        flags=method,
        useExtrinsicGuess=prior is not None,
        rvec=prior_rvec,
        tvec=prior_tvec,
    )
    if solutions == 0:
        raise RuntimeError("No solutions")

    rvecs = [np.squeeze(r) for r in rvecs]
    tvecs = [np.squeeze(t) for t in tvecs]
    return rvecs, tvecs, reprojection_errors.reshape(-1)


def solve_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray = None,
    prior: np.ndarray = None,
    tol: float = None,
    shape: Tuple[int, int] = None,
):
    """TODO

    Note that at least 6 points are required, or 4 if they're planar, or 3 with a prior.

    Args:
        tol: maximum acceptable RMS reprojection error in pixels.
        shape: (rows, cols) of image size, used for sanity checks.
    """
    require(np.all(image_points > 0), "Image points must be within image bounds")
    if shape:
        require(np.all(image_points < shape))

    rvecs, tvecs, reprojection_errors = _solve_pnp_generic(
        object_points,
        image_points,
        camera_matrix,
        distortion_coeffs,
        prior,
        cv2.SOLVEPNP_ITERATIVE,
    )

    assert len(rvecs) == 1
    if tol and reprojection_errors[0] > tol:
        raise RuntimeError(
            f"Reprojection error too high: {reprojection_errors[0]} > {tol}"
        )
    return rvecs[0], tvecs[0]
