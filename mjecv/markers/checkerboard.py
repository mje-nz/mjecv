from typing import Optional

import cv2
import numpy as np


def refine_subpixel(
    image: np.ndarray,
    corners: np.ndarray,
    window_size=15,
    zero_size: Optional[int] = None,
    max_iterations=1000,
    tolerance=0.001,
):
    """Refine the locations of corners or radial saddle points with sub-pixel accuracy.

    Args:
        image: single-channel 8-bit or float image.
        corners: Nx2 array of approximate corner locations.
        window_size: The width and height of the area to search around each location.
        zero_size: The width and height of the area to skip in the middle of the search
            window.
        max_iterations: Maximum number of refinement iterations.
        tolerance: Stop refining a corner when it moves less than this much in one
            refinement iteration.

    Returns:
        Nx2 array of refined corner locations.
    """
    # TODO: check precisely what impl does with window size and zero zone
    win_size = (window_size - 1) // 2
    if zero_size:
        zero_zone = (zero_size - 1) // 2
    else:
        zero_zone = -1
    assert max_iterations > 0
    assert tolerance > 0
    criteria = (
        cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
        max_iterations,
        tolerance,
    )
    corners_refined = np.squeeze(
        cv2.cornerSubPix(
            image, corners, (win_size, win_size), (zero_zone, zero_zone), criteria
        )
    )
    assert corners_refined.shape == corners.shape
    return corners_refined


def find_checkerboard_corners(image: np.ndarray, shape=(8, 5), refine=True):
    """Find the positions of internal corners of the chessboard.

    Args:
        image: 8-bit colour or grayscale image.
        shape: Checkerboard shape as internal corners (per row, per column).
        refine: Whether to perform sub-pixel corner refinement.

    Returns:
        Nx2 array of corners, or None if not found.
    """
    # TODO: convert other dtypes to uint8
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FILTER_QUADS
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(image, shape, flags=flags)
    if not found:
        return
    corners = np.squeeze(corners)
    assert corners.shape == (shape[0] * shape[1], 2)
    if refine:
        corners = refine_subpixel(image, corners)
    return corners