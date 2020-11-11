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
        image: single-channel 8-bit integer or 32-bit float image.
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
    # TODO: don't modify input!!
    # TODO: check precisely what impl does with window size and zero zone
    # TODO: convert image to np.float32 if not 8U or 32F
    # TODO: convert image to grayscale if necessary
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
    # TODO: reshape instead of squeeze
    assert corners_refined.shape == corners.shape
    return corners_refined
