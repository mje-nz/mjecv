import cv2
import numpy as np

from ..features import refine_subpixel

__all__ = ["find_checkerboard_corners"]


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
