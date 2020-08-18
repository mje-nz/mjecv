import cv2
import numpy as np

from ..features import refine_subpixel
from .targets import CalibrationTarget, CalibrationTargetType

__all__ = ["CheckerboardTarget", "find_checkerboard_corners"]


def find_checkerboard_corners(image: np.ndarray, shape=(8, 5), refine=True):
    """Find the positions of internal corners of the chessboard.

    Args:
        image: 8-bit colour or grayscale image.
        shape: Checkerboard shape as internal corners (per row, per column).
        refine: Whether to perform sub-pixel corner refinement.

    Returns:
        Nx2 array of corners, or None if not found.
    """
    # TODO: tests for corner ordering
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


class CheckerboardTarget(CalibrationTarget, type_=CalibrationTargetType.Checkerboard):
    def __init__(self, shape, size_or_square_height: float, square_width=None):
        """Construct a checkerboard calibration target.

        Args:
            shape (Tuple[int, int]): Number of internal corners (per row, per column).
            size_or_square_height: Square size in metres, or square height if
                `square_width` given.
            square_height (Optional[float]): Square height in metres.
        """
        cols, rows = shape
        if not square_width:
            square_width = size_or_square_height
        super().__init__(
            CalibrationTargetType.Checkerboard,
            rows,
            cols,
            row_spacing=size_or_square_height,
            col_spacing=square_width,
        )

    def detect(self, image: np.ndarray, refine=True):
        return find_checkerboard_corners(image, self.shape, refine)

    @classmethod
    def _from_kalibr_yaml(cls, target_yaml):
        rows = int(target_yaml["targetRows"])
        cols = int(target_yaml["targetCols"])
        row_spacing = float(target_yaml["rowSpacingMeters"])
        col_spacing = float(target_yaml["colSpacingMeters"])
        return cls((cols, rows), row_spacing, col_spacing)
