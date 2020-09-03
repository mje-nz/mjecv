import cv2
import numpy as np

from ..features import refine_subpixel
from .intrinsics import CameraIntrinsics
from .targets import CalibrationTarget, CalibrationTargetType

__all__ = [
    "CheckerboardTarget",
    "draw_checkerboard",
    "find_checkerboard_corners",
]


def find_checkerboard_corners(
    image: np.ndarray, shape=(8, 5), refine=True, tolerance=1e-3
):
    """Find the positions of internal corners of the chessboard.

    Args:
        image: 8-bit colour or grayscale image.
        shape: Checkerboard shape as internal corners (per row, per column).
        refine: Whether to perform sub-pixel corner refinement.
        tolerance: Tolerance for sub-pixel corner refinement.

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
        corners = refine_subpixel(image, corners, tolerance=tolerance)
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
        # TODO: flip order of height, width to match shape
        if not square_width:
            square_width = size_or_square_height
        super().__init__(
            CalibrationTargetType.Checkerboard,
            rows,
            cols,
            row_spacing=size_or_square_height,
            col_spacing=square_width,
        )

    @property
    def object_points(self):
        """3D coordinates of internal corners, number along rows from top left.

        Origin is at the center.
        """
        # Corners indexed as (row, col) from top left, number along rows first
        corner_indices = np.moveaxis(np.mgrid[: self.rows, : self.cols], 0, 2)
        corner_indices = corner_indices.reshape(-1, 2)
        # Indexed by (x, y)
        corner_indices = np.roll(corner_indices, 1, axis=1)
        # Move origin to center
        center = (np.array((self.cols, self.rows)) - 1) / 2
        corner_coords = corner_indices - center
        # Convert to metres
        corner_coords[:, 0] *= self.col_spacing
        corner_coords[:, 1] *= self.row_spacing
        # Add z axis
        corner_coords = np.c_[corner_coords, np.zeros(len(corner_coords))]
        return corner_coords

    def detect(self, image: np.ndarray, refine=True, tolerance=1e-3):
        return find_checkerboard_corners(image, self.shape, refine, tolerance)

    def estimate_pose(self, corners, intrinsics: CameraIntrinsics):
        return intrinsics.solve_pnp(self.object_points, corners)

    @classmethod
    def _from_kalibr_yaml(cls, target_yaml):
        rows = int(target_yaml["targetRows"])
        cols = int(target_yaml["targetCols"])
        row_spacing = float(target_yaml["rowSpacingMeters"])
        col_spacing = float(target_yaml["colSpacingMeters"])
        return cls((cols, rows), row_spacing, col_spacing)


def draw_checkerboard(
    target: CheckerboardTarget, border: float = 1, background=0, background_colour=128,
):
    """Draw a 2D checkerboard.

    Args:
        target: Target to draw; sizes interpreted in pixels.
        border: Width of white border as a factor of square size.
        background: Width of background around each edge in pixels.
        background_colour: Background colour.

    Returns:
        np.ndarray: single-channel uint8 image
    """
    assert target.row_spacing is not None
    assert target.col_spacing is not None
    h = int(target.row_spacing)
    w = int(target.col_spacing)
    height = int(h * (target.rows + 2 * border) + 2 * background)
    width = int(w * (target.cols + 2 * border) + 2 * background)

    im = np.ones((height, width), dtype=np.uint8)
    if background:
        # Start with image full of background colour
        im *= background_colour

        # Fill checkerboard area with white
        im[background:-background, background:-background] = 255
    else:
        # Start with image full of white
        im *= 255

    # Draw black squares
    for i in range(target.rows):
        y = int(background + (i + border) * h)
        for j in range(i % 2, target.cols, 2):
            x = int(background + (j + border) * w)
            im[y : y + h, x : x + w] = 0
    return im
