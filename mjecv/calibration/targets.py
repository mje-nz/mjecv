import enum
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml
from attr import attrs

from ..util import ensure_open
from .intrinsics import CameraIntrinsics

__all__ = [
    "AprilGridTarget",
    "CalibrationTarget",
    "CalibrationTargetType",
    "CircleGridTarget",
]


class CalibrationTargetType(enum.Enum):
    AprilGrid = "aprilgrid"
    Checkerboard = "checkerboard"
    CircleGrid = "circlegrid"


@attrs(auto_attribs=True)
class CalibrationTarget:

    """Base class for calibration target configurations."""

    type_: CalibrationTargetType
    # Number of tags/internal corners/circles down/across
    rows: int
    cols: int
    # AprilGrid only: size of each tag in metres
    size: Optional[float] = None
    # Aprilgrid: distance between tags as a fraction of size
    # CircleGrid: distance between circles in metres
    spacing: Optional[float] = None
    # Checkerboard: height/width of each square in metres
    row_spacing: Optional[float] = None
    col_spacing: Optional[float] = None
    # CircleGrid: Use asymmetric grid
    asymmetric_grid: Optional[bool] = None

    _types = {}  # type: Dict[CalibrationTargetType, Any]

    def __init_subclass__(cls, type_: CalibrationTargetType):
        """Register subclasses when they're declared."""
        cls._types[type_] = cls

    def __attrs_post_init__(self):
        if self.row_spacing != self.col_spacing:
            w, h = self.col_spacing, self.row_spacing
            warnings.warn(f"Are you sure you have rectangular ({w}x{h}) squares?")

    @property
    def shape(self):
        return self.cols, self.rows

    @property
    def object_points(self):
        raise NotImplementedError()

    def detect(self, image: np.ndarray, refine=True, tolerance=1e-3):
        """Detect the target in an image.

        Returns:
            np.ndarray: Nx2 list of image points (tag corners/internal corners/centers),
            or None if not found.
        """
        raise NotImplementedError()

    def estimate_pose(self, corners, intrinsics: CameraIntrinsics):
        """Estimate the target's pose.

        Returns: rotation vector, (x, y, z) translation in metres
        """
        raise NotImplementedError()

    @classmethod
    def from_kalibr_yaml(cls, file_or_str) -> "CalibrationTarget":
        """Load calibration target configuration from a Kalibr target yaml file.

        Args:
            file_or_str: Filename or string of yaml file to load.
        """
        target_yaml = yaml.load(ensure_open(file_or_str), Loader=yaml.SafeLoader)
        target_type = CalibrationTargetType(target_yaml["target_type"])
        return cls._types[target_type]._from_kalibr_yaml(target_yaml)


class AprilGridTarget(CalibrationTarget, type_=CalibrationTargetType.AprilGrid):
    def __init__(self, shape: Tuple[int, int], size: float, spacing=0.3):
        """Construct an AprilGrid calibration target.

        Args:
            shape: Number of tags (per row, per column).
            size: Size of each tag in metres.
            spacing: Distance between tags as a ratio of size.
        """
        cols, rows = shape
        super().__init__(CalibrationTargetType.AprilGrid, rows, cols, size, spacing)

    @classmethod
    def _from_kalibr_yaml(cls, target_yaml):
        rows = int(target_yaml["tagRows"])
        cols = int(target_yaml["tagCols"])
        size = float(target_yaml["tagSize"])
        spacing = float(target_yaml["tagSpacing"])
        return cls((cols, rows), size, spacing)


class CircleGridTarget(CalibrationTarget, type_=CalibrationTargetType.CircleGrid):
    def __init__(self, shape: Tuple[int, int], spacing: float, asymmetric_grid: bool):
        """Construct a circle grid calibration target.

        Args:
            shape: Number of circles (per row, per column).
            asymmetric_grid: Use asymmetric grid.
        """
        cols, rows = shape
        super().__init__(
            CalibrationTargetType.CircleGrid,
            rows,
            cols,
            spacing=spacing,
            asymmetric_grid=asymmetric_grid,
        )

    @classmethod
    def _from_kalibr_yaml(cls, target_yaml):
        rows = int(target_yaml["targetRows"])
        cols = int(target_yaml["targetCols"])
        spacing = float(target_yaml["spacingMeters"])
        asymmetric_grid = target_yaml["asymmetricGrid"]
        return cls((cols, rows), spacing, asymmetric_grid)
