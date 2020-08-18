import enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml
from attr import attrs

# The pinhole camera model in OpenCV uses the intrinsic matrix
#   [f_x  0    c_x]
#   [0    f_y  0  ]
#   [0    0    1  ]
# and distortion coefficients as the first 5, 8, 12, or 14 of
# (k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, τx, τy)
# TODO: copy in full eqn from
#       https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
#
# The fisheye camera model in OpenCV uses the same intrinsic matrix with a four-
# parameter distortion model
# TODO: copy in full eqn from
#       https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html
#
# Kalibr's pinhole camera model is the same as OpenCV.
# Kalibr's radtan distortion model is OpenCV's (k1, k2, p1, p2).
# Kalibr's equi distortion model is OpenCV's fisheye distortion model.
#
# I don't think they have anything else in common.


class CameraModel(enum.Enum):
    Pinhole = "pinhole"
    Omni = "omni"
    DoubleSphere = "ds"
    ExtendedUnified = "eucm"


class DistortionModel(enum.Enum):
    None_ = "none"
    RadTan = "radtan"
    Fisheye = "fisheye"
    Equi = "fisheye"
    Fov = "fov"

    @classmethod
    def from_name(cls, name):
        for v in cls:
            if v.name.lower() == name:
                return v
        raise ValueError(f"Invalid {cls.__name__}: {name}")


@attrs(auto_attribs=True)
class CameraIntrinsics:
    model: CameraModel
    intrinsics: np.ndarray
    distortion_model: DistortionModel
    distortion_coeffs: np.ndarray
    width: float
    height: float

    @property
    def intrinsic_matrix(self):
        if self.model != CameraModel.Pinhole:
            raise NotImplementedError("Intrinsic only supported for pinhole model")
        fx, fy, px, py = self.intrinsics
        return np.array(((fx, 0, px), (0, fy, py), (0, 0, 1)))

    @classmethod
    def from_kalibr_yaml(cls, file, camera_name: str = None):
        """Load camera intrinsics from a Kalibr camchain yaml file.

        Args:
            file: Filename or string of yaml file to load.
            camera_name: Name of camera to load intrinsics for (optional if only one).
        """
        if ":" not in file:
            # File is a filename
            file = open(file)
        camchain = yaml.load(file, Loader=yaml.SafeLoader)
        assert len(camchain) >= 1
        if not camera_name:
            if len(camchain) > 1:
                raise ValueError(
                    "Multiple cameras in camchain and no camera name specified"
                )
            camera_name = list(camchain.keys())[0]
        cam = camchain[camera_name]
        model = CameraModel(cam["camera_model"])
        intrinsics = np.array(cam["intrinsics"])
        distortion_model = DistortionModel.from_name(cam["distortion_model"])
        distortion_coeffs = np.array(cam["distortion_coeffs"])
        width, height = cam["resolution"]
        return cls(
            model, intrinsics, distortion_model, distortion_coeffs, width, height
        )


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

    @property
    def shape(self):
        return self.cols, self.rows

    def detect(self, image: np.ndarray, *args, **kwargs):
        """Detect the target in an image.

        Returns:
            np.ndarray: Nx2 list of image points (tag corners/internal corners/centers),
            or None if not found.
        """
        raise NotImplementedError()

    @classmethod
    def from_kalibr_yaml(cls, file):
        """Load calibration target configuration from a Kalibr target yaml file.

        Args:
            file: Filename or string of yaml file to load.
        """
        target_yaml = yaml.load(file, Loader=yaml.SafeLoader)
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

    @classmethod
    def _from_kalibr_yaml(cls, target_yaml):
        rows = int(target_yaml["targetRows"])
        cols = int(target_yaml["targetCols"])
        row_spacing = float(target_yaml["rowSpacingMeters"])
        col_spacing = float(target_yaml["colSpacingMeters"])
        return cls((cols, rows), row_spacing, col_spacing)


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
