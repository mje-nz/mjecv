import enum
from typing import Any, Dict, Tuple

import numpy as np
import yaml
from attr import attrs

from ..util import ensure_open, require

__all__ = ["CameraIntrinsics", "CameraModel", "DistortionModel"]


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
    RadialTangential = "radtan"
    Fisheye = "fisheye"
    Equi = "fisheye"
    Equidistant = "fisheye"
    Fov = "fov"

    # TODO: test "radial-tangential" and "equidistant", since they seem to be valid

    @classmethod
    def from_name(cls, name):
        for member_name, member in cls.__members__.items():
            if member_name.lower() == name.replace("-", ""):
                return member
        raise ValueError(f"Invalid {cls.__name__}: {name}")


@attrs(auto_attribs=True)
class CameraIntrinsics:
    model: CameraModel
    # TODO: convert arrays
    intrinsics: np.ndarray
    distortion_model: DistortionModel
    distortion_coeffs: np.ndarray
    # TODO: take shape instead of width, height
    # TODO: reorder so subclasses have simpler __init__s
    width: int
    height: int

    _derived = {}  # type: Dict[Tuple[CameraModel, DistortionModel], Any]

    def __init_subclass__(cls, model=None, distortion_model=None):
        """Register subclasses when they're declared."""
        if model and distortion_model:
            cls._derived[(model, distortion_model)] = cls

    @property
    def shape(self):
        return self.height, self.width

    # TODO: focal_length property
    # TODO: principal_point property

    @property
    def intrinsic_matrix(self):
        if self.model != CameraModel.Pinhole:
            raise NotImplementedError("Intrinsic only supported for pinhole model")
        fx, fy, px, py = self.intrinsics
        return np.array(((fx, 0, px), (0, fy, py), (0, 0, 1)), dtype=np.float64)

    def get_undistorter(self, *args, **kwargs):
        raise NotImplementedError()

    def undistort_image(self, image: np.ndarray, *args, **kwargs):
        return self.get_undistorter(*args, **kwargs).undistort_image(image)

    def project_points(self, points, rvec, tvec):
        raise NotImplementedError()

    def solve_pnp(self, object_points, image_points, prior=None, tol=None):
        raise NotImplementedError()

    @classmethod
    def from_kalibr_yaml(cls, file_or_str, camera_name: str = None):
        """Load camera intrinsics from a Kalibr camchain yaml file.

        Args:
            file_or_str: Filename or string of yaml file to load.
            camera_name: Name of camera to load intrinsics for (optional if only one).
        """
        camchain = yaml.load(ensure_open(file_or_str), Loader=yaml.SafeLoader)
        require(camchain is not None, "No calibration YAML provided")
        require(len(camchain) >= 1, "Calibration YAML must not be empty")
        if not camera_name:
            require(
                len(camchain) == 1,
                "Multiple cameras in camchain and no camera name specified",
            )
            camera_name = list(camchain.keys())[0]
        cam = camchain[camera_name]
        model = CameraModel(cam["camera_model"])
        intrinsics = np.array(cam["intrinsics"])
        distortion_model = DistortionModel.from_name(cam["distortion_model"])
        distortion_coeffs = np.array(cam["distortion_coeffs"])
        width, height = cam["resolution"]
        try:
            return cls._derived[(model, distortion_model)](
                intrinsics, distortion_coeffs, width, height
            )
        except KeyError:
            return cls(
                model, intrinsics, distortion_model, distortion_coeffs, width, height
            )
