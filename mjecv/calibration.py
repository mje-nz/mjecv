import enum

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
