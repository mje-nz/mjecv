import enum

import numpy as np
import yaml
from attr import attrib, attrs

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


@attrs
class CameraIntrinsics:
    model: CameraModel = attrib()
    intrinsics: np.ndarray = attrib()
    distortion_model: DistortionModel = attrib()
    distortion_coeffs: np.ndarray = attrib()
    width: float = attrib()
    height: float = attrib()

    @classmethod
    def from_kalibr_camchain(cls, filename, camera_name=None):
        camchain = yaml.load(open(filename), Loader=yaml.SafeLoader)
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
