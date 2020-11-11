import enum
from typing import Optional

import cv2
import numpy as np
from attr import attrib, attrs

from ..geometry import solve_pnp
from ..util import as_float, require
from .intrinsics import CameraIntrinsics, CameraModel, DistortionModel

__all__ = [
    "ImageExtent",
    "InterpolationMethod",
    "OpenCvUndistorter",
    "PinholeNoneIntrinsics",
    "PinholeRadTanIntrinsics",
]


# TODO: add implementations from
#       https://github.com/ethz-asl/image_undistort/blob/master/src/undistorter.cpp


# By default, OpenCV's undistort produces undistorted images with the same intrinsic
# matrix.
# Depending on the distortion coefficients, this can result in black regions at the
# edges (where no source image pixels fall within the new FOV) or throwing away valid
# pixels (where source image pixels fall outside the new FOV),
# `getOptimalNewCameraMatrix` offers a few solutions:
# * scale the image to include only valid pixels
# * scale the image to include all pixels with black regions
# * either of the above, keeping the principal point in the centre


class ImageExtent(enum.Enum):
    #: Keep the original intrinsic matrix (default).
    Original = enum.auto()
    #: Expand the image to contain every pixel from the source image.
    AllSource = enum.auto()
    #: Expand the image to contain every pixel from the source image, and place the
    #  new principal point at the centre of the image.
    AllSourceCentred = enum.auto()
    #: Expand the image to contain as much of the source image as possible without
    #  including any invalid areas.
    OnlyValid = enum.auto()
    #: Expand the image to contain as much of the source image as possible without
    #  including any invalid areas, and place the new principal point at the centre of
    #  the image.
    OnlyValidCentred = enum.auto()


class InterpolationMethod(enum.Enum):
    Nearest = cv2.INTER_NEAREST
    Bilinear = cv2.INTER_LINEAR
    Bicubic = cv2.INTER_CUBIC
    Lanczos = cv2.INTER_LANCZOS4


@attrs(auto_attribs=True)
class OpenCvUndistorter:

    """Undistort an image using OpenCV's initUndistortRectifyMap and remap functions."""

    intrinsic_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    width: int
    height: int
    extent: ImageExtent = ImageExtent.Original
    interpolation: InterpolationMethod = InterpolationMethod.Bilinear
    #: The colour to fill invalid areas of the undistorted image with, if applicable.
    invalid_colour: np.ndarray = 0
    #: Increase the image size by this factor while undistorting.
    scale: float = 1.0
    #: Zoom the image out by this factor while undistorting.
    zoom: float = 1.0
    #: The width of the undistorted images (calculated if not provided).
    new_width: Optional[int] = None
    #: The height of the undistorted images (calculated if not provided).
    new_height: Optional[int] = None
    #: The intrinsic matrix for the undistorted images.
    new_intrinsic_matrix: np.ndarray = attrib(init=False)

    def __attrs_post_init__(self):
        assert self.width
        assert self.height
        if not self.new_width:
            self.new_width = int(self.width * self.scale)
        if not self.new_height:
            self.new_height = int(self.height * self.scale)

        self.new_intrinsic_matrix = self._calculate_new_intrinsic_matrix()
        self.new_intrinsic_matrix[:2, :2] *= self.zoom
        self._maps = self._calculate_maps()

    def _get_optimal_camera_matrix(self, alpha):
        raise NotImplementedError()

    def _calculate_new_intrinsic_matrix(self):
        if self.extent == ImageExtent.Original:
            return self.intrinsic_matrix
        else:
            if self.extent in (ImageExtent.AllSource, ImageExtent.AllSourceCentred):
                alpha = 1
            else:
                alpha = 0
            # TODO: support Centred extents
            return self._get_optimal_camera_matrix(alpha)

    def _calculate_maps(self):
        raise NotImplementedError()

    def undistort_image(self, image: np.ndarray):
        return cv2.remap(
            image,
            *self._maps,
            interpolation=self.interpolation.value,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.invalid_colour
        )


class OpenCvUndistorterRadTan(OpenCvUndistorter):
    def _get_optimal_camera_matrix(self, alpha):
        new_intrinsic_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.intrinsic_matrix,
            self.distortion_coeffs,
            (self.width, self.height),
            alpha,
            (self.new_width, self.new_height),
        )
        return new_intrinsic_matrix

    def _calculate_maps(self):
        return cv2.initUndistortRectifyMap(
            self.intrinsic_matrix,
            self.distortion_coeffs,
            None,
            self.new_intrinsic_matrix,
            (self.new_width, self.new_height),
            cv2.CV_16SC2,
        )


class OpenCvUndistorterFisheye(OpenCvUndistorter):
    def _get_optimal_camera_matrix(self, alpha):
        new_intrinsic_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.intrinsic_matrix,
            self.distortion_coeffs,
            (self.width, self.height),
            np.eye(3),
            balance=alpha,
            new_size=(self.new_width, self.new_height),
        )
        return new_intrinsic_matrix

    def _calculate_maps(self):
        return cv2.fisheye.initUndistortRectifyMap(
            self.intrinsic_matrix,
            self.distortion_coeffs,
            np.eye(3),
            self.new_intrinsic_matrix,
            (self.new_width, self.new_height),
            cv2.CV_16SC2,
        )


def _project_points_radtan(points, rvec, tvec, intrinsic_matrix, distortion_coeffs):
    # TODO: use Pose
    points = as_float(points)
    # TODO: catch hard crash on wrong shape e.g.
    #       cv2.projectPoints(np.zeros((10, 3)), np.zeros((3, 4)), np.zeros((3, 4)),
    #       np.zeros((3, 3)), np.zeros((4)))
    rvec = as_float(rvec)
    tvec = as_float(tvec)
    intrinsic_matrix = as_float(intrinsic_matrix)
    if distortion_coeffs is not None:
        distortion_coeffs = as_float(distortion_coeffs)
    assert 1 <= points.ndim <= 3
    assert points.shape[-1] == 3
    projection, _ = cv2.projectPoints(
        points, rvec, tvec, intrinsic_matrix, distortion_coeffs
    )
    # Match original shape, dropping z-coord
    new_shape = (*points.shape[:-1], points.shape[-1] - 1)
    return projection.reshape(new_shape)


class PinholeNoneIntrinsics(
    CameraIntrinsics, model=CameraModel.Pinhole, distortion_model=DistortionModel.None_
):

    # TODO: subclass PinholeRadTan

    def __init__(
        self, intrinsics, distortion_coeffs=None, width=None, height=None, shape=None
    ):
        """Construct camera intrinsics with pinhole model and no distortion.

        N.B. This has to have the same signature as the others, so it does take a
        distortion_coeffs parameter.
        """
        require(
            distortion_coeffs is None or len(distortion_coeffs) == 0,
            "Distortion coefficients must be null",
        )
        if width is None:
            height, width = shape[:2]
        super().__init__(
            CameraModel.Pinhole, intrinsics, DistortionModel.None_, [], width, height,
        )

    def project_points(self, points, rvec, tvec):
        # TODO: use Pose
        return _project_points_radtan(points, rvec, tvec, self.intrinsic_matrix, None)

    def solve_pnp(self, object_points, image_points, prior=None, tol=None):
        return solve_pnp(
            object_points,
            image_points,
            self.intrinsic_matrix,
            self.distortion_coeffs,
            prior,
            tol,
            self.shape,
        )


class PinholeRadTanIntrinsics(
    CameraIntrinsics, model=CameraModel.Pinhole, distortion_model=DistortionModel.RadTan
):
    def __init__(
        self, intrinsics, distortion_coeffs, width=None, height=None, shape=None
    ):
        if width is None:
            height, width = shape[:2]
        super().__init__(
            CameraModel.Pinhole,
            intrinsics,
            DistortionModel.RadTan,
            distortion_coeffs,
            width,
            height,
        )

    def get_undistorter(self, *args, **kwargs):
        return OpenCvUndistorterRadTan(
            self.intrinsic_matrix,
            self.distortion_coeffs,
            self.width,
            self.height,
            *args,
            **kwargs
        )

    def project_points(self, points, rvec, tvec):
        # TODO: use Pose
        return _project_points_radtan(
            points, rvec, tvec, self.intrinsic_matrix, self.distortion_coeffs
        )

    def solve_pnp(self, object_points, image_points, prior=None, tol=None):
        return solve_pnp(
            object_points,
            image_points,
            self.intrinsic_matrix,
            self.distortion_coeffs,
            prior,
            tol,
            self.shape,
        )


class PinholeFisheyeIntrinsics(
    CameraIntrinsics,
    model=CameraModel.Pinhole,
    distortion_model=DistortionModel.Fisheye,
):
    def __init__(
        self, intrinsics, distortion_coeffs, width=None, height=None, shape=None
    ):
        if width is None:
            height, width = shape[:2]
        super().__init__(
            CameraModel.Pinhole,
            intrinsics,
            DistortionModel.Fisheye,
            distortion_coeffs,
            width,
            height,
        )

    def get_undistorter(self, *args, **kwargs):
        return OpenCvUndistorterFisheye(
            self.intrinsic_matrix,
            self.distortion_coeffs,
            self.width,
            self.height,
            *args,
            **kwargs
        )


# TODO: lots of tests
