import numpy as np
import pytest

from mjecv.calibration import CameraIntrinsics, CameraModel, DistortionModel

# Example from https://github.com/ethz-asl/kalibr/wiki/yaml-formats
camchain = """\
cam0:
  camera_model: pinhole
  intrinsics: [461.629, 460.152, 362.680, 246.049]
  distortion_model: radtan
  distortion_coeffs: [-0.27695497, 0.06712482, 0.00087538, 0.00011556]
  T_cam_imu:
  - [0.01779318, 0.99967549,-0.01822936, 0.07008565]
  - [-0.9998017, 0.01795239, 0.00860714,-0.01771023]
  - [0.00893160, 0.01807260, 0.99979678, 0.00399246]
  - [0.0, 0.0, 0.0, 1.0]
  timeshift_cam_imu: -8.121e-05
  rostopic: /cam0/image_raw
  resolution: [752, 480]
cam1:
  camera_model: omni
  intrinsics: [0.80065662, 833.006, 830.345, 373.850, 253.749]
  distortion_model: radtan
  distortion_coeffs: [-0.33518750, 0.13211436, 0.00055967, 0.00057686]
  T_cn_cnm1:
  - [ 0.99998854, 0.00216014, 0.00427195,-0.11003785]
  - [-0.00221074, 0.99992702, 0.01187697, 0.00045792]
  - [-0.00424598,-0.01188627, 0.99992034,-0.00064487]
  - [0.0, 0.0, 0.0, 1.0]
  T_cam_imu:
  - [ 0.01567142, 0.99978002,-0.01393948,-0.03997419]
  - [-0.99966203, 0.01595569, 0.02052137,-0.01735854]
  - [ 0.02073927, 0.01361317, 0.99969223, 0.00326019]
  - [0.0, 0.0, 0.0, 1.0]
  timeshift_cam_imu: -8.681e-05
  rostopic: /cam1/image_raw
  resolution: [752, 480]
"""


def _check_pinhole(cam):
    assert cam.model == CameraModel.Pinhole
    mat = np.array(((461.629, 0, 362.680), (0, 460.152, 246.049), (0, 0, 1)))
    assert np.allclose(cam.intrinsic_matrix, mat)
    assert cam.distortion_model == DistortionModel.RadTan
    assert np.allclose(
        cam.distortion_coeffs, [-0.27695497, 0.06712482, 0.00087538, 0.00011556]
    )
    assert cam.width == 752
    assert cam.height == 480


def test_camera_intrinsics_pinhole():
    cam = CameraIntrinsics.from_kalibr_yaml(camchain, "cam0")
    _check_pinhole(cam)


def test_camera_intrinsics_only():
    camchain_first = "\n".join(camchain.splitlines()[:13])
    cam = CameraIntrinsics.from_kalibr_yaml(camchain_first)
    _check_pinhole(cam)


def test_camera_intrinsics_omni():
    cam = CameraIntrinsics.from_kalibr_yaml(camchain, "cam1")

    assert cam.model == CameraModel.Omni
    assert np.allclose(cam.intrinsics, [0.80065662, 833.006, 830.345, 373.850, 253.749])
    with pytest.raises(NotImplementedError):
        cam.intrinsic_matrix
    assert cam.distortion_model == DistortionModel.RadTan
    assert np.allclose(
        cam.distortion_coeffs, [-0.33518750, 0.13211436, 0.00055967, 0.00057686]
    )
    assert cam.width == 752
    assert cam.height == 480
