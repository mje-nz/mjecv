import cv2
import numpy as np
import transforms3d

__all__ = [
    "as_float",
    "validate_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "rotation_vector_to_quaternion",
]


def as_float(value):
    """Convert an iterable to an `np.array`, ensuring it has a floating point dtype."""
    v = np.array(value)
    if v.dtype not in (np.float16, np.float32, np.float64):
        try:
            v = v.astype(np.float64)
        except TypeError as e:
            raise ValueError("Invalid floating-point array {}".format(value)) from e
    return v


def validate_rotation_matrix(value):
    R = as_float(value)
    if R.shape != (3, 3):
        raise ValueError("Invalid rotation matrix (wrong shape) {}".format(value))
    if not np.allclose(R.dot(R.T), np.eye(3)):
        raise ValueError("Invalid rotation matrix (not orthogonal) {}".format(value))
    if not np.allclose(np.linalg.det(R), 1):
        raise ValueError("Invalid rotation matrix (determinant not 1) {}".format(value))
    return R


def rotation_matrix_to_quaternion(matrix):
    return transforms3d.quaternions.mat2quat(validate_rotation_matrix(matrix))


def validate_rotation_vector(value):
    v = as_float(value).flatten()
    if len(v) != 3:
        raise ValueError("Invalid rotation vector (wrong shape) {}".format(value))
    return v


def rotation_vector_to_quaternion(vec):
    rotation_matrix, _ = cv2.Rodrigues(validate_rotation_vector(vec))
    return rotation_matrix_to_quaternion(rotation_matrix)


def quaternion_to_euler(quat):
    # TODO: document
    # TODO: rotation sequences
    return transforms3d.euler.quat2euler(quat)
