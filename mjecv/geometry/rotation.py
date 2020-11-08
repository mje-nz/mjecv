# https://github.com/mje-nz/mjemarker/blob/master/mjemarker/geometry.py
import cv2
import numpy as np
import transforms3d

from ..util import as_float

__all__ = [
    "quaternion_to_euler",
    "rotation_matrix_to_quaternion",
    "rotation_matrix_to_vector",
    "rotation_vector_to_quaternion",
    "rotation_vector_to_matrix",
    "quaternion_to_rotation_matrix",
    "quaternion_to_rotation_vector",
]


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


def rotation_vector_to_matrix(vec):
    rotation_matrix, _ = cv2.Rodrigues(validate_rotation_vector(vec))
    return rotation_matrix


def rotation_matrix_to_vector(mat):
    rotation_vector, _ = cv2.Rodrigues(validate_rotation_matrix(mat))
    return rotation_vector


def rotation_vector_to_quaternion(vec):
    return rotation_matrix_to_quaternion(rotation_vector_to_matrix(vec))


def quaternion_to_euler(quat):
    # TODO: document
    # TODO: rotation sequences
    return transforms3d.euler.quat2euler(quat)


quaternion_to_rotation_matrix = transforms3d.quaternions.quat2mat


def quaternion_to_rotation_vector(quaternion):
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
    return validate_rotation_vector(rotation_vector)
