import cv2
import numpy as np

__all__ = ["solve_pnp"]


def solve_pnp(
    object_points: np.ndarray, image_points: np.ndarray, camera_matrix: np.ndarray
):
    """Find an object pose from 3D-2D point correspondences.

    The return value is a transformation consisting of a fixed-axis ZYX rotation
    followed by a translation.  It is not always correct, as cv2.solvePnP has
    a tendency to silently fail -- only in corner cases as far as I know however.

    N.B. OpenCV requires the points to be Nx3 and Nx2. This function will
    silently flip them around if they're 3xN or 2xN, but if they're 3x3 or 2x2
    they must be one row per point.

    Arguments:
        object_points: vector of 3D points in object coordinates
        image_points: vector of corresponding 2D points in image coordinates
        camera_matrix: camera matrix
    Returns: rotation vector, (x, y, z) translation in metres
    """
    # Check arguments
    if object_points.shape[1] != 3:
        assert object_points.shape[0] == 3, "Object points must be 3xN or Nx3"
        object_points = object_points.transpose()
    if image_points.shape[1] != 2:
        assert image_points.shape[0] == 2, "Image points must be 2xN or Nx2"
        image_points = image_points.transpose()
    if np.any(np.isnan(image_points)):
        raise ValueError("Image points may not be NaN")
    if np.any(np.isinf(image_points)):
        raise ValueError("Image points may not be infinite")

    found, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,  # Nx3 matrix of points in object coords
        image_points,  # Nx2 matrix of points in image coords
        camera_matrix,
        None,
    )
    if not found:
        raise RuntimeError("Marker not found?")
    return rotation_vector, translation_vector
