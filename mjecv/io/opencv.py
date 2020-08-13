import cv2
import numpy as np


def imwrite(filename: str, image: np.ndarray):
    success = cv2.imwrite(filename, image)
    if not success:
        raise IOError("cv2.imwrite failed")
