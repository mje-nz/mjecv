import cv2
import numpy as np


def imread(filename):
    image = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError("cv2.imread failed")
    return image


def imwrite(filename, image: np.ndarray):
    success = cv2.imwrite(str(filename), image)
    if not success:
        raise IOError("cv2.imwrite failed")
