from pathlib import Path

import cv2
import numpy as np

from ..util import free_space


def imread(filename):
    image = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
    if image is None:
        message = f'Could not read "{filename}"'
        if not Path(filename).exists():
            raise FileNotFoundError(message)
        raise IOError(message)
    return image


def imwrite(filename, image: np.ndarray):
    success = cv2.imwrite(str(filename), image)
    if not success:
        message = f'Could not write "{filename}"'
        if not Path(filename).parent.is_dir():
            message += " (parent dir does not exist)"
        elif free_space(filename) < image.nbytes:
            message += " (out of disk space)"
        raise IOError(message)
