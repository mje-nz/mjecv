import os

import numpy as np

try:
    from .multiprocessing import dill_for_apply
except ImportError:
    dill_for_apply = None  # type: ignore

__all__ = ["as_float", "dill_for_apply"]


def as_float(arr):
    """Convert an iterable to an `np.array`, ensuring it has a floating-point dtype."""
    if not hasattr(arr, "dtype") or arr.dtype not in (np.float32, np.float64):
        try:
            return np.array(arr, dtype=np.float64)
        except TypeError as e:
            raise ValueError("Invalid floating-point array {}".format(arr)) from e
    return arr


def ensure_open(obj, mode="r", accept_string=True):
    """Given a filename, Path, or open file, return an open file.

    If accept_string is True then passing a string that isn't a filename will return the
    string.
    """
    if hasattr(obj, "read"):
        return obj
    if issubclass(type(obj), os.PathLike):
        return obj.open(mode)
    if not accept_string:
        return open(obj, mode)
    if "\n" not in obj:
        # Probably a filename
        try:
            return open(obj, mode)
        except FileNotFoundError:
            pass
    # Probably a string
    return obj


def require(condition, message=""):
    """Assert-like syntax for raising exceptions which doesn't optimise out."""
    if not condition:
        if type(message) is str:
            raise ValueError(message)
        else:
            raise message
