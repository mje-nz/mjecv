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


def require(condition, message=""):
    """Assert-like syntax for raising exceptions which doesn't optimise out."""
    if not condition:
        if type(message) is str:
            raise ValueError(message)
        else:
            raise message
