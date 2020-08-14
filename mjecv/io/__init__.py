try:
    from .opencv import imwrite
except ImportError:
    imwrite = None  # type: ignore

from .base import ImageSequenceWriter, MultiprocessingImageSequenceWriter

ParallelImageSequenceWriter = MultiprocessingImageSequenceWriter

try:
    from .ray import RayImageSequenceWriter
    ParallelImageSequenceWriter = RayImageSequenceWriter
except ImportError:
    RayImageSequenceWriter = None  # type: ignore


__all__ = [
    "imwrite",
    "ImageSequenceWriter",
    "MultiprocessingImageSequenceWriter",
    "ParallelImageSequenceWriter",
    "RayImageSequenceWriter",
]
