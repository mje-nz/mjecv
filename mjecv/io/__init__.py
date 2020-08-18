try:
    from .opencv import imread, imwrite
except ImportError:
    imread = None  # type: ignore
    imwrite = None  # type: ignore

from .base import ImageSequenceWriter, MultiprocessingImageSequenceWriter

ParallelImageSequenceWriter = MultiprocessingImageSequenceWriter

try:
    from .ray import RayImageSequenceWriter

    ParallelImageSequenceWriter = RayImageSequenceWriter  # type: ignore
except ImportError:
    RayImageSequenceWriter = None  # type: ignore


__all__ = [
    "imwrite",
    "ImageSequenceWriter",
    "MultiprocessingImageSequenceWriter",
    "ParallelImageSequenceWriter",
    "RayImageSequenceWriter",
]
