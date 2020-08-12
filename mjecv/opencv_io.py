import multiprocessing

import cv2
import numpy as np

__all__ = ["ImageSequenceSaver", "MultithreadedImageSequenceSaver"]


class ImageSequenceSaver:
    def __init__(self, pattern, *, max_index=None):
        assert type(pattern) is str and "{" in pattern
        self._pattern = pattern
        self._max_index = max_index
        self._index = 1

    def _save(self, filename: str, image: np.ndarray):
        cv2.imwrite(filename, image)

    def save(self, image: np.ndarray):
        index = str(self._index)
        if self._max_index:
            index = "{:0{}d}".format(self._index, len(str(self._max_index)))
        filename = self._pattern.format(self._index, index=index)
        self._save(filename, image)
        self._index += 1

    def finish(self):
        pass


class MultithreadedImageSequenceSaver(ImageSequenceSaver):
    def __init__(self, *args, max_threads=None, **kwargs):
        super().__init__(*args, **kwargs)
        if max_threads is None:
            max_threads = multiprocessing.cpu_count() - 1
        self._pool = multiprocessing.get_context("spawn").Pool(max_threads)
        self._results = []

    def _save(self, filename: str, image: np.ndarray):
        result = self._pool.apply_async(cv2.imwrite, (filename, image))
        self._results.append(result)

    def finish(self):
        self._pool.close()
        for result in self._results:
            result.get()
        self._pool.join()
