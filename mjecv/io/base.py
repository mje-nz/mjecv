import multiprocessing
from typing import List, Optional

import numpy as np

from ..util import dill_for_apply


class ImageSequenceWriter:
    def __init__(self, pattern, writer, *, max_index=None):
        if type(pattern) is not str:
            raise ValueError("Pattern must be string")
        if pattern.format(1, index="1") == pattern.format(2, index="2"):
            raise ValueError("Pattern must use {} or {index}")
        self._pattern = pattern
        self._writer = writer
        self._max_index = max_index
        self._index = 1

    @property
    def next_filename(self):
        index = str(self._index)
        if self._max_index:
            index = "{:0{}d}".format(self._index, len(str(self._max_index)))
        return self._pattern.format(self._index, index=index)

    def _save(self, filename: str, image: np.ndarray):
        self._writer(filename, image)

    def save(self, image: np.ndarray):
        self._save(self.next_filename, image)
        self._index += 1

    def finish(self):
        pass


class MultiprocessingImageSequenceWriter(ImageSequenceWriter):
    """Image sequence writer that uses multiprocessing to save several images in
    parallel.

    This falls apart for large objects, as multiprocessing pickles them and pipes them
    into the subprocesses.
    """

    def __init__(self, *args, max_workers=None, max_waiting=None, **kwargs):
        super().__init__(*args, **kwargs)
        if max_workers is None:
            max_workers = multiprocessing.cpu_count() - 1
        ctx = multiprocessing.get_context("spawn")
        self._pool = ctx.Pool(max_workers)
        if max_waiting is not None:
            # Semaphore's value is number of slots available for tasks to wait in
            self._sem = ctx.Semaphore(
                max_waiting
            )  # type: Optional[multiprocessing.synchronize.Semaphore]
        else:
            self._sem = None
        self._results = []  # type: List[multiprocessing.pool.AsyncResult]

    def __del__(self):
        self.terminate()

    def _save(self, filename: str, image: np.ndarray):
        # Limit number of waiting tasks
        if self._sem:
            self._sem.acquire()

            def callback(v):
                assert self._sem is not None
                self._sem.release()

        else:
            callback = None  # type: ignore
        args = (self._writer, (filename, image))
        if dill_for_apply:
            # Use dill instead of pickle, and make sure writer returns the filename
            _writer = self._writer  # Exclude self from capture to avoid dilling _pool
            args = dill_for_apply(lambda f, i: _writer(f, i) or f, filename, image)
        result = self._pool.apply_async(
            *args, callback=callback, error_callback=callback,
        )
        self._results.append(result)

    def terminate(self):
        self._pool.terminate()
        self._pool.join()

    def finish(self, result_handler=None):
        try:
            # self._pool.close()
            for result in self._results:
                filename = result.get()
                if result_handler is not None:
                    result_handler(filename)
            self._pool.close()
        except KeyboardInterrupt:
            self._pool.terminate()
        finally:
            self._pool.join()
