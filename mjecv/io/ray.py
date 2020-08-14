from typing import List

import numpy as np
import ray

from .base import ImageSequenceWriter


@ray.remote(num_cpus=1)
def _ray_save(writer, filename: str, image: np.ndarray):
    writer(filename, image)
    return filename


class RayImageSequenceWriter(ImageSequenceWriter):
    def __init__(self, *args, num_cpus=None, max_waiting=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not ray.is_initialized():
            # By default ray reserves 30% of available memory for the object store, but
            # image writer tasks hardly need any, so use more.
            mem = ray.utils.estimate_available_memory()*0.7
            ray.init(num_cpus=num_cpus, object_store_memory=mem)
        self._max_waiting = max_waiting
        self._waiting = []  # type: List[ray.ObjectID]

    def _wait(self):
        ready, self._waiting = ray.wait(self._waiting)
        return ready

    def _save(self, filename: str, image: np.ndarray):
        if self._max_waiting and len(self._waiting) >= self._max_waiting:
            self._wait()
        result_id = _ray_save.remote(self._writer, filename, image)
        self._waiting.append(result_id)

    def finish(self, result_handler=None):
        try:
            while self._waiting:
                ready = self._wait()
                if result_handler:
                    for filename_id in ready:
                        result_handler(ray.get(filename_id))
        except KeyboardInterrupt:
            for result_id in self._waiting:
                ray.cancel(result_id)
