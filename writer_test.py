import numpy as np

from mjecv.io import MultiprocessingImageSequenceWriter  # noqa: F401
from mjecv.io import RayImageSequenceWriter, imwrite

if __name__ == "__main__":
    size = 2000
    Writer = RayImageSequenceWriter
    max_waiting = None

    im = np.random.random((size, size))
    s = Writer("{index}.png", imwrite, max_waiting=max_waiting)
    for i in range(100):
        print("Saving", i)
        s.save(im)
    s.finish(lambda f: print("Finished", f))
