try:
    from .multiprocessing import dill_for_apply
except ImportError:
    dill_for_apply = None  # type: ignore
