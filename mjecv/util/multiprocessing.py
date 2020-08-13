"""Helper functions for using dill with multiprocessing.

https://stackoverflow.com/a/32807763
"""

import dill


def _undill_for_apply(dilled_function, *args, **kwargs):
    function = dill.loads(dilled_function)
    return function(*args, **kwargs)


def dill_for_apply(function, *args, **kwargs):
    dilled_function = dill.dumps(function)
    return _undill_for_apply, (dilled_function, *args), kwargs
