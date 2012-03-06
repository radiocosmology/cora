"""Utility functions to help with pure numpy stuff."""

import numpy as np

def save_ndarray_list(fname, la):
    """Save a list of numpy arrays to disk.

    This is designed so it can be reloaded exactly (with the exact
    same ordering) by `load_ndarray_list`.

    Parameters
    ----------
    fname : string
        filename to save to.
    la : list of np.ndarrays
        list of arrays to save.
    """
    d1 = { repr(i) : v for i, v in enumerate(la)}

    np.savez(fname, **d1)

def load_ndarray_list(fname):
    """Load a list of arrays saved by `save_ndarray_list`.

    Parameters
    ----------
    fname : string
        filename to load.

    Returns
    -------
    la : list of np.ndarrays
        The list of loaded numpy arrays. This should be identical tp
        what was saved by `save_ndarray_list`.
    """

    d1 = np.load(fname)
    la = [ v for i, v in sorted(d1.iteritems(), key=lambda kv: int(kv[0]))]

    return la
