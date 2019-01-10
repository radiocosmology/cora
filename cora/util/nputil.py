"""Utility functions to help with pure numpy stuff."""

import numpy as np
import scipy.linalg as la

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


def matrix_root_manynull(mat, threshold=1e-16, truncate=True):
    """Square root a matrix.

    An inefficient alternative to the Cholesky decomposition for a
    matrix with a large dynamic range in eigenvalues. Numerical
    roundoff causes Cholesky to fail as if the matrix were not
    positive semi-definite. This does an explicit eigen-decomposition,
    setting small and negative eigenvalue to zero.

    Parameters
    ==========
    mat - ndarray
        An N x N matrix to decompose.
    threshold : scalar, optional
        Set any eigenvalues a factor `threshold` smaller than the
        largest eigenvalue to zero.
    truncate : boolean, optional
        If True (default), truncate the matrix root, to the number of positive
        eigenvalues.

    Returns
    =======
    root : ndarray
        The decomposed matrix. This is truncated to the number of
        non-zero eigen values (if truncate is set).
    num_pos : integer
            The number of positive eigenvalues (returned only if truncate is set).
    """

    # Try to perform a Cholesky first as it's much faster (8x)
    try:
        root = la.cholesky(mat, lower=True)
        num_pos = mat.shape[0]

    # If that doesn't work do an eigenvalue and throw out any tiny modes
    except la.LinAlgError:
        evals, evecs = la.eigh(mat)

        evals[np.where(evals < evals.max() * threshold)] = 0.0
        num_pos = len(np.flatnonzero(evals))

        if truncate:
            evals = evals[np.newaxis, -num_pos:]
            evecs = evecs[:, -num_pos:]

        root = (evecs * evals[np.newaxis, :]**0.5)

    if truncate:
        return root, num_pos
    else:
        return root


def complex_std_normal(shape):
    """Get a set of complex standard normal variables.

    Parameters
    ----------
    shape : tuple
        Shape of the array of variables.

    Returns
    -------
    var : np.ndarray[shape]
        Complex gaussian variates.
    """

    return (np.random.standard_normal(shape) + 1.0J * np.random.standard_normal(shape)) / 2**0.5
