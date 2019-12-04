# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import numpy as np

import numpy as np
import warnings

_use_anfft = True

try:
    import anfft
    _use_anfft = True
except ImportError:
    _use_anfft = False


def rfftfreqn(n, d = None):
    """Generate an array of FFT frequencies for an n-dimensional *real* transform.

    Parameters
    ----------
    n : tuple
        A tuple containing the dimensions of the window in each
        dimension.
    d : float or tuple, optional
        The sample spacing along each dimension

    Returns
    -------
    frequencies : ndarray
        An array of shape (n[0], n[1], ..., n[-1]/2+1, len(n)),
        containing the frequency vector of each sample.
    """

    n = np.array(n)
    if d is None:
        d = n
    else:
        if(len(d) != len(n)):
            raise Exception("Sample spacing array is the wrong length.")
        d *= n

    # Use slice objects and mgrid to produce an array with
    # zero-centered frequency vectors, except the last dimension (as
    # we are doing a real transform).
    mgslices = [slice(-x/2, x/2, 1.0) for x in n[:-1]] + [slice(0, n[-1]//2+1, 1.0)]
    mgarr = np.mgrid[mgslices]

    # Use fftshift to move the zero elements to the start of each
    # dimension (except the last dimension).
    shiftaxes = list(range(1,len(n)))
    shiftarr = np.fft.fftshift(mgarr, axes=shiftaxes)

    # Roll the first axis to the back so the components of the
    # frequency are the final axis.
    rollarr = np.rollaxis(shiftarr, 0, len(n)+1)

    # If we are given the sample spacing, scale each vector component
    # such that it is a proper frequency.
    rollarr[...,:] /= d

    return rollarr



def rfftn(arr):

    if arr.shape[-1] % 2 != 0:
        warnings.warn("Last axis length not multiple of 2. fftutil.irfftn will not reproduce this exactly.")

    if _use_anfft:
        print("Parallel FFT.")
        return anfft.rfftn(arr)
    else:
        return np.fft.rfftn(arr)

def irfftn(arr):
    if _use_anfft:
        print("Parallel IFFT.")
        return anfft.irfftn(arr)
    else:
        return np.fft.irfftn(arr)
