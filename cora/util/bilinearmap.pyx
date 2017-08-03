cimport cython
from cython.parallel import prange

import numpy as np
cimport numpy as np


cdef inline double clip(double x, double a, double b) nogil:
    return a if x < a else (b if x > b else x)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void interp(double[:, ::1] arr, double[::1] x, double[::1] y, double[::1] v) nogil:
    """Interpolate a 2D array onto new points.

    Parameters
    ----------
    arr : np.ndarray[:, :]
        Array to interpolate.
    x, y : np.ndarray[:]
        New co-ordinates.
    v : np.ndarray[:]
        Array to fill with interpolated values.
    """

    cdef unsigned int i, nx, ny, x0, x1, y0, y1
    cdef double xx, yy
    cdef double wa, wb, wc, wd
    cdef double Ia, Ib, Ic, Id
    cdef double ux, uy

    nx = arr.shape[0]
    ny = arr.shape[1]

    ux = nx - 1e-5
    uy = ny - 1e-5

    for i in prange(x.shape[0]):

        xx = clip(x[i], 0.0, ux)
        yy = clip(y[i], 0.0, uy)

        x0 = <unsigned int>xx
        x1 = x0 + 1
        y0 = <unsigned int>yy
        y1 = y0 + 1

        wa = (x1 - xx) * (y1 - yy)
        wb = (x1 - xx) * (yy - y0)
        wc = (xx - x0) * (y1 - yy)
        wd = (xx - x0) * (yy - y0)

        Ia = arr[x0, y0]
        Ib = arr[x0, y1]
        Ic = arr[x1, y0]
        Id = arr[x1, y1]

        v[i] = wa * Ia + wb * Ib + wc * Ic + wd * Id
