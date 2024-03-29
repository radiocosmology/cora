"""Module for interpolating sampled functions."""

import numpy as np

# Cython import
cimport numpy as np
cimport cython

from cython.parallel import prange

from libc.math cimport exp, log, sinh, asinh

ctypedef np.float64_t dbltype_t


def _int_fromfile(cls, file, colspec=None):
        """Perform Interpolation from given file.

        Load up the file and generate an Interpolater.

        colspec - 2 element list of columns to interpolate
        """
        if colspec is None:
            colspec = [0,1]

        if len(colspec) != 2:
            print(f"cls: {cls}, file: {file}, colspec: {colspec}")
            raise InterpolationException("Can only use two columns.")

        d1 = np.loadtxt(file, usecols = colspec)
        return cls(d1)

class InterpolationException(Exception):
    """Exceptions in the Interpolation module. """
    pass


cdef class Interpolater(object):
    """A class to interpolate data sets.

    This class performs cubic spline interpolation. See
    Numerical Recipes for details. Uses Natural spline conditions,
    y''[0] = y''[-1] = 0. Written using Cython for massive speedup.
    """

    cdef np.ndarray __data_
    cdef np.ndarray __y2_

    cdef double * _data_p
    cdef double * _y2_p
    cdef Py_ssize_t _n


    fromfile = classmethod(_int_fromfile)


    def __init__(self, data1, data2=None):
        """Constructor to initialise from data.

        Need to supply a 2d data array of X-Y pairs to
        interpolate. i.e. [[x0,y0], [x1,y1], ...]
        """

        if data2 is None:
            data = data1
        else:
            try:
                data = np.dstack((data1, data2))[0]
            except ValueError as e:
                raise InterpolationException("Failure stacking x and y data.") from e

        # Check to ensure array contains correct data.
        s = data.shape
        if(len(s) != 2):
            raise InterpolationException("Array must be 2d.")
        if(s[1] != 2):
            raise InterpolationException("Array must consist of X-Y pairs.")
        if(s[0] < 4):
            raise InterpolationException("Cubic spline interpolation requires at least 4 points.")

        # Check to ensure data is all valid.
        if(np.isinf(data).any() or np.isnan(data).any()):
            raise InterpolationException("Some values invalid.")

        self.__data_ = np.ascontiguousarray(data).astype(np.float64, copy=False)
        self.__gen_spline_()

        self._data_p = <double *>self.__data_.data
        self._y2_p = <double *>self.__y2_.data
        self._n = <Py_ssize_t>self.__data_.shape[0]


    def value(self, x):
        """Returns the value of the function at x. """
        if isinstance(x, np.ndarray):
            return self.value_array(x)

        return self.value_cdef(x)



    def __call__(self, x):
        """Returns the value of the function at x."""
        return self.value(x)

    @cython.boundscheck(False)
    def value_array(self, x):


        cdef Py_ssize_t i, nr

        cdef double[::1] xr = np.ravel(x, order='C')
        r = np.empty_like(x)
        cdef double[::1] rr = r.ravel()

        nr = xr.size

        for i in prange(nr, nogil=True):
            rr[i] = self.value_cdef(xr[i])

        return r


    @cython.cdivision(True)
    @cython.boundscheck(False)
    cdef dbltype_t value_cdef(self, dbltype_t x) nogil:
        """Returns the value of the function at x.

        Cdef'd function to do the work.
        """

        cdef int kl,kh,kn
        cdef dbltype_t h,a,b,c,d

        cdef double * data = self._data_p
        cdef double * y2 = self._y2_p

        # cdef np.ndarray[dbltype_t, ndim=2] data = self.__data_
        # cdef np.ndarray[dbltype_t, ndim=1] y2 = self.__y2_

        kl = 0
        kh = self._n

        if(x < data[0]):
            # Extrapolate gradient from first point
            h = data[2] - data[0]
            a = (data[3] - data[1]) / h
            return (a - h * y2[1] / 6) * (x - data[0]) + data[1]

        if(x >= data[2*(kh-1)]):
            # Extrapolate gradient from last point
            kh = kh-1
            h = data[2*kh] - data[2*(kh-1)]
            a = (data[2*kh+1] - data[2*(kh-1)+1]) / h
            return (a + h * y2[kh-1] / 6) * (x - data[2*kh]) + data[2*kh+1]

        # Use bisection to locate the interval
        while(kh - kl > 1):
            kn = (kh + kl) / 2
            if(data[2*kn] > x):
                kh = kn
            else:
                kl = kn

        # Define variables as in NR.
        h = data[2*kh] - data[2*kl]

        a = (data[2*kh] - x) / h
        b = (x - data[2*kl]) / h
        c = (a**3 - a) * h**2 / 6
        d = (b**3 - b) * h**2 / 6

        # Return the spline interpolated value.
        return (a * data[2*kl+1] + b * data[2*kh+1]
                + c * y2[kl] + d * y2[kh])

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def __gen_spline_(self):
        # Solve for the second derivative matrix to generate the
        # splines.

        cdef Py_ssize_t n = self.__data_.shape[0]
        cdef Py_ssize_t length = n - 2
        cdef Py_ssize_t i

        cdef double[::1] al = np.zeros(length, dtype=np.float64)
        cdef double[::1] bt = np.zeros(length, dtype=np.float64)
        cdef double[::1] gm = np.zeros(length, dtype=np.float64)
        cdef double[::1] f = np.zeros(length, dtype=np.float64)
        cdef double[::1] z = np.zeros(length, dtype=np.float64)
        cdef double[::1] l = np.zeros(length, dtype=np.float64)
        cdef double[::1] m = np.zeros(length, dtype=np.float64)

        self.__y2_ = np.zeros(n, dtype=np.float64)
        cdef double[:] y2 = self.__y2_

        # Fill out matrices (al diagonal, bt lower, gm upper).
        cdef double[:] y = self.__data_[:,1]
        cdef double[:] x = self.__data_[:,0]

        for i in xrange(length):

            f[i] = (y[i+2] - y[i+1]) / (x[i+2] - x[i+1]) - (y[i+1] - y[i]) / (x[i+1] - x[i])

            al[i] = (x[i+2] - x[i]) / 3
            if(i != 0):
                bt[i] = (x[i+1] - x[i]) / 6
            if(i != n-3):
                gm[i] = (x[i+2] - x[i+1]) / 6

        # Solve for l and mu. To produce LU decomposition.
        l[0] = al[0]
        m[0] = gm[0] / al[0]

        for i in xrange(1, length):
            l[i] = al[i] - bt[i] * m[i-1]
            m[i] = gm[i] / l[i]

        # Solve L.z = f
        z[0] = f[0] / l[0]
        for i in xrange(1, length):
            z[i] = (f[i] - bt[i] * z[i-1]) / l[i]

        # Solve U.x = z
        for i in xrange(n-4,-1,-1):
            z[i] = z[i] - m[i] * z[i+1]

        # Set second derivatives.
        for i in xrange(1,length + 1):
            y2[i] = z[i-1]

    def data(self):
        """Return the data array."""
        return (self.__data_, self.__y2_)

    def test(self, min, max, samp):
        """Take regular samples from the Interpolater.

        Take the interpolater, and take samp samples between min and
        max. Returning an array.
        """

        a = np.zeros((samp, 2))
        h = 1.0 * (max - min) /  samp

        for i in xrange(samp):
            a[i,0] = min + i*h
            a[i,1] = self.value(min + i*h)

        return a


cdef class LogInterpolater(Interpolater):
    """Perform a cubic spline interpolation in log-space."""

    def __init__(self, data):
        """Initialise the interpolater. Data must not be negative or zero,
        otherwise logarithmic interpolation blows up."""

        if(np.any(data <= 0)):
            raise InterpolationException("Data must be non-negative.")

        Interpolater.__init__(self, np.log(data))

    def value(self, x):
        """ Return the value of the log-interpolated function."""
        if isinstance(x, np.ndarray):
            return self.value_log_array(x)

        return exp(self.value_cdef(log(x)))


    @cython.boundscheck(False)
    def value_log_array(self, x):

        cdef double[::1] xr = np.ravel(x, order='C')
        r = np.empty_like(x)
        cdef double[:] rr = r.ravel()

        cdef Py_ssize_t i, nr

        nr = xr.size

        for i in prange(nr, nogil=True):
            rr[i] = exp(self.value_cdef(log(xr[i])))

        return r


cdef class SinhInterpolater(Interpolater):
    """Perform a cubic spline interpolation in a sinh scaled space.

    The interpolation is done within the space of `arcsinh(x / x_t)` and `arcsinh(f /
    f_t)`, this gives an effective log interpolation for absolute values greater than
    the threshold and linear when less than the threshold. This allows an effective
    log interpolation which will handle zero and negative values.

    Parameters
    ----------
    data : np.ndarray[:, 2]
        The data points packed as [[x0, f0], ...].
    x_t
        The sinc threshold value for the positions.
    f_t
        The function value threshold.
    """

    cdef double x_t
    cdef double f_t

    def __init__(self, data, x_t, f_t):
        self.x_t = x_t
        self.f_t = f_t

        thresholds = np.array([x_t, f_t], dtype=np.float64)
        Interpolater.__init__(self, np.arcsinh(data / thresholds))

    def value(self, x):
        """ Return the value of the log-interpolated function."""
        if isinstance(x, np.ndarray):
            return self.value_sinh_array(x)

        return self.f_t * sinh(self.value_cdef(asinh(x / self.x_t)))


    @cython.boundscheck(False)
    def value_sinh_array(self, x):

        cdef np.ndarray[dbltype_t, ndim=1] xr = np.ravel(x, order='C')
        r = np.empty_like(x)
        cdef np.ndarray[dbltype_t, ndim=1] rr = r.ravel()

        cdef Py_ssize_t i, nr


        nr = xr.size

        for i in prange(nr, nogil=True):
            rr[i] = self.f_t * sinh(self.value_cdef(asinh(xr[i] / self.x_t)))

        return r
