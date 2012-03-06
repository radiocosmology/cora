
""" Module for interpolating sampled functions. """

import numpy as np

# Cython import
cimport numpy as np
cimport cython

# Import functions from math.h
cdef extern from "math.h":
    double exp(double x)
    double log(double x)

dbltype = np.int
ctypedef np.float64_t dbltype_t


def _int_fromfile(cls, file, colspec = None):
        """Perform Interpolation from given file. 

        Load up the file and generate an Interpolater.
        
        colspec - 2 element list of columns to interpolate
        """
        if colspec == None:
            colspec = [0,1]

        if len(colspec) != 2:
            print cls, file, colspec
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


    fromfile = classmethod(_int_fromfile)
    
    
    def __init__(self, data1, data2 = None):
        """Constructor to initialise from data. 

        Need to supply a 2d data array of X-Y pairs to
        interpolate. i.e. [[x0,y0], [x1,y1], ...]
        """
        
        if data2 == None:
            data = data1
        else:
            data = np.dstack((data1, data2))[0]

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

        self.__data_ = data
        self.__gen_spline_()


    def value(self, x):
        """Returns the value of the function at x. """
            
        return self.value_cdef(x)


    def __call__(self, x):
        """Returns the value of the function at x. """
        if isinstance(x, np.ndarray):
            r = np.empty_like(x)
            for index, xv in np.ndenumerate(x):
                r[index] = self.value_cdef(xv)
            return r
        
        return self.value_cdef(x)

    cdef dbltype_t value_cdef(self, dbltype_t x):
        """Returns the value of the function at x. 

        Cdef'd function to do the work.
        """

        cdef int kl,kh,kn
        cdef dbltype_t h,a,b,c,d

        cdef np.ndarray[dbltype_t, ndim=2] data = self.__data_
        cdef np.ndarray[dbltype_t, ndim=1] y2 = self.__y2_

        kl = 0
        kh = data.shape[0]

        if(x < data[0,0]):
            # Extrapolate gradient from first point
            h = data[1,0] - data[0,0]
            a = (data[1,1] - data[0,1]) / h
            return (a - h * y2[1] / 6) * (x - data[0,0]) + data[0,1]

        if(x >= data[kh-1,0]):
            # Extrapolate gradient from last point
            kh = kh-1
            h = data[kh,0] - data[kh-1,0]
            a = (data[kh,1] - data[kh-1,1]) / h
            return (a + h * y2[kh-1] / 6) * (x - data[kh,0]) + data[kh,1]

        # Use bisection to locate the interval
        while(kh - kl > 1):
            kn = (kh + kl) / 2
            if(data[kn, 0] > x):
                kh = kn
            else:
                kl = kn

        # Define variables as in NR.
        h = data[kh, 0] - data[kl, 0]

        a = (data[kh, 0] - x) / h
        b = (x - data[kl, 0]) / h
        c = (a**3 - a) * h**2 / 6
        d = (b**3 - b) * h**2 / 6
        
        # Return the spline interpolated value.
        return (a * data[kl, 1] + b * data[kh, 1] 
                + c * y2[kl] + d * y2[kh])


    def __gen_spline_(self):
        # Solve for the second derivative matrix to generate the
        # splines.
        
        n = self.__data_.shape[0]

        al = np.zeros(n-2)
        bt = np.zeros(n-2)
        gm = np.zeros(n-2)
        f = np.zeros(n-2)
        z = np.zeros(n-2)
        l = np.zeros(n-2)
        m = np.zeros(n-2)

        self.__y2_ = np.zeros(n)

        # Fill out matrices (al diagonal, bt lower, gm upper).
        y = self.__data_[:,1]
        x = self.__data_[:,0]

        for i in xrange(0, n-2):

            f[i] = (y[i+2] - y[i+1]) / (x[i+2] - x[i+1]) - (y[i+1] - y[i]) / (x[i+1] - x[i])

            al[i] = (x[i+2] - x[i]) / 3
            if(i != 0):
                bt[i] = (x[i+1] - x[i]) / 6
            if(i != n-3):
                gm[i] = (x[i+2] - x[i+1]) / 6

        # Solve for l and mu. To produce LU decomposition.
        l[0] = al[0]
        m[0] = gm[0] / al[0]

        for i in xrange(1, n-2):
            l[i] = al[i] - bt[i] * m[i-1]
            m[i] = gm[i] / l[i]

        # Solve L.z = f
        z[0] = f[0] / l[0]
        for i in xrange(1, n-2):
            z[i] = (f[i] - bt[i] * z[i-1]) / l[i]

        # Solve U.x = z
        for i in xrange(n-4,-1,-1):
            z[i] = z[i] - m[i] * z[i+1]
            
        # Set second derivatives.
        for i in xrange(1,n-1):
            self.__y2_[i] = z[i-1]

    def data(self):
        """Return the data array. """
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
    """Perform a cubic spline interpolation in log-space. """

    def __init__(self, data):
        """Initialise the interpolater. Data must not be negative or zero,
        otherwise logarithmic interpolation blows up. """

        if(np.any(data <= 0)):
            raise InterpolationException("Data must be non-negative.")

        Interpolater.__init__(self, np.log(data))

    def value(self, dbltype_t x):
        """ Return the value of the log-interpolated function."""

        return exp(Interpolater.value_cdef(self, log(x)))


    def __call__(self, x):
        """Returns the value of the function at x. """
        if isinstance(x, np.ndarray):
            x = np.log(x)
            r = np.empty_like(x)
            for index, xv in np.ndenumerate(x):
                r[index] = self.value_cdef(xv)
            return np.exp(r)
        
        return np.exp(self.value_cdef(np.log(x)))




