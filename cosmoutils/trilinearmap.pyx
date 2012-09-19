# Provide a simple replacement for scipy.ndimage.map_coordinates
# for use in corr.py, in the hopes of avoiding the scinet bug

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport floor, ceil

DTYPE = np.float
ctypedef np.float_t DTYPE_t

#cdef inline int floor(double a): return <int>a if a>=0.0 else (<int>a)-1
#cdef inline int ceil(double a): return <int>a+1 if a>=0.0 else (<int>a)-1

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)

def map_coordinates(np.ndarray[DTYPE_t, ndim=3] input, np.ndarray[DTYPE_t, ndim=3] coords, int order=1):
    """
    Quick and dirty replacement of scipy.ndimage.map_coordinates,
    which has a bug at large spatial resolutions.
    Does not pretend to replace all functionality, just the bare
    minimum. Written by Greg Paciga.

    Inupts:
    -------------------------
    input: ndarray, 3 dimensions, the input array we will sample
    coords: an array such that the first axis gives the three
      coordinates of the input array to sample, and the second two
      are the coordinates of the output array at which to put the
      value found from input.
    order: in theory this would specify the order of the spline
      interpolation. Only accepts 0 or 1, and 1 only works if
      the arrays are 3D.

    Returns:
    -------------------------
    output: ndarray, with the shape of coords after dropping the
      first axis.
    """

    cdef unsigned int ix, iy
    cdef int ndim, nx, ny
    cdef int x0, x1, y0, y1, z0, z1
    cdef double x, y, z, xd, yd, zd
    cdef double c00, c10, c01, c11, c0, c1, c
    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] output
    cdef np.ndarray[DTYPE_t, ndim=1] coord #only used for nearest neighbour


    ndim = coords.shape[0]
    nx = coords.shape[1]
    ny = coords.shape[2]

    if ndim != input.ndim:
        print "Dimensions don't match"
        return False

    output = np.zeros((nx, ny))

    if order == 1 and ndim == 3:
        #print "doing trilinear interpolation"
        mode="trilinear"
    else:
        print "doing nearest neighbour"

    for ix in range(0,nx):
        for iy in range(0,ny):

            ## too fast to care
            #if (iy + ny*ix) % 500000 == 0:
            #    print "done %d of %d" % (iy + ny*ix, nx*ny)

            if mode is "trilinear":
                # do trilinear interpolation

                # the requested points
                x = coords[0,ix,iy]
                y = coords[1,ix,iy]
                z = coords[2,ix,iy]

                # points just below the requested point
                x0 = <int>floor(x)
                y0 = <int>floor(y)
                z0 = <int>floor(z)

                # points just above
                x1 = <int>ceil(x)
                y1 = <int>ceil(y)
                z1 = <int>ceil(z)

                # distance from lower point to requested point on each axis
                # if we're exactly on the point, no interpolation required
		# distance (x1-x0) is otherwise guaranteed to be 1.0
                if x1 == x0:
                    xd = 0.0
                else:
                    #xd = (x - x0)/(x1 - x0)
                    xd = (x - x0)

                if y1 == y0:
                    yd = 0.0
                else:
                    yd = (y - y0)

                if z1 == z0:
                    zd = 0.0
                else:
                    zd = (z - z0)

                # interpolate along x, to get the four points around
                # our requested point in that plane
                c00 = input[x0,y0,z0]*(1-xd) + input[x1,y0,z0]*xd
                c10 = input[x0,y1,z0]*(1-xd) + input[x1,y1,z0]*xd
                c01 = input[x0,y0,z1]*(1-xd) + input[x1,y0,z1]*xd
                c11 = input[x0,y1,z1]*(1-xd) + input[x1,y1,z1]*xd

                # now along y, to get the points directly above and below in z
                c0 = c00*(1-yd) + c10*yd
                c1 = c01*(1-yd) + c11*yd

                # and finally, interpolate along z
                c = c0*(1-zd) + c1*zd

                output[ix, iy] = c


            else:
                sample = input
                coord = coords[:,ix,iy]
                for dim in coord:
                    sample = sample[np.rint(dim)]

                output[ix, iy] = sample


    return output


