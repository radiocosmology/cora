import numpy as np

cimport numpy as np

cdef extern from "gsl/gsl_sf_bessel.h":
    double gsl_sf_bessel_jl(int l, double x)
    int gsl_sf_bessel_jl_e(int l, double x, double * r)
    void * gsl_set_error_handler_off ()

def _jl_gsl(l, z):
    cdef int e
    cdef double d[2]
    gsl_set_error_handler_off ()
    l,z = np.broadcast_arrays(l, z)
    r = np.zeros_like(l).astype(np.float64)
    for i in range(r.size):
        e = <int>gsl_sf_bessel_jl_e(<int>(l.flat[i]), <double>(z.flat[i]), d)
        #if e != 0:
        #    print l.flat[i], z.flat[i]
        #    print "Something went wrong in GSL."
        r.flat[i] = d[0]

    return r
