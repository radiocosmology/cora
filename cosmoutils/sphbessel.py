import numpy as np

from _sphbessel_c import _jl_gsl

def jl_d(l, z):
    jl0 = jl(l, z)
    jl1 = jl(l+1, z)
    return -jl1 + (l / z) *jl0

def jl_d2(l, z):
    jl0 = jl(l, z)
    jl1 = jl(l+1, z)
    jl2 = jl(l+2, z)
    return jl2 - (2 * l / z) * jl1 + ((l**2 - 1) / z**2) *jl0


def jl(l, z):

    lt = l
    zt = z

    l = np.atleast_1d(l)
    z = np.atleast_1d(z)

    zca = np.logical_and(z == 0.0, l > 0)
    zza = np.where(zca)
    lca = np.logical_and(np.logical_or(np.logical_and(l > 20, z < (l / 5.0)), z < (l * 1e-6)), z > 0.0)
    lza = np.where(lca)
    hca = np.logical_and(l > 20, z > 10.0 * l)
    hza = np.where(hca)
    gza = np.where(np.logical_not(np.logical_or(np.logical_or(lca, hca), zca)))
    la, za = np.broadcast_arrays(l, z)

    jla = np.empty_like(la).astype(np.float64)

    jla[zza] = 0.0
    jla[lza] = _jl_approx_lowz(la[lza], za[lza])
    jla[hza] = _jl_approx_highz(la[hza], za[hza])
    jla[gza] = _jl_gsl(la[gza], za[gza])

    if isinstance(lt, np.ndarray) or isinstance(zt, np.ndarray):
        return jla
    else:
        return jla[0]

def _jl_approx_lowz(l, z):
    
    nu = l + 0.5
    nutanha = (nu * nu - z * z)**0.5
    arg = nutanha - nu * np.arccosh(nu / z)
    
    return (np.exp(arg) * (1 + 1.0 / (8 * nutanha) - 5.0 * nu**2 / (24 * nutanha**3)
                           + 9.0 / (128 * nutanha**2) - 231.0 * nu**2 / (576 * nutanha**4))
            / (2 * (z * nutanha)**0.5))


def _jl_approx_highz(l, z):
    
    nu = l + 0.5
    sinb = (1 - nu*nu / (z*z))**0.5
    cotb = nu / (z * sinb)
    arg = z * sinb - nu * (np.pi / 2 - np.arcsin(nu / z)) - np.pi / 4
    
    return (np.cos(arg) * (1 - 9.0 * cotb**2 / (128.0 * nu**2)) 
            + np.sin(arg) * (cotb / (8 * nu))) / (z * sinb**0.5)

