"""
Special functions for calculations on the sphere.

These routines require `pygsl` to be installed. Although these functions are
available in `scipy` they tend to be inaccurate at large values of `l`.
"""

import numpy as np
import scipy.special as ss


def jl(l, z):
    """A fast spherical bessel function using approximations at low and high z.

    Parameters
    ----------
    l, z : scalar or np.ndarray
        Spherical bessel function arguments.

    Returns
    -------
    jl : scalar or np.ndarray
        Spherical bessel functions.
    """

    lt = l
    zt = z

    l = np.atleast_1d(l)
    z = np.atleast_1d(z)

    zca = np.logical_and(z == 0.0, l > 0)
    zza = np.where(zca)
    lca = np.logical_and(
        np.logical_or(np.logical_and(l > 20, z < (l / 5.0)), z < (l * 1e-6)), z > 0.0
    )
    lza = np.where(lca)
    hca = np.logical_and(l > 20, z > 10.0 * l)
    hza = np.where(hca)
    gza = np.where(np.logical_not(np.logical_or(np.logical_or(lca, hca), zca)))
    la, za = np.broadcast_arrays(l, z)

    jla = np.empty_like(la).astype(np.float64)

    jla[zza] = 0.0
    jla[lza] = _jl_approx_lowz(la[lza], za[lza])
    jla[hza] = _jl_approx_highz(la[hza], za[hza])
    jla[gza] = ss.spherical_jn(la[gza], za[gza])

    if isinstance(lt, np.ndarray) or isinstance(zt, np.ndarray):
        return jla
    else:
        return jla[0]


def jl_d(l, z):
    """First derivative of jl.

    Parameters
    ----------
    l, z : scalar or np.ndarray
        Spherical bessel function arguments.

    Returns
    -------
    jl_d : scalar or np.ndarray
        Derivate of spherical bessel functions.
    """
    jl0 = jl(l, z)
    jl1 = jl(l + 1, z)
    return -jl1 + (l / z) * jl0


def jl_d2(l, z):
    """Second derivative of jl.

    Parameters
    ----------
    l, z : scalar or np.ndarray
        Spherical bessel function arguments.

    Returns
    -------
    jl_d2 : scalar or np.ndarray
        Second derivate of spherical bessel functions.
    """
    jl0 = jl(l, z)
    jl1 = jl(l + 1, z)
    jl2 = jl(l + 2, z)
    return jl2 - (2 * l / z) * jl1 + ((l**2 - 1) / z**2) * jl0


def Ylm(l, m, theta, phi):
    """Calculate the spherical harmonic functions.

    Parameters
    ----------
    l, m : int or array_like
        Multipoles to calculate.
    theta, phi : float or array_like
        Angular position.

    Returns
    -------
    ylm : float or array_like
    """

    l = np.array(l).astype(np.int32)
    m = np.array(m).astype(np.int32)
    x = np.array(np.cos(theta)).astype(np.float64)

    return sf.legendre_sphPlm(l, m, x) * np.exp(1.0j * m * phi)


def Ylm_array(lmax, theta, phi):
    """Calculate the spherical harmonics up to lmax.

    Parameters
    ----------
    lmax : integer
        Maximum multipole.
    theta, phi : float or array_like
        Angular position.

    Returns
    -------
    ylm : np.ndarray[..., (lmax + 1) * (lmax + 2) / 2]
        The spherical harmonics for each angular position. The output array is
        the same shape at `theta`, `phi` but with a new last axis for the
        multipoles.
    """
    m, l = np.triu_indices(lmax + 1)

    return Ylm(l, m, np.array(theta)[..., np.newaxis], np.array(phi)[..., np.newaxis])


def Ylm_spin2(l, m, theta, phi):
    """Evaluate the spin-2 spherical harmonics.

    Uses Eq. 14 from Wiaux et al. 2007 (arXiv:astro-ph/0508514).


    Parameters
    ----------
    l, m : int or array_like
        Multipoles to calculate.
    theta, phi : float or array_like
        Angular position.

    Returns
    -------
    ylm_spin_plus2, ylm_spin_minus2 : float or array_like
        Two arrays of the +2 and -2 spin harmonics.
    """

    def alpha(sign, l, m, theta):

        t = (
            2 * m**2
            - l * (l + 1.0)
            - sign * 2 * m * (l - 1.0) * np.cos(theta)
            + l * (l - 1) * np.cos(theta) ** 2
        )

        return t / np.sin(theta) ** 2

    def beta(sign, l, m, theta):

        t = (
            2
            * ((2.0 * l + 1.0) / (2.0 * l - 1.0) * (l**2 - m**2)) ** 0.5
            * (sign * m + np.cos(theta))
        )

        return t / np.sin(theta) ** 2

    y0 = Ylm(l, m, theta, phi)
    y1 = np.where(l <= m, 0.0, Ylm(l - 1, m, theta, phi))

    fac = (l - 1) * l * (l + 1) * (l + 2)
    fac = np.where(l < 2, 0.0, fac**-0.5)

    y2plus = fac * (alpha(1, l, m, theta) * y0 + beta(1, l, m, theta) * y1)
    y2minus = fac * (alpha(-1, l, m, theta) * y0 + beta(-1, l, m, theta) * y1)

    return y2plus, y2minus


def Ylm_spin2_array(lmax, theta, phi):
    """Calculate the spin-2 spherical harmonics up to lmax.

    Parameters
    ----------
    lmax : integer
        Maximum multipole.
    theta, phi : float or array_like
        Angular position.

    Returns
    -------
    ylm_spin_plus2, ylm_spin_minus2 : np.ndarray[..., (lmax + 1) * (lmax + 2) / 2]
        The spin spherical harmonics for each angular position. The output array is
        the same shape at `theta`, `phi` but with a new last axis for the
        multipoles.
    """
    m, l = np.triu_indices(lmax + 1)

    return Ylm_spin2(
        l, m, np.array(theta)[..., np.newaxis], np.array(phi)[..., np.newaxis]
    )


def _jl_approx_lowz(l, z):
    """Approximation of j_l for low z.

    From Gradsteyn and Ryzhik 8.452.
    """

    nu = l + 0.5
    nutanha = (nu * nu - z * z) ** 0.5
    arg = nutanha - nu * np.arccosh(nu / z)

    return (
        np.exp(arg)
        * (
            1
            + 1.0 / (8 * nutanha)
            - 5.0 * nu**2 / (24 * nutanha**3)
            + 9.0 / (128 * nutanha**2)
            - 231.0 * nu**2 / (576 * nutanha**4)
        )
        / (2 * (z * nutanha) ** 0.5)
    )


def _jl_approx_highz(l, z):
    """Approximation of j_l for large z (where z > l).

    From Gradsteyn and Ryzhik 8.453.
    """

    nu = l + 0.5
    sinb = (1 - nu * nu / (z * z)) ** 0.5
    cotb = nu / (z * sinb)
    arg = z * sinb - nu * (np.pi / 2 - np.arcsin(nu / z)) - np.pi / 4

    return (
        np.cos(arg) * (1 - 9.0 * cotb**2 / (128.0 * nu**2))
        + np.sin(arg) * (cotb / (8 * nu))
    ) / (z * sinb**0.5)


def _jl_gsl(l, z):
    """Use GSL routines to calculate Spherical Bessel functions.

    Array arguments only.
    """
    return sf.bessel_jl(l.astype(np.int32), z.astype(np.float64))
