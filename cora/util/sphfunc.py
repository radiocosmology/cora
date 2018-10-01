"""Special functions for calculations on the sphere.

These routines require `pygsl` to be installed. Although these functions are
available in `scipy` they tend to be inaccurate at large values of `l`.
"""

import numpy as np


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

    return sf.legendre_sphPlm(l, m, x) * np.exp(1.0J * m * phi)


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

        t = (2 * m**2 - l * (l + 1.0) -
             sign * 2 * m * (l - 1.0) * np.cos(theta) +
             l * (l - 1) * np.cos(theta)**2)

        return t / np.sin(theta)**2

    def beta(sign, l, m, theta):

        t = (2 * ((2.0 * l + 1.0) / (2.0 * l - 1.0) * (l**2 - m**2))**0.5 *
             (sign * m + np.cos(theta)))

        return t / np.sin(theta)**2

    y0 = Ylm(l, m, theta, phi)
    y1 = np.where(l <= m, 0.0, Ylm(l - 1, m, theta, phi))

    fac = (l - 1) * l * (l + 1) * (l + 2)
    fac = np.where(l < 2, 0.0, fac**-0.5)

    y2plus = fac * (alpha(1, l, m, theta) * y0 + beta(1, l, m, theta) * y1)
    y2minus = fac * (alpha(-1, l, m, theta) * y0 + beta(-1, l, m, theta) * y1)

    return y2plus, y2minus


def _compute_coefficients(L):
    """Precompute coefficients `a_l^m` and `b_l^m`.
    """

    coeff = np.zeros((2, L + 1, L + 1))

    t = np.arange(L + 1)
    l, m = t[:, np.newaxis], t[np.newaxis, :]

    coeff[0] = ((4 * l**2 - 1.) / (l**2 - m**2))**0.5
    coeff[1] = -(((l - 1)**2 - m**2) / (4 * (l - 1)**2 - 1.))**0.5

    return coeff


_SCL_OFFSET = 1e-280


def _compute_p(L, x, coeff, P):
    """Compute an entire set of scaled, renormalised Legendre Polynomials using
    the given coefficients, and store in the array P. """

    assert coeff.shape == (2, L + 1, L + 1)
    assert P.shape == (L + 1, L + 1)

    # Initialise l = 0
    P[:] = 0.0
    P[0, 0] = (4 * np.pi)**-0.5 * _SCL_OFFSET

    if L > 0:

        # Initialise l = 1
        P[1, 1] = -(3.0 / 2.0)**0.5 * P[0, 0]
        P[1, 0] = 3.0**0.5 * x * P[0, 0]

        for l in range(2, L + 1):

            mlim = l - 1

            # Use the recurrence relation to generate higher l
            P[l, :mlim] = coeff[0, l, :mlim] * (x * P[l - 1, :mlim] +
                                                coeff[1, l, :mlim] * P[l - 2, :mlim])

            # Set the last elements for this l
            P[l, l - 1] = x * (2 * (l - 1) + 3)**0.5 * P[l - 1, l - 1]
            P[l, l] = -(1.0 + 0.5 / l)**0.5 * P[l - 1, l - 1]

    return P


def Ylm_array(L, theta, phi, extended_precision=False, output='square'):
    """Compute spherical harmonics up to degree L at given positions.

    This routine carefully evaluates the spherical harmonics using a scaled
    recurrence relation. In particular it uses the `modified forward column
    method` of Holmes and Featherstone, Journal of Geodesy 76.5 (2002): 279-299.
    APA

    The code is a ported and very heavily modified version of
    https://github.com/milthorpe/SphericalHarmonics.jl.

    Parameters
    ----------
    L : integer
        Maximum l to calculate (inclusive).
    theta, phi : float or array_like
        Angular position on the sphere (radians).
    output : {'column', 'row', 'square'}
        The output format. Either the (l, m) triangle column-major (Healpix
        style), row-major or the full (l, m) square (with zeros in the upper
        triangle).
    extended_precision : bool
        Use and return `long double` type for calculation. When set it seems to
        be accurate for ``l < 6000``, when not set (default) the routine has
        numercical isses for ``l > 2000``.

    Returns
    -------
    Y : np.ndarray[ps ind..., lm ind]
    """

    with np.errstate(divide='ignore', invalid='ignore'):

        # Pick the precision to use for the calculation
        dt = np.longdouble if extended_precision else np.float64

        # Precompute the coefficients. These are position independent and so can be
        # reused
        coeff = _compute_coefficients(L)
        m = np.arange(L + 1)[np.newaxis, :]

        def _ylm(t, p):
            # Worker routine for calculating the Y_lms for a single position
            x = np.cos(t)
            u = np.sin(t)

            # Compute the Legendre's
            P = np.zeros((L + 1, L + 1), dtype=dt)
            _compute_p(L, x, coeff, P)

            # Carefully turn into the Y_lm's
            logscl = m * np.log(u) - np.log(_SCL_OFFSET)
            Y = (np.sign(P) * np.exp(np.log(np.abs(P)) + logscl) *
                 np.exp(1.0J * m * p))

            if output == 'column':
                Y = Y.T[np.triu_indices(L + 1)].copy()
            elif output == 'row':
                Y = Y[np.tril_indices(L + 1)].copy()
            elif output == 'square':
                pass
            else:
                raise ValueError('Unrecognized output format: %s' % output)

            return Y

        b = np.broadcast(theta, phi)
        Y = np.array([_ylm(t, p) for t, p in b])
        Y = Y.reshape(b.shape + Y.shape[1:])

    return Y


def Ylm_spin2_array(L, theta, phi, extended_precision=False, output='square'):
    """Evaluate the spin-2 spherical harmonics.

    Uses Eq. 14 from Wiaux et al. 2007 (arXiv:astro-ph/0508514).


    Parameters
    ----------
    L : integer
        Maximum multipole.
    theta, phi : float or array_like
        Angular position.

    Returns
    -------
    ylm_spin_plus2, ylm_spin_minus2 : float or array_like
        Two arrays of the +2 and -2 spin harmonics. Only supports the square
        output format.
    """

    # Need to be really careful with the indexing and broadcasting in this
    # function, it's quite subtle

    with np.errstate(divide='ignore', invalid='ignore'):

        def alpha(sign, l, m, theta):

            t = (2 * m**2 - l * (l + 1.0) -
                 sign * 2 * m * (l - 1.0) * np.cos(theta) +
                 l * (l - 1) * np.cos(theta)**2)

            return np.where(m > l, 0.0, t / np.sin(theta)**2)

        def beta(sign, l, m, theta):

            t = (2 * ((2.0 * l + 1.0) / (2.0 * l - 1.0) * (l**2 - m**2))**0.5 *
                 (sign * m + np.cos(theta)))

            return np.where(m > l, 0.0, t / np.sin(theta)**2)

        t = np.arange(L + 1)
        l, m = t[:, np.newaxis], t[np.newaxis, :]

        y0 = Ylm_array(L, theta, phi, output='square')
        y1 = np.roll(y0, 1, axis=-2)
        y1[..., 0, :] = 0.0

        fac = (l - 1) * l * (l + 1) * (l + 2)
        fac = np.where(l < 2, 0.0, fac**-0.5)

        # Expand the axes so that we successfully broadcast against the l, m
        # indices
        theta = np.array(theta)[..., np.newaxis, np.newaxis]

        y2plus = fac * (alpha(1, l, m, theta) * y0 + beta(1, l, m, theta) * y1)
        y2minus = fac * (alpha(-1, l, m, theta) * y0 + beta(-1, l, m, theta) * y1)

    return y2plus, y2minus


def _jl_approx_lowz(l, z):
    """Approximation of j_l for low z.

    From Gradsteyn and Ryzhik 8.452.
    """

    nu = l + 0.5
    nutanha = (nu * nu - z * z)**0.5
    arg = nutanha - nu * np.arccosh(nu / z)

    return (np.exp(arg) * (1 + 1.0 / (8 * nutanha) - 5.0 * nu**2 / (24 * nutanha**3) +
                           9.0 / (128 * nutanha**2) - 231.0 * nu**2 / (576 * nutanha**4)) /
            (2 * (z * nutanha)**0.5))


def _jl_approx_highz(l, z):
    """Approximation of j_l for large z (where z > l).

    From Gradsteyn and Ryzhik 8.453.
    """

    nu = l + 0.5
    sinb = (1 - nu * nu / (z * z))**0.5
    cotb = nu / (z * sinb)
    arg = z * sinb - nu * (np.pi / 2 - np.arcsin(nu / z)) - np.pi / 4

    return (np.cos(arg) * (1 - 9.0 * cotb**2 / (128.0 * nu**2)) +
            np.sin(arg) * (cotb / (8 * nu))) / (z * sinb**0.5)


def _jl_gsl(l, z):
    """Use GSL routines to calculate Spherical Bessel functions.

    Array arguments only.
    """
    return sf.bessel_jl(l.astype(np.int32), z.astype(np.float64))
