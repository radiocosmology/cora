from typing import Callable, Union, Optional, Tuple

import numpy as np
import scipy.integrate as si
import scipy.special as ss
from scipy.fftpack import dct

import hankel

from cora.util import bilinearmap

from .lssutil import sinh_interpolate


def _corr_direct(psfunc, log_k0, log_k1, r, k=16):
    # Integrate a power spectrum with direct numerical integration (between k0 and k1)
    # to calculate a correlation function at r.

    # Get the grid of k we will integrate over
    ka = np.logspace(log_k0, log_k1, (1 << k) + 1)[np.newaxis, :]
    ra = r[:, np.newaxis]

    dlk = np.log(ka[0, 1] / ka[0, 0])

    integrand = psfunc(ka) * ka ** 3 / (2 * np.pi ** 2) * np.sinc(ka * ra / np.pi)

    return si.romb(integrand) * dlk


def ps_to_corr(
    psfunc: Callable[[np.ndarray], np.ndarray],
    minlogr: float = -1,
    maxlogr: float = 5,
    switchlogr: float = 2,
    samples: bool = False,
    log_threshold: float = 1,
    samples_per_decade: int = 100,
    h: float = 1e-5,
) -> Union[Callable[[np.ndarray], np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Transform a 3D power spectrum into a correlation function.

    Parameters
    ----------
    psfunc
        The power spectrum function to integrate.
    minlogr, maxlogr
        The minimum and maximum values of r to calculate (given as log_10(r h /
        Mpc)).
    switchlogr
        The scale below which to directly integrate. Above this the `hankel` package
        integration is used.
    samples
        Return the samples instead of the interpolation function. Mostly useful for
        debugging.
    log_threshold
        Below this level in the correlation function use a linear interpolation. This
        should be set to be around the level where the correlation function starts to
        switch sign.
    samples_per_decade
        How many samples per decade to calculate.
    h
        Numerical accuracy parameter for the `hankel` modules transform.

    Returns
    -------
    [type]
        [description]
    """

    # Create a transform class suitable for integration of density and potential power
    # spectra. They key here is choosing appropriate values for N and h, and these have
    # been tested against other exact integration approach
    ft = hankel.SymmetricFourierTransform(ndim=3, N=(np.pi / h), h=h)

    # Create a grid with 100 points per decade
    rlow = np.logspace(
        minlogr,
        switchlogr,
        int((switchlogr - minlogr) * samples_per_decade),
        endpoint=False,
    )
    rhigh = np.logspace(
        switchlogr, maxlogr, int((maxlogr - switchlogr) * samples_per_decade) + 1
    )

    # Perform the transform
    Fhigh = ft.transform(psfunc, rhigh, ret_err=False) / (2 * np.pi) ** 3

    # Explicitly calculate and insert the zero lag correlation
    # TODO: remove the hardcoded limits
    rlow = np.insert(rlow, 0, 0.0)
    Flow = _corr_direct(psfunc, -5, 3, rlow)

    ra = np.concatenate([rlow, rhigh])
    Fr = np.concatenate([Flow, Fhigh])

    if samples:
        return ra, Fr

    return sinh_interpolate(ra, Fr, x_t=ra[1], f_t=log_threshold)


def legendre_array(lmax: int, mu: np.ndarray) -> np.ndarray:
    """Calculate the Legendre polynomials up to lmax at give mu.

    Parameters
    ----------
    lmax
        The maximum l to calculate.
    mu
        The values of mu (=cos(theta)) to calculate at.

    Returns
    -------
    P_l
        A 2D array of the polynomials for each l and mu.
    """
    lm = np.zeros((lmax + 1, len(mu)), dtype=np.float64)

    for i, v in enumerate(mu):
        lm[:, i] = ss.lpn(lmax, v)[0]

    return lm


def corr_to_clarray(
    corr: Callable[[np.ndarray], np.ndarray],
    lmax: int,
    xarray: np.ndarray,
    xromb: int = 3,
    xwidth: Optional[float] = None,
    q: int = 2,
):
    """Calculate an array of :math:`C_l(\chi_1, \chi_2)`.

    Parameters
    ----------
    corr
        The real space correlation function.
    c
        The cosmology object.
    lmax
        Maximum l to calculate up to.
    xarray
        Array of comoving distances to calculate at.
    xromb
        The Romberg order for integrating over radial bins. This generates an
        exponentially increasing amount of work, so increase carefully.
    xwidth
        Width of radial bin to integrate over. If None (default),
        calculate from the separation of the first two bins.
    q
        Integration accuracy parameter for the Legendre transform

    Returns
    -------
    clxx
        Array of the :math:`C_l(\chi_1, \chi_2)` values, with l as the first axis.
    """

    # The integration over angle will be performed by Gauss-Legendre integration. Here
    # we calculate the points that it will be evaluated at....
    M = q * lmax
    mu, w, wsum = ss.roots_legendre(M, mu=True)

    xlen = xarray.size
    corr_array = np.zeros((M, xlen, xlen))

    # If xromb > 0 we need to integrate over the radial bin width, start by modifying
    # the array of distances to add extra points over which we'll integrate
    if xromb > 0:
        # Calculate the half bin width
        # TODO: make this more accurate for non-uniform bin spacings
        xsort = np.sort(xarray)
        xhalf = np.abs(xsort[1] - xsort[0]) / 2.0 if xwidth is None else xwidth / 2.0
        xint = 2 ** xromb + 1
        xspace = 2.0 * xhalf / 2 ** xromb

        # Calculate the extended z-array with the extra intervals to integrate over
        xa = (
            xarray[:, np.newaxis] + np.linspace(-xhalf, xhalf, xint)[np.newaxis, :]
        ).flatten()
    else:
        xa = xarray

    x1 = xa[np.newaxis, :, np.newaxis]
    x2 = xa[np.newaxis, np.newaxis, :]

    # Split the set of theta values to fill into groups of length ~50 and process each,
    # this is helps reduce memory usage which otherwise we be massively inflated by the
    # extra points we integrate over
    for msec in np.array_split(np.arange(M), M // 50):

        ms = mu[msec][:, np.newaxis, np.newaxis]

        # This a the cosine rule but rewritten in a numerically stable form
        rc = ((x1 - x2) ** 2 + 2 * x1 * x2 * (1 - ms)) ** 0.5
        corr1 = corr(rc)

        # If zromb then we need to integrate over the redshift bins which we do by
        # fixed order romberg
        if xromb > 0:
            corr1 = corr1.reshape(-1, xlen, xint, xlen, xint)
            corr1 = si.romb(corr1, dx=xspace, axis=4)
            corr1 = si.romb(corr1, dx=xspace, axis=2)
            corr1 /= (2 * xhalf) ** 2  # Normalise

        corr_array[msec, :, :] = corr1

    lm = legendre_array(lmax, mu)
    lm *= w * 4.0 * np.pi / wsum

    clxx = np.dot(lm, corr_array.reshape(M, -1))
    clxx = clxx.reshape(lmax + 1, xlen, xlen)

    return clxx


def ps_to_aps_flat(
    psfunc: Callable[[np.ndarray], np.ndarray],
    n_k: int = 0,
    n_mu: int = 0,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Calculate a multi-distance angular power spectrum from a 3D power spectrum.

    This takes a flat sky limit.

    Parameters
    ----------
    psfunc
        The power spectrum function.
    c
        The cosmology object.
    n_k
        Multiply the power spectrum by extra powers of k.
    n_mu
        Multiply by powers of mu, the angle of the Fourier mode to the line of sight.

    Returns
    -------
    aps
        A function which can calculate the power spectrum :math:`C_l(\chi_1, \chi_2)`
    """
    kperpmin = 1e-4
    kperpmax = 40.0
    nkperp = 500
    kparmax = 20.0
    nkpar = 32768

    kperp = np.logspace(np.log10(kperpmin), np.log10(kperpmax), nkperp)[:, np.newaxis]
    kpar = np.linspace(0, kparmax, nkpar)[np.newaxis, :]

    k = (kpar ** 2 + kperp ** 2) ** 0.5
    mu = kpar / k

    # Calculate the power spectrum on a 2D grid and FFT the line of sight axis to turn
    # it into something like a separation
    dd = psfunc(k) * k ** n_k * mu ** n_mu
    aps_dd = dct(dd, type=1) * kparmax / (2 * nkpar)

    def _interp2d(arr, x, y):
        # Interpolate a 2D array, broadcasting as needed
        x, y = np.broadcast_arrays(x, y)
        sh = x.shape

        x, y = x.flatten(), y.flatten()
        v = np.zeros_like(x)
        bilinearmap.interp(arr, x, y, v)

        return v.reshape(sh)

    def _aps(la, xa1, xa2):
        # A closure that will calculate the angular power spectrum
        xc = 0.5 * (xa1 + xa2)
        rpar = np.abs(xa2 - xa1)

        # Bump anything that is zero upwards to avoid a log zero warning.
        la = np.where(la == 0.0, 1e-10, la)

        x = (
            (np.log10(la) - np.log10(xc * kperpmin))
            / np.log10(kperpmax / kperpmin)
            * (nkperp - 1)
        )
        y = rpar / (np.pi / kparmax)
        clzz = _interp2d(aps_dd, x, y) / (xc ** 2 * np.pi)

        return clzz

    return _aps
