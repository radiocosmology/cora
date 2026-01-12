from typing import Callable, Union, Tuple, List

import numpy as np
import scipy.integrate as si
import scipy.special as ss
import scipy.signal as ssig
from scipy.fftpack import dct

from caput import mpiarray

import hankl
import hankel
import pyfftlog

from ..util import bilinearmap, coord
from ..util.nputil import FloatArrayLike

from .lssutil import (
    diff2,
    calculate_width,
    integrate_uniform_into_bins,
    exponential_FoG_kernel_1d,
)


def richardson(
    estimates: List[FloatArrayLike],
    t: float,
    base_pow: int = 1,
    return_table: bool = False,
) -> Union[FloatArrayLike, List[List[FloatArrayLike]]]:
    """Use Richardson extrapolation to improve the accuracy of a sequence of estimates.

    Parameters
    ----------
    estimates
        A list of the estimates. Each successive entry should have an accuracy parameter
        `h` (i.e. step size) which decreases by a factor `t`.
    t
        The rate at which the step size decreases, e.g. t = 2 corresponds to halving the
        step size for each successive entry.
    base_pow
        The powers in the error series. By default this is 1 which will cancel all error
        terms up to and including `k - 1` (where `k` is the length of `estimates`. If
        the series only includes even error terms, this can be set to 2, which will
        cancel up to `2 (k - 1)`.
    return_table
        If True return the full Richardson table, if False just return the highest
        accuracy entry.

    Returns
    -------
    ret
        The highest accuracy estimate after extrapolation if `table` is False, otherwise
        the full triangular extrapolation table.
    """

    k = len(estimates)

    table = []

    for row_ind in range(k):

        newrow = [estimates[row_ind]]

        # Cancel off each successive power at this step size using previous estimates
        for col_ind in range(1, row_ind + 1):

            n = col_ind * base_pow
            r = (t**n * newrow[col_ind - 1] - table[row_ind - 1][col_ind - 1]) / (
                t**n - 1.0
            )

            newrow.append(r)

        table.append(newrow)

    return table if return_table else table[k - 1][k - 1]


def _corr_direct(psfunc, log_k0, log_k1, r, k=16):
    # Integrate a power spectrum with direct numerical integration (between k0 and k1)
    # to calculate a correlation function at r.

    # Get the grid of k we will integrate over
    ka = np.logspace(log_k0, log_k1, (1 << k) + 1)[np.newaxis, :]
    ra = r[:, np.newaxis]

    dlk = np.log(ka[0, 1] / ka[0, 0])

    integrand = psfunc(ka) * ka**3 / (2 * np.pi**2) * np.sinc(ka * ra / np.pi)

    return si.romb(integrand) * dlk


def _corr_fftlog(
    func,
    logrmin,
    logrmax,
    samples_per_decade,
    upsample=1000,
    pad_low=2,
    pad_high=1,
    q_bias=0.0,
):
    # Evaluate a correlation function from a power spectrum using the pyfftlog library.
    # The default values here should mostly work for density and potential power spectra

    # Pad the limits
    rlow = logrmin - pad_low
    rhigh = logrmax + pad_high
    n = samples_per_decade * upsample * (rhigh - rlow)

    dlnr = (rhigh - rlow) * np.log(10.0) / n
    rc = 10 ** ((rhigh + rlow) / 2.0)

    # Initialise and calculate the optimal kr for this n
    kr, xsave = pyfftlog.fhti(n, 0.5, dlnr, q=q_bias, kr=1.0, kropt=1)

    # Get the actual sample locations
    ja = np.arange(n)
    jc = (n + 1) / 2.0
    kc = kr / rc
    ra = rc * np.exp((ja - jc) * dlnr)
    ka = kc * np.exp((ja - jc) * dlnr)
    del ja

    # Perform the transform
    f = func(ka) * ka
    del ka
    ft = pyfftlog.fftl(f, xsave, rk=(kc / rc), tdir=1)

    # Decimate and trim to get to the requested sample density
    rd = ra[::upsample]
    fd = ft[::upsample]
    del ra, ft
    mask = (np.log10(rd) >= logrmin) & (np.log10(rd) <= logrmax)
    rd = rd[mask]
    fd = fd[mask]

    # Apply the final scaling and return
    return rd, fd / rd / (8 * np.pi**3) ** 0.5


def _corr_hankel(func, logrmin, logrmax, samples_per_decade, h=1e-5):
    # Perform a hankel integration of density and potential power
    # spectra to get a correlation function. They key here is choosing appropriate values for N and h, and these have
    # been tested against other exact integration approach
    ft = hankel.SymmetricFourierTransform(ndim=3, N=(np.pi / h), h=h)

    r = np.logspace(logrmin, logrmax, int((logrmax - logrmin) * samples_per_decade) + 1)

    # Perform the transform
    F = ft.transform(func, r, ret_err=False) / (2 * np.pi) ** 3

    return r, F


def _corr_hankl_richardson(
    func, logrmin, logrmax, samples_per_decade, richardson_n=6, pad_low=2, pad_high=1
):
    # Evaluate a correlation function from a power spectrum using the hankl library with
    # Richardson extrapolation.
    # The default values here should mostly work for density and potential power spectra

    # Pad the limits
    rlow = logrmin - pad_low
    rhigh = logrmax + pad_high
    n = int(samples_per_decade * (rhigh - rlow))

    # Calculate the correlation function upsampling by a factor 2**ii
    def _work(ii):

        u = 2**ii
        k = np.logspace(-rhigh, -rlow, n * u, endpoint=False)

        r, xi = hankl.P2xi(k, func(k), 0, lowring=False)

        rt = r[(u - 1) :: u]
        xit = xi.real[(u - 1) :: u]

        return rt, xit

    rs, estimates = zip(*[_work(ii) for ii in range(richardson_n)])

    # Double check that the samples have all ended up at the same r points
    for r in rs[1:]:
        assert np.allclose(r, rs[0])

    # Trim to the requested set of r values
    mask = (np.log10(rs[0]) >= logrmin) & (np.log10(rs[0]) <= logrmax)
    r = rs[0][mask]
    estimates = [e[mask] for e in estimates]

    # Perform and return the Richardson extrapolation
    return r, richardson(estimates, 2.0)


def ps_to_corr(
    psfunc: Callable[[np.ndarray], np.ndarray],
    minlogr: float = -1,
    maxlogr: float = 5,
    switchlogr: float = 2,
    samples_per_decade: int = 100,
    fftlog: bool = True,
    minlogk: float = -5,
    maxlogk: float = 3,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a 3D power spectrum into a correlation function.

    Parameters
    ----------
    psfunc
        The power spectrum function. Takes a single (vector) argument of the wavenumber k.
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
    fftlog
        If set, use the FFTlog method with Richardson extrapolation for the large-r
        calculations. If False use the hankel package.
    minlogk, maxlogk
        Limits for the direct integration of the power spectrum for small r. The default
        parameters are suitable for appropriately regularised matter and potential power
        spectra.
    kwargs
        If using the hankel package there is one parameter available: `h`. If using
        fftlog there are four parameters: `upsample`, `pad_low`, `pad_high`, and
        `q_bias`.

    Returns
    -------
    r, corr
        The correlation function samples and the locations they are calculated at.
    """

    # Create a grid with 100 points per decade
    rlow = np.logspace(
        minlogr,
        switchlogr,
        int((switchlogr - minlogr) * samples_per_decade),
        endpoint=False,
    )

    # Perform the high-r transform
    if fftlog:
        rhigh, Fhigh = _corr_hankl_richardson(
            psfunc, switchlogr, maxlogr, samples_per_decade, **kwargs
        )
    else:
        rhigh, Fhigh = _corr_hankel(
            psfunc, switchlogr, maxlogr, samples_per_decade, **kwargs
        )

    # Explicitly calculate and insert the zero lag correlation
    rlow = np.insert(rlow, 0, 0.0)
    Flow = _corr_direct(psfunc, minlogk, maxlogk, rlow)

    ra = np.concatenate([rlow, rhigh])
    Fr = np.concatenate([Flow, Fhigh])

    return ra, Fr


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
    xwidth: float = None,
    q: int = 2,
    chunksize: int = 50,
    channel_method: str = "gauss-legendre",
    chi1_2nd_derivative: bool = False,
    chi2_2nd_derivative: bool = False,
    FoG_convolve: bool = False,
    FoG_sigmaP: float = None,
    FoG_kernel_max_nchannels: int = None,
    channel_profile: Callable[[np.ndarray], np.ndarray] = None,
):
    r"""Calculate an array of :math:`C_\ell(\chi_1, \chi_2)`.

    This routine computes the following triple integral:

    .. math::
        C_\ell(\chi_1, \chi_2) = 2\pi \int_{-1}^1 P_\ell(\mu)
        \int d\chi_1' W_1(\chi_1') \int d\chi_2' W_2(\chi_2')
        \xi(r[\mu, \chi_1', \chi_2'])

    where :math:`W_1` and :math:`W_2` are normalized channel profiles centered on
    :math:`\chi_1` and :math:`\chi_2`.

    Parameters
    ----------
    corr
        The real space correlation function.
    lmax
        Maximum ell to calculate up to.
    xarray
        Array of comoving distances to calculate at.
    xromb
        The order for integrating within radial bins, related to the number of samples
        within each bin by `2**xromb + 1`. This generates an exponentially increasing
        amount of work, so increase carefully. Note that despite the parameter name
        this no longer uses a Romberg integrator, but either a Gauss-Legendre
        quadrature rule or a combination of Simpson's rule and the trapezoid rule,
        depending on `channel_method`.
    xwidth
        Assume each radial bin has this width when integrating. If None (default),
        use nonuniform bin width determined from `xarray`.
    q
        Integration accuracy parameter for the integral over mu. The number of mu
        samples is equal to `q * lmax`.
    chunksize
        Chunk size for evaluating discrete samples of angular integrand in batches.
        Default: 50.
    channel_method
        Method for integrating within radial bins: Gauss-Legendre quadrature
        ("gauss-legendre") or a combination of Simpson's rule and the trapezoid rule
        ("uniform") based on uniform sampling of the entire comoving-distance
        range being considered. (The latter is needed for Finger-of-God damping
        to be incorporated.)
    chi1_2nd_derivative, chi2_2nd_derivative
        Whether to take finite-difference 2nd derivatives of the integrand in chi_1
        or chi_2 prior to integrating. This is useful for computing angular power
        spectra involving the Kaiser redshift-space distortion term by
        differentiating power spectra of the gravitational potential.
    FoG_convolve
        Whether to convolve the integrand with the Finger-of-God damping kernel
        prior to integration. Must choose `channel_method = "uniform"`.
    FoG_sigmaP
        Finger-of-God damping scale to use in kernel.
    FoG_kernel_max_nchannels
        Restrict the width of the Finger-of-God kernel to the comoving distance
        interval equal to twice this number of frequency channels. This reduces
        the computational cost of the convolution.
    channel_profile
        Frequency channel profile to use in the channel integration.
        The profile function must be defined on [-0.5, 0.5].
        If None, a top-hat profile is used. Default: None.

    Returns
    -------
    clxx
        Array of the :math:`C_\ell(\chi_1, \chi_2)` values, with l as the first axis.

    Notes
    -----
    The integration scheme over radial bins evaluates a point at :math:`\xi(r=0)`,
    which can have an outsized influence for very blue input power spectra, because
    the number of samples in the integration is quite small and the correlation
    function can diverge quite strongly as :math:`r \to 0`. While this could
    make a difference in the point variance, in practice this doesn't seem to make
    much difference to the output maps, as the transform from :math:`\theta \to \ell`
    seems to wash out this impact.
    """

    if channel_method not in ["gauss-legendre", "uniform"]:
        raise RuntimeError("channel method must be one of 'gauss-legendre', 'uniform'")

    # Catch combinations of parameters that are not implemented
    if channel_method == "gauss-legendre":
        if FoG_convolve:
            raise NotImplementedError(
                "Finger-of-God damping is not implemented for Gauss-Legendre "
                "channel integration"
            )
        if channel_profile is not None:
            raise NotImplementedError(
                "Integration over a nontrivial channel profile is not implemented "
                "for Gauss-Legendre channel integration"
            )

    elif channel_method == "uniform":
        if xromb == 0:
            raise NotImplementedError(
                "Need xromb > 0 if using Simpson-trapezoid channel integration scheme"
            )
        if xwidth is not None:
            raise NotImplementedError(
                "Uniform channel width not implemented for Simpson-trapezoid "
                "channel integration scheme"
            )

    # The integration over mu will be performed by Gauss-Legendre quadrature.
    # Here we calculate the points that it will be evaluated at.
    M = q * lmax
    mu, w, wsum = ss.roots_legendre(M, mu=True)

    # Some of the logic below depends on xarray being in increasing order,
    # so if it's not, we operate on a reversed version and then reverse
    # the final results at the end
    sorted_xarray = np.sort(xarray)
    if np.array_equal(xarray, sorted_xarray):
        flipped_xarray = False
    elif np.array_equal(xarray[::-1], sorted_xarray):
        flipped_xarray = True
    else:
        raise RuntimeError("xarray must be strictly increasing or decreasing")

    # Make MPIArray to store channel-integrated correlation function
    xlen = xarray.size
    corr_array = mpiarray.zeros((M, xlen, xlen), axis=0)
    clo = corr_array.local_offset[0]
    _len = corr_array.local_array.shape[0]

    # Determine full set of chi values for integrand
    if channel_method == "gauss-legendre":

        if xromb > 0:
            # Calculate the half bin width
            if xwidth is None:
                xhalf = 0.5 * calculate_width(sorted_xarray)
            else:
                xhalf = np.ones(shape=xarray.shape) * xwidth / 2.0

            xint = 2**xromb + 1

            # Get the quadrature points and weights
            x_r, x_w, x_wsum = ss.roots_legendre(xint, mu=True)
            x_w /= x_wsum

            # Calculate the extended chi array, with the extra intervals
            # to integrate over
            xarray_full = (
                sorted_xarray[:, np.newaxis] + xhalf[:, np.newaxis] * x_r
            ).flatten()

        else:
            xarray_full = sorted_xarray

    else:

        # Determine half-channel widths and channel boundaries in chi
        xhalf = 0.5 * calculate_width(sorted_xarray)
        xedges = np.concatenate(
            [sorted_xarray - xhalf, [sorted_xarray[-1] + xhalf[-1]]]
        )

        # Generate uniformly-spaced array of chi values over full chi range
        xint = 2**xromb + 1
        xarray_full = np.linspace(
            xedges[0], xedges[-1], len(xarray) * xint, endpoint=True
        )

        if channel_profile is not None:
            # Determine the channel index of each point in xarray_full
            chan_idx = np.searchsorted(xedges, xarray_full, side="right") - 1
            chan_idx = np.clip(chan_idx, 0, xlen - 1)

            # Compute the distance of each point from the channel center,
            # divided by the channel width
            rel_dx_full = (xarray_full - sorted_xarray[chan_idx]) / (
                2 * xhalf[chan_idx]
            )

            # Evaluate the channel profile at these values
            profile_full = channel_profile(rel_dx_full)

            # Integrate channel profile within each channel.
            # (We'll use these values to normalize our final channel
            # integrals.)
            norm = 2 * xhalf * si.quad(channel_profile, -0.5, 0.5)[0]
        else:
            profile_full = None

            # For top-hat profile, normalization factors are just the
            # channel widths
            norm = 2 * xhalf

        # If convolving with Finger-of-God kernel, precompute kernel
        if FoG_convolve:

            # Compute the maximum number of |dchi| values at which to
            # evaluate the kernel
            if FoG_kernel_max_nchannels is None:
                # Just use half of the full xarray
                n_dx = len(xarray_full) // 2
            else:
                # Compute the maximum number of xarray_full elements
                # corresponding to the number of padding channels,
                # considering the higher and lower sets of padding
                # channels
                n_dx = max(
                    np.sum(xarray_full < xedges[FoG_kernel_max_nchannels + 1]),
                    np.sum(xarray_full > xedges[-FoG_kernel_max_nchannels]),
                )

            # Make array of dchi for evaluating kernel
            dxarray = xarray_full[:n_dx] - xarray_full[0]
            dxarray = np.concatenate([-dxarray[1:][::-1], dxarray])

            # Evaluate kernel, and downselect to nonzero elements.
            # (Some elements may be zero due to small-value truncation
            # within exponential_FoG_kernel_1d.)
            FoG_kernel = exponential_FoG_kernel_1d(dxarray, FoG_sigmaP)
            FoG_kernel = FoG_kernel[FoG_kernel > 0.0]

            # Compute sum of kernel values that will be used in convolution
            # at every element of xarray_full. This will be used to normalize
            # the kernel later.
            FoG_kernel_norm = ssig.oaconvolve(
                np.ones_like(xarray_full),
                FoG_kernel,
                mode="same",
            )

    # Split mu values into chunks, to avoid memory usage blowing up
    for msec in np.array_split(np.arange(_len), _len // chunksize):
        # Index into the global index in mu, and evaluate the correlation
        # function at the relevant (mu,chi_1,chi_2) values
        rc = coord.cosine_rule(mu[clo + msec], xarray_full, xarray_full)
        corr1 = corr(rc)

        # If desired, compute second derivatives in chi_1 and/or chi_2.
        # (Numerical stability is better if this is done before the FoG
        # convolution rather than after.)
        if chi1_2nd_derivative:
            corr1 = diff2(corr1, xarray_full, axis=1)
        if chi2_2nd_derivative:
            corr1 = diff2(corr1, xarray_full, axis=2)

        # Integrate over each channel
        if channel_method == "gauss-legendre":
            # If xromb>0, we integrate using Gauss-Legendre quadrature,
            # implemented as a matrix multiply
            if xromb > 0:
                corr1 = corr1.reshape(-1, xint)
                corr1 = np.matmul(corr1, x_w).reshape(-1, xlen, xint, xlen)
                corr1 = np.matmul(corr1.transpose(0, 1, 3, 2), x_w)
        else:
            # If desired, convolve with Finger-of-God kernel in chi_1 and
            # chi_2. We normalize the kernel such that it integrates to
            # unity, by dividing by the sums computed earlier.
            if FoG_convolve:
                corr1 = (
                    ssig.oaconvolve(
                        corr1,
                        FoG_kernel[np.newaxis, :, np.newaxis],
                        axes=(1,),
                        mode="same",
                    )
                    / FoG_kernel_norm[np.newaxis, :, np.newaxis]
                )
                corr1 = (
                    ssig.oaconvolve(
                        corr1,
                        FoG_kernel[np.newaxis, np.newaxis, :],
                        axes=(2,),
                        mode="same",
                    )
                    / FoG_kernel_norm[..., :]
                )

            # Perform channel integrals in chi_1
            corr1 = integrate_uniform_into_bins(
                corr1, xarray_full, xedges, axis=1, norm=norm, window=profile_full
            )

            # Perform channel integrals in chi_2
            corr1 = integrate_uniform_into_bins(
                corr1, xarray_full, xedges, axis=2, norm=norm, window=profile_full
            )

        # Save results
        corr_array.local_array[msec] = corr1

    # Perform the dot product split over ranks for
    # memory and time
    lm = legendre_array(lmax, mu)
    lm *= w[np.newaxis] * 4.0 * np.pi / wsum

    # Reshape and properly distribute the array to
    # perform the dot product
    corr_array = corr_array.reshape(None, -1).redistribute(axis=1)

    # Carry out Gauss-Legendre integration over mu, via matrix multiplication
    clxx = np.dot(lm, corr_array.local_array)
    clxx = mpiarray.MPIArray.wrap(clxx, axis=1).redistribute(axis=0)
    clxx = clxx.reshape(None, xlen, xlen)

    # If we reversed the input order of xarray, reverse the x axes
    # in the final output
    if flipped_xarray:
        clxx = clxx[:, ::-1, ::-1]

    return clxx


def ps_to_aps_flat(
    psfunc: Callable[[np.ndarray], np.ndarray],
    n_k: int = 0,
    n_mu: int = 0,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Calculate a multi-distance angular power spectrum from a 3D power spectrum.

    This uses a flat sky limit. See equation 21 of arXiv:astro-ph/0605546.

    Parameters
    ----------
    psfunc
        The power spectrum function. Takes a single (vector) argument of the wavenumber k.
    c
        The cosmology object.
    n_k
        Multiply the power spectrum by extra powers of k.
    n_mu
        Multiply by powers of mu, the angle of the Fourier mode to the line of sight.

    Returns
    -------
    aps
        A function which can calculate the power spectrum :math:`C_l(\chi_1, \chi_2)`.
        Takes three broadcastable arguments `aps(l, chi1, chi2)` for the multipole `l`,
        and the two comoving distances `chi1` and `chi2`.
    """
    kperpmin = 1e-4
    kperpmax = 40.0
    nkperp = 500
    kparmax = 20.0
    nkpar = 32768

    kperp = np.logspace(np.log10(kperpmin), np.log10(kperpmax), nkperp)[:, np.newaxis]
    kpar = np.linspace(0, kparmax, nkpar)[np.newaxis, :]

    k = (kpar**2 + kperp**2) ** 0.5
    mu = kpar / k

    # Calculate the power spectrum on a 2D grid and FFT the line of sight axis to turn
    # it into something like a separation
    dd = psfunc(k) * k**n_k * mu**n_mu
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
        clzz = _interp2d(aps_dd, x, y) / (xc**2 * np.pi)

        return clzz

    return _aps
