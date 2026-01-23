from typing import Callable, Union, Tuple, Optional

import numpy as np
from scipy.fftpack import dct
import scipy.integrate as si
import scipy.sparse as ssparse

import healpy

from caput import config
from draco.util import tools

from ..util import cubicspline as cs
from ..util import hputil
from ..util.nputil import FloatArrayLike
from ..util import bilinearmap
from ..util import cosmology as cora_cosmology


def linspace(x: Union[dict, list, np.ndarray]) -> np.ndarray:
    """A config parser that generates a linearly spaced set of values.

    This feeds the parameters directly into numpy linspace.

    Parameters
    ----------
    x
        If a dict, this requires `start`, `stop`, and `num` keys, while there is an
        optional `endpoint` key. If a list the elements are interpreted as `[start,
        stop, num, endpoint]` where `endpoint` is again optional. If an array, it is
        just returned as is (this is useful for configuring a pipeline in Python rather
        than YAML).

    Returns
    -------
    linspace
        A linearly spaced numpy array.
    """
    if not isinstance(x, (dict, list, np.ndarray)):
        raise config.CaputConfigError(
            f"Require a dict, list or array type. Got a {type(x)}."
        )

    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, dict):
        start = x["start"]
        stop = x["stop"]
        num = x["num"]
        endpoint = x.get("endpoint", True)
    else:
        start = x[0]
        stop = x[1]
        num = x[2]
        endpoint = x[3] if len(x) == 4 else True

    return np.linspace(start, stop, num, endpoint=endpoint)


def sinh_interpolate(
    x: FloatArrayLike, f: FloatArrayLike, x_t: float = 1, f_t: float = 1
) -> Callable[[FloatArrayLike], FloatArrayLike]:
    """Produce an 1D interpolation function using a sinh scaling.

    The interpolation is done within the space of `arcsinh(x / x_t)` and `arcsinh(f /
    f_t)`, this gives an effective log interpolation for absolute values greater than
    the threshold and linear when less than the threshold. This allows an effective
    log interpolation which will handle zero and negative values.

    Parameters
    ----------
    x
        The positions the function is defined it.
    f
        The values of the function.
    x_t
        The sinc threshold value for the positions.
    f_t
        The function value threshold.

    Returns
    -------
    f_int
        The interpolation function.
    """

    # Scale and take arcsinh
    asf = np.arcsinh(f / f_t)
    asx = np.arcsinh(x / x_t)

    # Create cubic spline interpolater
    fs = cs.Interpolater(asx, asf)

    def _thinwrap(x):
        return fs(x)

    def _f_asinh(x_):
        sx = np.arcsinh(x_ / x_t)
        fv = _thinwrap(sx)
        return f_t * np.sinh(fv)

    return _f_asinh


def diff2(f: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Take a non-uniform second order derivative.

    Parameters
    ----------
    f
        The function to take the derivative of.
    x
        The locations of the samples along the first axis.
    axis
        The axis to take the second derivative along.

    Returns
    -------
    d2fdx2
        The second derivative along the axis. Same shape as f, but the first two and
        elements and the final element will be zero.

    Notes
    -----
    Main scheme from https://doi.org/10.1016/0307-904X(94)00020-7,
    one-sided schemes from https://en.wikipedia.org/wiki/Finite_difference_coefficient#Arbitrary_stencil_points
    """

    d2 = np.zeros_like(f)

    # Resolve axis
    axis = axis % f.ndim

    # Quick function to generate slices along the right axis
    def sl(v):
        return (slice(None),) * axis + (v,)

    # Compute elements from i=2 to N-2
    for i in range(2, f.shape[axis] - 1):

        dm2 = x[i] - x[i - 2]
        dm1 = x[i] - x[i - 1]
        dp1 = x[i + 1] - x[i]

        alpha = 2 * (dp1 - dm1) / (dm2 * (dm2 + dp1) * (dm2 - dm1))
        beta = 2 * (dm2 - dp1) / (dm1 * (dm2 - dm1) * (dm1 + dp1))
        gamma = 2 * (dm2 + dm1) / (dp1 * (dm1 + dp1) * (dm2 + dp1))

        d2[sl(i)] = alpha * f[sl(i - 2)]
        d2[sl(i)] += beta * f[sl(i - 1)]
        d2[sl(i)] -= (alpha + beta + gamma) * f[sl(i)]
        d2[sl(i)] += gamma * f[sl(i + 1)]

    # Compute element i=0 with one-sided 4-point 2nd derivative
    dp1 = x[1] - x[0]
    dp2 = x[2] - x[0]
    dp3 = x[3] - x[0]

    alpha = 2 * (dp1 + dp2 + dp3) / (dp1 * dp2 * dp3)
    beta = -2 * (dp2 + dp3) / (dp1 * (dp1 - dp2) * (dp1 - dp3))
    gamma = 2 * (dp1 + dp3) / ((dp1 - dp2) * dp2 * (dp2 - dp3))
    delta = 2 * (dp1 + dp2) / ((dp1 - dp3) * dp3 * (-dp2 + dp3))

    d2[sl(0)] = alpha * f[sl(0)] + beta * f[sl(1)] + gamma * f[sl(2)] + delta * f[sl(3)]

    # Compute element i=1 with 4-point 2nd derivative
    dm1 = x[1] - x[0]
    dp1 = x[2] - x[1]
    dp2 = x[3] - x[1]

    alpha = 2 * (dp1 + dp2) / (dm1 * (dm1 + dp1) * (dm1 + dp2))
    beta = 2 * (dm1 - dp1 - dp2) / (dm1 * dp1 * dp2)
    gamma = 2 * (dm1 - dp2) / (dp1 * (dm1 + dp1) * (dp1 - dp2))
    delta = -2 * (dm1 - dp1) / ((dp1 - dp2) * dp2 * (dm1 + dp2))

    d2[sl(1)] = alpha * f[sl(0)] + beta * f[sl(1)] + gamma * f[sl(2)] + delta * f[sl(3)]

    # Compute element i=N-1 with one-sided 4-point 2nd derivative
    dm1 = x[-1] - x[-2]
    dm2 = x[-1] - x[-3]
    dm3 = x[-1] - x[-4]

    alpha = 2 * (dm1 + dm2) / ((dm1 - dm3) * dm3 * (-dm2 + dm3))
    beta = 2 * (dm1 + dm3) / ((dm1 - dm2) * dm2 * (dm2 - dm3))
    gamma = -2 * (dm2 + dm3) / (dm1 * (dm1 - dm2) * (dm1 - dm3))
    delta = 2 * (dm1 + dm2 + dm3) / (dm1 * dm2 * dm3)

    d2[sl(-1)] = (
        alpha * f[sl(-4)] + beta * f[sl(-3)] + gamma * f[sl(-2)] + delta * f[sl(-1)]
    )

    return d2


def laplacian(maps: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Take the Laplacian of a set of Healpix maps.

    Parameters
    ----------
    maps
        An 2D array of maps. First axis is the x coord, second is Healpix pixels.
    x
        The coordinate along the first axis. This axis need not be uniform.

    Returns
    -------
    l2_maps
        The maps with the Laplacian operator applied. Due to the finite difference
        scheme along the first axis, the edge entries will be zero.
    """

    nside = healpy.npix2nside(maps.shape[1])
    alms = healpy.map2alm(maps, pol=False)

    lmax = healpy.Alm.getlmax(alms.shape[1])
    ell = healpy.Alm.getlm(lmax)[0]

    # Calculate the angular part of the Laplacian by multiplying by -l(l+1)...
    alms *= -ell * (ell + 1)

    # ... and transforming back
    d2 = healpy.alm2map(alms, nside=nside, pol=False)
    d2 /= x[:, np.newaxis] ** 2

    # Calculate the radial part by finite difference (should be accurate to second
    # order)
    d2 += diff2(maps, x, axis=0) + 2 * np.gradient(maps, x, axis=0) / x[:, np.newaxis]

    return d2


def gradient(maps: np.ndarray, x: np.ndarray, grad0: bool = True) -> np.ndarray:
    """Take the gradient of a set of maps.

    Parameters
    ----------
    maps
        An 2D array of maps. First axis is the x coord, second is Healpix pixels.
    x
        The coordinate along the first axis. This axis need not be uniform.
    grad0
        If True, compute the gradient along axis 0. Otherwise, output `grad[0]`
        will not be computed. This is useful if `maps` is distributed.
        Default is True.

    Returns
    -------
    grad
        The maps giving the gradient along each axis. The gradient directions are the
        first axis.
    """
    nside = healpy.npix2nside(maps.shape[1])
    nmaps = maps.shape[0]

    grad = np.zeros((3,) + maps.shape, dtype=maps.dtype)

    for i in range(nmaps):

        alm = healpy.map2alm(maps[i], pol=False, use_pixel_weights=True)

        # Get the derivatives in each angular direction
        # NOTE: the zeroth component doesn't yet contain dr
        grad[1:, i] = healpy.alm2map_der1(alm, nside)[1:] / x[i]

    if grad0:
        # TODO: try and do this in place so as not to balloon the memory
        grad[0] = np.gradient(maps, x, axis=0)

    return grad


def cutoff(x: FloatArrayLike, cut: float, sign: int, width: float, index: float):
    """A function to give a cutoff, i.e. roughly ones then a power law dropoff.

    Parameters
    ----------
    x
        The position to calculate the function at.
    cut
        The location of the cut, in log10.
    sign
        The direction of the cutoff. +1 will cut off lower values, and -1 will cut
        higher values.
    width
        The width of the cutoff transition in decades. This isn't really an exact
        measure as the transition isn't sharp.
    index
        The steepness of the cutoff. This roughly corresponds to the power law index
        of the region.

    Returns
    -------
    cutoff
        The cutoff function values.
    """
    sign = np.sign(sign)

    return (0.5 * (1 + np.tanh(sign * (np.log10(x) - cut) / width))) ** index


def pk_flat(
    maps: np.ndarray,
    chi: np.ndarray,
    maps2: Optional[np.ndarray] = None,
    lmax: Optional[int] = None,
    window: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a 2D kpar, kperp power spectrum from a set of spherical maps.

    This works within a flat sky, thin shell approximation where the angular
    transform approximates the k_perp direction, and the shell separations are
    roughly linear.

    Parameters
    ----------
    maps
        The Healpix maps at each distance. The pixel axis is the second axis.
    chi
        The distance to each shell in comoving Mpc/h.
    lmax
        Maximum l to use. This changes the max k_perp.
    maps2
        A second of set of maps. We instead calculate the cross-power spectrum.
    window
        Undo the effective window function caused by the finite map width.

    Returns
    -------
    pk
        The 2D power spectrum as k_par, k_perp.
    k_par, k_perp
        The wavenumber along each axis.
    """

    if maps2 is not None and maps.shape != maps2.shape:
        raise ValueError(
            f"Shape of maps2 ({maps2.shape}) is not compatible with maps ({maps.shape})"
        )

    chi_mean = chi.mean()
    nside = healpy.npix2nside(maps.shape[1])

    if lmax is None:
        lmax = 3 * nside

    N = len(chi)
    dx = np.ptp(chi) / (N - 1)

    # The effective L is slightly longer than the range because of the channel widths
    L = N * dx

    cn = np.fft.rfft(maps, axis=0) / N

    # Need to use this wrapper from cora (rather than any healpy functions) as we must
    # transform a complex field
    almn_square = np.array([hputil.sphtrans_complex(m, lmax) for m in cn])

    ell = np.arange(lmax + 1)
    n = np.arange(cn.shape[0])

    # Sum over m direction
    if maps2 is None:
        cln = (np.abs(almn_square) ** 2).sum(axis=-1)
    else:
        # If we need to do a cross-power we first need to calculate the almns for the
        # second field
        cn2 = np.fft.rfft(maps2, axis=0) / N
        almn_square2 = np.array([hputil.sphtrans_complex(m, lmax) for m in cn2])
        cln = (almn_square * almn_square2.conj()).sum(axis=-1).real

    # Divide by the number of m's at each l to get an average
    cln /= (2 * ell + 1)[np.newaxis, :]

    # Get the effective axes in k space
    kperp = ell / chi_mean
    kpar = 2 * np.pi * n / L

    cln *= L * chi_mean**2

    if window:
        Wk = np.sinc(kpar * dx / (2 * np.pi))
        cln /= Wk[:, np.newaxis] ** 2

    return cln, kpar, kperp


def corrfunc(
    maps: np.ndarray,
    chi: np.ndarray,
    lmax: Optional[int] = None,
    rmax: float = 1e3,
    numr: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a 1D correlation function from a set of spherical maps.

    Parameters
    ----------
    maps
        The Healpix maps at each distance. The pixel axis is the second axis.
    chi
        The distance to each shell in comoving Mpc/h.
    lmax
        Maximum l to use. This is for the intermediate calculation.
    rmax
        The maximum r to bin up to.
    numr
        The number of r bins to use.

    Returns
    -------
    cr
        The 1D correlation function.
    r
        The point the correlation function is calculated at.
    """
    from cora.signal.corrfunc import legendre_array

    clxx = healpy.anafast(maps, pol=False, lmax=lmax)

    # Calculate the pairs of displacements for each entry in the power spectrum array
    nx = len(chi)
    xxp = []
    for i in range(nx):
        for j in range(i, nx):
            xxp.append((chi[j - i], chi[j]))

    r1, r2 = np.array(xxp).T

    t = np.linspace(0, np.pi, 2048)
    mu = np.cos(t)

    Pl_arr = legendre_array(lmax, mu)
    Pl_arr *= (2 * np.arange(lmax + 1)[:, np.newaxis] + 1) / (4 * np.pi)

    cthetaxx = np.dot(clxx, Pl_arr)

    r1 = r1[:, np.newaxis]
    r2 = r2[:, np.newaxis]
    mu = mu[np.newaxis, :]
    rc = ((r1 - r2) ** 2 + 2 * r1 * r2 * (1 - mu)) ** 0.5

    rbins = np.linspace(0, rmax, numr + 1)
    rcentre = 0.5 * (rbins[1:] + rbins[:-1])

    r_ind = np.digitize(rc.ravel(), rbins)
    norm = np.bincount(r_ind, minlength=(numr + 2))
    csum = np.bincount(r_ind, weights=cthetaxx.ravel(), minlength=(numr + 2))

    cf = (csum * tools.invert_no_zero(norm))[1:-1].copy()

    return cf, rcentre


def ang_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate the angular correlation function between two maps.

    Parameters
    ----------
    x, y
        Healpix maps.

    Returns
    -------
    r_l
        The correlation between the two fields as a function of l.
    """

    cl_xx = healpy.anafast(x)
    cl_yy = healpy.anafast(y)
    cl_xy = healpy.anafast(x, y, pol=False)

    return cl_xy / (cl_xx * cl_yy) ** 0.5


def transfer(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate the angular transfer function between two maps.

    This calculates the amount of the power in X that comes from field Y.

    Parameters
    ----------
    x
        The field we are calculating the transfer function of.
    y
        The reference field.

    Returns
    -------
    T_l
        The transfer function between the two fields as a function of l.
    """

    cl_yy = healpy.anafast(y)
    cl_xy = healpy.anafast(x, y, pol=False)

    return cl_xy / cl_yy


def calculate_width(centres: np.ndarray) -> np.ndarray:
    """Estimate the width of a set of contiguous bin from their centres.

    Parameters
    ----------
    centres
        The array of the bin centre positions.

    Returns
    -------
    widths
        The estimate bin widths. Entries are always positive.
    """
    widths = np.zeros(len(centres))

    # Estimate the interior bin widths by assuming the second derivative of the bin
    # widths is small
    widths[1:-1] = (centres[2:] - centres[:-2]) / 2.0

    # Estimate the edge widths by determining where the first bin boundaries is using
    # the width of the next bin
    widths[0] = 2 * (centres[1] - (widths[1] / 2.0) - centres[0])
    widths[-1] = 2 * (centres[-1] - (widths[-2] / 2.0) - centres[-2])

    return np.abs(widths)


def integrate_uniform_into_bins(
    y: np.ndarray,
    x: np.ndarray,
    xedges: np.ndarray,
    axis: Optional[int] = -1,
    norm: Optional[np.ndarray] = None,
    window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Integrate a uniformly-sampled function into bins with arbitrary edges.

    This function uses Simpson's rule to integrate over the
    samples falling within each bin, and then uses the trapezoid
    rule to add the contributions between the lowest/highest samples
    per bin and the bin boundaries.

    The user can optionally provide an array of window function values
    that the function should be multiplied by prior to integration.
    (The hybrid Simpson-trapezoid scheme does not work if the product
    of the window and function is passed directly to this routine.)

    X values and bin edges must be strictly increasing.

    Parameters
    ----------
    y
        Function values on uniform grid.
    x
        Grid values on which function was evaluated.
    xedges
        Edges of bins to integrate within.
    axis
        Axis to integrate over.
    norm
        Array of values to divide each bin by after integration.
        If no window is used, passing `norm=np.diff(xedges)` will
        compute the mean over each bin. Default: None.
    window
        Array of window function values. Must have the same shape as `x`.

    Returns
    -------
    y_int
        Bin-integrated function.
    """

    if window is None:
        window = np.ones_like(x)

    nbins = len(xedges) - 1
    dx = x[1] - x[0]

    # Move integration axis to the end, to make slicing easier
    axis = axis % y.ndim
    y = np.moveaxis(y, axis, -1)

    # Determine array indices corresponding to x values that are
    # just above each lower bin boundary and just below each upper
    # bin boundary. (The lowest idx_lo and highest idx_hi
    # indices will correspond to lowest and highest bin boundaries.)
    idx_lo = np.searchsorted(x, xedges[:-1], side="left")
    idx_hi = np.concatenate([idx_lo[1:] - 1, [len(x) - 1]])

    # Perform cumulative Simpson integration over full x range, with
    # the integrand multiplied by the window if one is provided.
    y_int = si.cumulative_simpson(y * window, dx=dx, axis=-1, initial=0)

    # Compute differences between cumulative-integral values at indices
    # corresponding to highest and lowest x values within each bin, to obtain
    # integrals over samples that lie solely within each bin.
    y_int = y_int[..., idx_hi] - y_int[..., idx_lo]

    # The above differences will miss contributions to each bin integral
    # between the lowest/highest samples and the bin boundaries.
    # To incorporate these contributions, we first use linear interpolation
    # to estimate the y values at the bin boundaries, excluding the
    # lower boundary of the lowest bin and the upper boundary of the higest
    # bin (since they correspond to the first and last y samples).
    # This interpolation is done on the unwindowed function values, because
    # the window may vary more rapidly than the function, and this would
    # make interpolation of the windowed function unreliable.
    y_at_bounds = (
        y[..., idx_lo[1:]]
        - (y[..., idx_lo[1:]] - y[..., idx_lo[1:] - 1])
        * (x[idx_lo[1:]] - xedges[1:-1])
        / dx
    )

    # We then use these interpolated y values, along with the y values
    # at the lowest and highest samples within the bin, to integrate
    # the missing edge contributions using the trapezoid rule, and add
    # them to the results from above. If specified, the appropriate
    # window values are incorporated here.
    w_edge = window[0]
    y_int[..., 1:nbins] += (
        0.5
        * (x[idx_lo[1:]] - xedges[1:-1])
        * (w_edge * y_at_bounds + window[idx_lo[1:]] * y[..., idx_lo[1:]])
    )
    y_int[..., : nbins - 1] += (
        0.5
        * (xedges[1:-1] - x[idx_hi[:-1]])
        * (w_edge * y_at_bounds + window[idx_hi[:-1]] * y[..., idx_hi[:-1]])
    )

    # Divide by normalization factors, if specified
    if norm is not None:
        y_int /= norm

    # Move integration axis back to original position
    return np.moveaxis(y_int, -1, axis)


def integrate_uniform_into_tapered_channels(
    y: np.ndarray,
    x: np.ndarray,
    xcenters: np.ndarray,
    xwidths: np.ndarray,
    channel_profile: Callable[[np.ndarray], np.ndarray],
    axis: Optional[int] = -1,
    n_overlap: Optional[int] = 1,
    use_cached_weights: Optional[bool] = False,
):
    """Integrate a uniformly-sampled function into channels with tapered profiles.

    This function uses Simpson's rule to integrate a uniformly-sampled
    function against a set of identical profiles defined with respect
    to a list of profile centers. The profile must taper smoothly to
    a small amplitude, such that the results are not sensitive to
    the precise integration bounds as long as they enclose most of
    the profile's support. The profiles are allowed to overlap; otherwise,
    `integrate_uniform_into_bins` should be used instead.

    X values and centers must be strictly increasing.

    The integration weights can be cached and re-used in subsequent
    calls if `use_cached_weights` is set. The weights will be recomputed
    if the function is evaluated with different coordinate arguments
    or a different profile.

    Parameters
    ----------
    y
        Function values on uniform grid.
    x
        Grid values on which function was evaluated.
    xcenters
        Coordinates of channel centers. These values don't need
        to be elements of `x`.
    xwidths
        Coordinate widths of each channel.
    channel_profile
        Channel profile to use in the channel integration.
        The profile function must be defined in terms of coordinate relative
        to channel center, in units of channel width (e.g. center = 0,
        edges = [-0.5, 0.5]). The profile can extend beyond the
        channel width, but is still defined relative to this width.
    axis
        Axis to integrate over.
    n_overlap
        Number of channels on either side of the channel of interest
        to integrate over. Should be chosen to cover most of profile's
        support.
    use_cached_weights
        Store integration weights for subsequent re-use if function is
        called again with same coordinate arguments and profile.

    Returns
    -------
    y_int
        Bin-integrated function.
    """

    nx = len(x)
    dx = x[1] - x[0]
    nchan = len(xcenters)

    if not use_cached_weights or not hasattr(
        integrate_uniform_into_tapered_channels, "_weights"
    ):
        recompute_weights = True
    else:
        recompute_weights = (
            not np.isclose(integrate_uniform_into_tapered_channels._x0, x[0])
            or not np.isclose(integrate_uniform_into_tapered_channels._dx, dx)
            or integrate_uniform_into_tapered_channels._nx != nx
            or not np.allclose(
                integrate_uniform_into_tapered_channels._xcenters, xcenters
            )
            or not np.allclose(
                integrate_uniform_into_tapered_channels._xwidths, xwidths
            )
            or integrate_uniform_into_tapered_channels._channel_profile
            is not channel_profile
            or integrate_uniform_into_tapered_channels._n_overlap != n_overlap
        )

    # Generate weights for each channel integral, if not cached
    # and/or not using cache
    if recompute_weights:

        # Compute channel edges based on centers and widths
        xedges = np.concatenate(
            [xcenters - 0.5 * xwidths, [xcenters[-1] + 0.5 * xwidths[-1]]]
        )

        # Determine the channel index of each point in x
        chan_idx = np.searchsorted(xedges, x, side="right") - 1
        chan_idx = np.clip(chan_idx, 0, nx - 1)

        # Create arrays for building sparse weight array
        weight_vals = []
        weight_idx = []
        weight_idxptr = [0]

        # Loop over channels
        for c in range(nchan):
            # Find indices of elements in x that are within
            # n_overlap channels of current channel
            x_mask = (chan_idx >= c - n_overlap) & (chan_idx <= c + n_overlap)
            x_idx = np.where(x_mask)[0]

            # Ensure set of indices has odd length (so we can apply
            # Simpson's rule). If even, remove lowest index from set.
            # As long as we're probing far enough into the tail of
            # the channel profile, this should incur a small amount
            # of numerical error.
            if len(x_idx) % 2 == 0:
                x_idx = x_idx[1:]

            # Compute distance of each point from center of current
            # channel, divided by the channel width
            rel_dx_full = (x[x_idx] - xcenters[c]) / xwidths[c]

            # Evaluate the channel profile at these values
            w = channel_profile(rel_dx_full)

            # Multiply channel profile by Simpson-rule weights:
            # [1, 4, 2, 4, ..., 1] * dx / 3
            w[1:-1:2] *= 4
            w[2:-1:2] *= 2
            w *= dx / 3

            # Integrate channel profile within each channel, using
            # Simpson's rule, and use this to normalize the weights.
            # (This ensures that the final integrals are means over
            # each channel's profile.)
            norm = np.sum(w)
            w /= norm

            # Store weights and indices in format appropriate for
            # sparse (CSR) array
            weight_vals.append(w)
            weight_idx.append(x_idx)
            weight_idxptr.append(weight_idxptr[-1] + len(x_idx))

        # Concatenate weight values and index arrays
        weight_vals = np.concatenate(weight_vals)
        weight_idx = np.concatenate(weight_idx)

        # Construct sparse array of weights
        integrate_uniform_into_tapered_channels._weights = ssparse.csr_array(
            (weight_vals, weight_idx, weight_idxptr), dtype=np.float64
        )

        # Save relevant input arguments
        integrate_uniform_into_tapered_channels._x0 = x[0]
        integrate_uniform_into_tapered_channels._dx = dx
        integrate_uniform_into_tapered_channels._nx = nx
        integrate_uniform_into_tapered_channels._xcenters = xcenters
        integrate_uniform_into_tapered_channels._xwidths = xwidths
        integrate_uniform_into_tapered_channels._channel_profile = channel_profile
        integrate_uniform_into_tapered_channels._n_overlap = n_overlap

    # Move integration axis to the beginning, in preparation for multiplication
    # by sparse array
    axis = axis % y.ndim
    y = np.moveaxis(y, axis, 0)
    shp = y.shape

    # Multiply y by matrix of weights. In the result, the first axis is
    # channel number. Need to first reshape y, because only 2d x 2d matrix
    # multiplication is supported for sparse arrays.
    y_int = integrate_uniform_into_tapered_channels._weights @ y.reshape(shp[0], -1)
    y_int = y_int.reshape((y_int.shape[0],) + shp[1:])

    # Move channel axis back to original position
    return np.moveaxis(y_int, 0, axis)


def exponential_FoG_kernel_1d(
    dchi: np.ndarray,
    sigmaP: float,
    cut: Optional[float] = 1e-5,
    normalize: Optional[bool] = False,
) -> np.ndarray:
    """Compute analytical position-space Finger-of-God kernel.

    Parameters
    ----------
    dchi
        Comoving distance differences at which to evaluate kernel.
    sigmaP
        Damping scale parameter.
    cut
        Only compute kernel values that are above this fraction of the maximum
        kernel value (to avoid underflow errors). Zeros are returned for lower
        values.
    normalize
        Whether to normalize the returned kernel values by their sum.

    Returns
    -------
    kernel
        Array of kernel values evaluated at input dchi values.
    """
    a = 2**0.5 / sigmaP
    denom = a * sigmaP**2

    kernel = np.zeros_like(dchi, dtype=np.float64)

    # Make mask that selects kernel values above cut
    arg = a * np.abs(dchi)
    mask = arg <= -np.log(cut)

    # Compute relevant kernel values
    kernel[mask] = np.exp(-arg[mask]) / denom

    if normalize:
        kernel[mask] /= np.sum(kernel[mask])

    return kernel


def exponential_FoG_kernel(
    chi: np.ndarray,
    sigmaP: FloatArrayLike,
    D: FloatArrayLike,
    full_channel_kernel: bool = False,
) -> np.ndarray:
    r"""Get a smoothing kernel for approximating Fingers of God.

    Parameters
    ----------
    chi
        Radial distance of the bins.
    sigmaP
        The smooth parameter for each radial bin.
    D
        The growth factor for each radial bin.
    full_channel_kernel
        Whether to integrate the continuous kernel over each radial bin for both
        the "input" and "output" channel (True), or just the "input" channel
        (False). The former is more correct, but the latter is the default for
        legacy purposes. Default. False.

    Returns
    -------
    kernel
        A `len(chi) x len(chi)` matrix that applies the smoothing.

    Notes
    -----
    This generates an exponential smoothing matrix, which is the Fourier conjugate of a
    Lorentzian attenuation of the form :math:`(1 + k_\parallel^2 \sigma_P^2 / 2)^{-1}`
    applied to the density contrast in k-space.

    It accounts for the finite bin width by integrating the continuous kernel over the
    width of each radial bin.

    It also accounts for the growth factor already applied to each radial bin by
    dividing out the growth factor pre-smoothing, and then re-applying it after
    smoothing.
    """

    if not isinstance(sigmaP, np.ndarray):
        sigmaP = np.ones_like(chi) * sigmaP

    if not isinstance(D, np.ndarray):
        D = np.ones_like(chi) * D

    # This is the main parameter of the exponential kernel and comes from the FT of the
    # canonically defined Lorentzian
    a_1D = 2**0.5 / sigmaP
    a_r = a_1D[:, np.newaxis]

    # Get bin widths for the radial axis
    dchi_1D = calculate_width(chi)
    dchi_r = dchi_1D[:, np.newaxis]
    dchi_rp = dchi_1D[np.newaxis, :]

    chi_sep = np.abs(chi[:, np.newaxis] - chi[np.newaxis, :])

    def sinhc(x):
        return np.sinh(x) / x

    # Create a matrix to apply the smoothing with an exponential kernel.
    if full_channel_kernel:
        # See CHIME doclib:XXXX for derivations of these expressions
        K = (
            (2.0 / dchi_rp)
            * np.exp(-a_r * chi_sep)
            * np.sinh(a_r * dchi_r / 2.0)
            * np.sinh(a_r * dchi_rp / 2.0)
        )
        np.fill_diagonal(
            K,
            a_1D
            - (2.0 / dchi_1D)
            * np.exp(-a_1D * dchi_1D / 2.0)
            * np.sinh(a_1D * dchi_1D / 2.0),
        )
    else:
        # NOTE: because of the finite radial bins we should calculate the average
        # contribution over the width of each bin. That gives rise to the sinhc terms which
        # slightly boost the weights of the non-zero bins
        K = np.exp(-a_r * chi_sep) * sinhc(a_r * dchi_rp / 2.0)

        # The zero-lag bins are a special case because of the reflection about zero
        # Here the weight is slightly less than if we evaluated exactly at zero
        np.fill_diagonal(
            K, np.diagonal(np.exp(-a_r * dchi_rp / 4) * sinhc(a_r * dchi_rp / 4))
        )

    # Normalise each row to ensure conservation of mass
    K /= np.sum(K, axis=1)[:, np.newaxis]

    # Remove any already applied growth factor
    K /= D[np.newaxis, :]

    # Re-apply the growth factor for each redshift bin
    K *= D[:, np.newaxis]

    return K


def pad_frequencies(
    frequencies: np.ndarray,
    num: int = 1,
    use_FoG_kernel: bool = False,
    cosmology: cora_cosmology.Cosmology = None,
    sigma_P: float = None,
    FoG_threshold: float = 0.99,
    maxnum: int = None,
):
    """Pad list of frequencies.

    Input frequency list will be extended by an equal number of frequencies
    at the high and low ends.

    Parameters
    ----------
    frequencies
        Array of frequencies to pad.
    num
        Number of frequencies to pad by. If determining padding using FoG
        kernel, this number is treated as the minimum number of frequencies
        to pad by.
    use_FoG_kernel
        Whether to use the FoG kernel to determine the number of padding
        frequencies. The number of extra frequencies is computed such that the
        FoG convolution for lowest frequency in the input list captures
        contributions from some fraction of the integral of the FoG kernel,
        specified by `FoG_threshold`.
    cosmology
        Cosmology object to use for computing comoving distances.
    sigma_P
        FoG damping scale for kernel.
    FoG_threshold
        See description of `use_FoG_kernel`.
    maxnum
        Maximum number of padding frequencies.

    Returns
    -------
    padded_frequencies
        Padded list of frequencies.
    n_pad
        Number of frequencies added at each end of input list.
    n_pad_raw
        Number of frequencies that would have been if `maxnum` was not set.
    kernel_frac
        Fraction of kernel integral covered by padded frequencies at lowest
        input frequency. (Can be compared with `FoG_threshold` to determine
        how much of the kernel was cut off by the specified `maxnum`.)
    """

    if use_FoG_kernel and ((cosmology is None) or (sigma_P is None)):
        raise RuntimeError(
            "If using FoG kernel to determine padding, "
            "cosmology and sigmaP must both be specified"
        )

    # Sort frequencies and find spacing
    nfreq = len(frequencies)
    freqs_sorted = np.sort(frequencies)
    dfreq = np.median(np.diff(freqs_sorted))

    if not use_FoG_kernel:
        n_pad_raw = num
        n_pad = num
        kernel_frac = 1.0
    else:
        # Generate list of frequencies that's extended by a factor
        # of 2 at the low end
        freqs_extended = np.concatenate([freqs_sorted - nfreq * dfreq, freqs_sorted])

        # Compute corresponding extended list of comoving-distance
        # differences from lowest frequency in original list
        dx_extended = cosmology.comoving_distance(freqs_extended)
        dx_extended -= dx_extended[nfreq]

        # Evaluate normalized FoG kernel at values in this list
        # and evaluate cumulative sum
        kernel_extended = exponential_FoG_kernel_1d(
            dx_extended, sigma_P, normalize=True
        )
        kernel_cumsum = np.cumsum(kernel_extended)

        # Find number of extra frequencies required to cover
        # requested fraction of kernel integral
        n_pad_raw = max(
            np.sum(kernel_cumsum > 1 - FoG_threshold) - nfreq,
            0,
        )
        if maxnum is None:
            maxnum = nfreq + 1
        n_pad = min(n_pad_raw, maxnum)

        # Save fraction of kernel covered by padded frequencies
        # at lowest input frequency
        kernel_frac = (1 - kernel_cumsum)[nfreq - n_pad]

    # Make new frequency list
    padded_frequencies = np.concatenate(
        [
            freqs_sorted[0] - np.arange(1, n_pad + 1)[::-1] * dfreq,
            freqs_sorted,
            freqs_sorted[-1] + np.arange(1, n_pad + 1) * dfreq,
        ]
    )

    # Reverse order if initial sorting had an effect
    if not np.array_equal(padded_frequencies, frequencies):
        padded_frequencies = padded_frequencies[::-1]

    return padded_frequencies, n_pad, n_pad_raw, kernel_frac


def lognormal_transform(
    field: np.ndarray, out: Optional[np.ndarray] = None, axis: int = None
) -> np.ndarray:
    """Transform to a lognormal field with the same first order two point statistics.

    Parameters
    ----------
    field
        Input field.
    out
        Array to write the output into. If not supplied a new array is created.
    axis
        Calculate the normalising variance along this given axis. If not specified
        all axes are averaged over.

    Returns
    -------
    out
        The field the output was written into.
    """

    if out is None:
        out = np.zeros_like(field)
    elif field.shape != out.shape or field.dtype != out.dtype:
        raise ValueError("Given output array is incompatible.")

    if field is not out:
        out[:] = field

    var = field.var(axis=axis, keepdims=True)
    out -= var / 2.0

    np.exp(out, out=out)
    out -= 1

    return out


def assert_shape(arr, shape, name):

    if arr.ndim != len(shape):
        raise ValueError(
            f"Array {name} has wrong number of dimensions (got {arr.ndim}, "
            f"expected {len(shape)}"
        )

    if arr.shape != shape:
        raise ValueError(
            f"Array {name} has the wrong shape (got {arr.shape}, expected {shape}"
        )


class Pk2d_to_Cl:
    """Converter from 2d power spectrum to multi-frequency angular power spectrum.

    This class packages the algorithm from the angular_powerspectrum_fft
    method from the cora.signal.corr.RedshiftCorrelation class in a
    self-contained way, suitable for working with an externally-defined
    power spectrum in terms of k_parallel and k_perp. These different
    implementations should eventually be refactored, but will remain
    separate for now.

    Parameters
    ----------
    pk2d : function
        Power spectrum function, with arguments (k_parallel, k_perp).
    chi_of_z : function
        Function to convert redshift to comoving distance.
    kparmax : float, optional
        Maximum k_parallel for integration. (Note that the minimum
        k_parallel is 0.) Default: 50.
    nkpar : int, optional
        Number of k_parallel values to use in discrete cosine transform.
        Default: 32768.
    kperpmin, kperpmax : float, optional
        Minimum and maximum k_perp values. These determine the minimum
        and maximum ell values that are accessible. Default: 1e-4 and 40.
    nkperp : int, optional
        Number of k_perp values to interpolate between. Default: 500.
    k_meshgrid : bool, optional
        Whether pk2d function can accept (kpar, kperp) values evaluated
        on a numpy meshgrid. Default: True.
    """

    def __init__(
        self,
        pk2d,
        chi_of_z,
        kparmax=50.0,
        nkpar=32768,
        kperpmin=1e-4,
        kperpmax=40.0,
        nkperp=500,
        k_meshgrid=True,
    ):

        self.kparmax = kparmax
        self.nkpar = nkpar
        self.kperpmin = kperpmin
        self.kperpmax = kperpmax
        self.nkperp = nkperp
        self.chi_of_z = chi_of_z

        kperp = np.logspace(np.log10(kperpmin), np.log10(kperpmax), nkperp)
        kpar = np.linspace(0, kparmax, nkpar)

        if k_meshgrid:
            kperp, kpar = np.meshgrid(kperp, kpar, indexing="ij")
        else:
            kperp = kperp[:, np.newaxis]
            kpar = kpar[np.newaxis, :]

        dd = pk2d(kpar, kperp)

        self.aps_cache = dct(dd, type=1) * kparmax / (2 * nkpar)

    def eval(self, la, za1, za2, const_chi=False):
        """Evaluate C_ell(z, z').

        Parameters
        ----------
        la : array_like
            Ell values to evaluate at.
        za1, za2 : array_like
            Redshifts to evaluate at
        const_chi : bool, optional
            Evaluate k-ell relationship and prefactor at
            constant comoving distance. Default: False.

        Returns
        -------
        cl : array_like
            Array of C_ell(z, z') values.
        """

        xa1 = self.chi_of_z(za1)
        xa2 = self.chi_of_z(za2)

        xc = 0.5 * (xa1 + xa2)
        if const_chi:
            xc = np.mean(xc)
        rpar = np.abs(xa2 - xa1)

        # Bump anything that is zero upwards to avoid a log zero warning.
        la = np.where(la == 0.0, 1e-10, la)

        x = (
            (np.log10(la) - np.log10(xc * self.kperpmin))
            / np.log10(self.kperpmax / self.kperpmin)
            * (self.nkperp - 1)
        )
        y = rpar / (np.pi / self.kparmax)

        def _interp2d(arr, x, y):
            x, y = np.broadcast_arrays(x, y)
            sh = x.shape

            x, y = x.flatten(), y.flatten()
            v = np.zeros_like(x)
            bilinearmap.interp(arr, x, y, v)

            return v.reshape(sh)

        psdd = _interp2d(self.aps_cache, x, y)

        return 1 / (xc**2 * np.pi) * psdd
