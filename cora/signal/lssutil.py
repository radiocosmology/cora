from typing import Callable, Union, Tuple, Optional

import numpy as np

import healpy

from caput import config, tools

from ..util import cubicspline as cs
from ..util import hputil
from ..util.nputil import FloatArrayLike


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
    dx = chi.ptp() / (N - 1)

    # The effective L is slightly longer than the range because of the channel widths
    L = N * dx

    cn = np.fft.rfft(maps, axis=0) / N

    # Need to use this wrapper from cora (rather than any healpy functions) as we must
    # transform a complex field
    almn_square = np.array([hputil.sphtrans_complex(m, lmax) for m in cn])

    l = np.arange(lmax + 1)
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
    cln /= (2 * l + 1)[np.newaxis, :]

    # Get the effective axes in k space
    kperp = l / chi_mean
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


def exponential_FoG_kernel(
    chi: np.ndarray, sigmaP: FloatArrayLike, D: FloatArrayLike
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
    a = 2**0.5 / sigmaP
    ar = a[:, np.newaxis]

    # Get bin widths for the radial axis
    dchi = calculate_width(chi)[np.newaxis, :]

    chi_sep = np.abs(chi[:, np.newaxis] - chi[np.newaxis, :])

    def sinhc(x):
        return np.sinh(x) / x

    # Create a matrix to apply the smoothing with an exponential kernel.
    # NOTE: because of the finite radial bins we should calculate the average
    # contribution over the width of each bin. That gives rise to the sinhc terms which
    # slightly boost the weights of the non-zero bins
    K = np.exp(-ar * chi_sep) * sinhc(ar * dchi / 2.0)

    # The zero-lag bins are a special case because of the reflection about zero
    # Here the weight is slightly less than if we evaluated exactly at zero
    np.fill_diagonal(K, np.diagonal(np.exp(-ar * dchi / 4) * sinhc(ar * dchi / 4)))

    # Normalise each row to ensure conservation of mass
    K /= np.sum(K, axis=1)[:, np.newaxis]

    # Remove any already applied growth factor
    K /= D[np.newaxis, :]

    # Re-apply the growth factor for each redshift bin
    K *= D[:, np.newaxis]

    return K


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
