"""
===============================================
Co-ordinate Utilities (:mod:`~cora.util.coord`)
===============================================

A module of useful functions when dealing with Spherical Polar
co-ordinates.

Transforms
==========

.. autosummary::
    :toctree: generated/

    sph_to_cart
    cart_to_sph

Vector operations
=================

.. autosummary::
    :toctree: generated/

    sph_dot

Tangent Plane
=============

.. autosummary::
    :toctree: generated/

    thetaphi_plane
    thetaphi_plane_cart

Miscellaneous
=============

.. autosummary::
    :toctree: generated/

    great_circle_points

"""


import numpy as np


def sph_to_cart(sph_arr):
    """Convert a vector in Spherical Polar coordinates to Cartesians.

    Parameters
    ----------
    sph_arr : np.ndarry
        A vector (or array of) in spherical polar co-ordinates. Values should be
        packed as [r, theta, phi] along the last axis. Alternatively they can be
        packed as [theta, phi] in which case r is assumed to be one.

    Returns
    -------
    cart_arr : np.ndarry
        Array of equivalent vectors in cartesian coordinartes.
    """

    # Create an array of the correct size for the output.
    shape = list(sph_arr.shape)
    shape[-1] = 3
    cart_arr = np.empty(shape)

    # Useful quantities
    if sph_arr.shape[-1] == 3:
        radius = sph_arr[..., 0]
    else:
        radius = 1.0

    sintheta = np.sin(sph_arr[..., -2])

    cart_arr[..., 0] = radius * sintheta * np.cos(sph_arr[..., -1])  # x-axis
    cart_arr[..., 1] = radius * sintheta * np.sin(sph_arr[..., -1])  # y-axis
    cart_arr[..., 2] = radius * np.cos(sph_arr[..., -2])  # z-axis

    return cart_arr


def cart_to_sph(cart_arr):
    """Convert a cartesian vector into Spherical Polars.

    Uses the same convention as `sph_to_cart`.

    Parameters
    ----------
    cart_arr : np.ndarray
        Array of cartesians.

    Returns
    -------
    sph_arr : np.ndarray
        Array of spherical polars (packed as [[ r1, theta1, phi1], [r2, theta2,
        phi2], ...]
    """
    sph_arr = np.empty_like(cart_arr)

    sph_arr[..., 2] = np.arctan2(cart_arr[..., 1], cart_arr[..., 0])

    sph_arr[..., 0] = np.sum(cart_arr ** 2, axis=-1) ** 0.5

    sph_arr[..., 1] = np.arccos(cart_arr[..., 2] / sph_arr[..., 0])

    return sph_arr


def sph_dot(arr1, arr2):
    """Take the scalar product in spherical polars.

    Parameters
    ----------
    arr1, arr2 : np.ndarray
        Two arrays of vectors in spherical polars [theta, phi], (or
        alternatively as [theta, phi]). Should be broadcastable against each
        other.

    Returns
    -------
    dot : np.ndarray
        An array of the dotted vectors.

    """
    return np.inner(sph_to_cart(arr1), sph_to_cart(arr2))


def thetaphi_plane(sph_arr):
    """For each position, return the theta, phi unit vectors (in spherical
    polars).

    Parameters
    ----------
    sph_arr : np.ndarray
        Angular positions (in spherical polar co-ordinates).

    Returns
    -------
    thetahat, phihat : np.ndarray
        Unit vectors in the theta and phi directions (still in spherical
        polars).
    """
    thetahat = sph_arr.copy()
    thetahat[..., -2] += np.pi / 2.0

    phihat = sph_arr.copy()
    phihat[..., -2] = np.pi / 2.0
    phihat[..., -1] += np.pi / 2.0

    if sph_arr.shape[-1] == 3:
        thetahat[..., 0] = 1.0
        phihat[..., 0] = 1.0

    return thetahat, phihat


def thetaphi_plane_cart(sph_arr):
    """For each position, return the theta, phi unit vectors (in cartesians).

    Parameters
    ----------
    sph_arr : np.ndarray
        Angular positions (in spherical polar co-ordinates).

    Returns
    -------
    thetahat, phihat : np.ndarray
        Unit vectors in the theta and phi directions (in cartesian coordinates).
    """
    that, phat = thetaphi_plane(sph_arr)

    return sph_to_cart(that), sph_to_cart(phat)


def groundsph_to_cart(gsph, zenith):

    ta = gsph[..., 0]
    pa = gsph[..., 1]

    zenc = sph_to_cart(zenith)

    that, phat = thetaphi_plane_cart(zenith)

    cart_arr = np.empty(ta.shape + (3,), dtype=ta.dtype)

    cart_arr = (
        ta[..., np.newaxis] * that
        + pa[..., np.newaxis] * phat
        + ((1.0 - ta ** 2 - pa ** 2) ** 0.5)[..., np.newaxis] * zenc
    )

    return cart_arr


def great_circle_points(sph1, sph2, npoints):
    """Return the intermediate points on the great circle between points `sph1`
    and `sph2`.

    Parameters
    ----------
    sph1, sph2 : np.ndarray
        Points on sphere, packed as [theta, phi]
    npoints : integer
        Number of intermediate points

    Returns
    -------
    intpoints : np.ndarray
        Intermediate points, packed as [ [theta1, phi1], [theta2, phi2], ...]
    """

    # Turn points on circle into Cartesian vectors
    c1 = sph_to_cart(sph1)
    c1 = sph_to_cart(sph2)

    # Construct the difference vector
    dc = c2 - c1
    if np.sum(dc ** 2) == 4.0:
        raise Exception("Points antipodal, path undefined.")

    # Create difference unit vector
    dcn = dc / np.dot(dc, dc) ** 0.5

    # Calculate central angle, and angle on corner O C1 C2
    thc = np.arccos(np.dot(c1, c2))
    alp = np.arccos(-np.dot(c1, dcn))

    # Map regular angular intervals into distance along the displacement between
    # C1 and C2
    f = np.linspace(0.0, 1.0, npoints)
    x = np.sin(f * thc) / np.sin(f * thc + alp)

    # Generate the Cartesian vectors for intermediate points
    cv = x[:, np.newaxis] * dcn[np.newaxis, :] + c1[np.newaxis, :]

    # Return the angular poitions of the points
    return cart_to_sph(cv)[:, 1:]
