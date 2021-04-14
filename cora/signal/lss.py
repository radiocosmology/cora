from typing import Tuple, Optional
from cora.core import skysim
from cora.util import cosmology
from cora.util import hputil
from cora.util.pmesh import (
    calculate_positions,
    _bin_delta,
    _pixel_weights,
    _radial_weights,
)

import numpy as np

import healpy

from caput import config
from caput import pipeline

from cora.util.cosmology import Cosmology, growth_factor
from cora.util import units

from draco.core import task
from draco.core.containers import Map

from .lsscontainers import InterpolatedFunction, InitialLSS, BiasedLSS
from . import lssutil
from .lssutil import FloatArrayLike
from . import corrfunc


class CalculateCorrelations(task.SingleTask):
    """Calculate the density and potential correlation functions from a power spectrum."""

    run = False

    def setup(self):

        # TODO: get PS function in a better way
        # TODO: ensure we are scaled to the correct redshift
        from . import corr21cm

        cr = corr21cm.Corr21cm()
        self.power_spectrum = cr.ps_vv
        self.ps_redshift = cr.ps_redshift
        self.cosmology = cr.cosmology

    def _ps_n(self, n):
        # Get the power spectrum multiplied by k**-n and a low-k cutoff

        def _ps(k):
            Dr = cosmology.growth_factor(0.0, self.cosmology) / cosmology.growth_factor(
                self.ps_redshift, self.cosmology
            )
            return (
                Dr ** 2
                * lssutil.cutoff(k, -4, 1, 0.2, 6)
                * self.power_spectrum(k)
                * k ** -n
            )

        return _ps

    def process(self) -> InterpolatedFunction:
        """Calculate the correlation functions.

        Returns
        -------
        corr
            An container holding the density correlation function (`corr0`), the
            potential correlation function (`corr4`) and their cross correlation
            (`corr2`).
        """

        if self.run:
            raise pipeline.PipelineStopIteration()

        # Calculate the correlation functions
        self.log.debug("Generating C_dd(r)")
        k0, c0 = corrfunc.ps_to_corr(
            self._ps_n(0), samples=True, samples_per_decade=1000
        )
        self.log.debug("Generating C_dp(r)")
        k2, c2 = corrfunc.ps_to_corr(
            self._ps_n(2), h=3e-6, samples=True, samples_per_decade=1000
        )
        self.log.debug("Generating C_pp(r)")
        k4, c4 = corrfunc.ps_to_corr(
            self._ps_n(4), h=2e-6, samples=True, samples_per_decade=1000
        )

        func = InterpolatedFunction()

        func.add_function("corr0", k0, c0, type="sinh", x_t=k0[1], f_t=1e-12)
        func.add_function("corr2", k2, c2, type="sinh", x_t=k2[1], f_t=1e-6)
        func.add_function("corr4", k4, c4, type="sinh", x_t=k4[1], f_t=1e2)

        self.run = True

        return func


class GenerateInitialLSS(task.SingleTask):
    """Generate the initial LSS distribution.

    Attributes
    ----------
    nside : int
        The Healpix resolution to use.
    redshift : np.ndarray
        The redshifts for the map slices.
    frequencies : np.ndarray
        The frequencies that determine the redshifts for the map slices.
        Overrides redshift property if specified.
    num : int
        The number of simulations to generate.
    start_seed : int
        The random seed to use for generating the first simulation.
        Sim i (indexed from 0) is generated using start_seed+i.
        Default: 0.
    xromb : int, optional
        Romberg order for integrating C_ell over radial bins.
        xromb=0 turns off this integral. Default: 2.
    leg_q : int, optional
        Integration accuracy parameter for Legendre transform for C_ell
        computation. Default: 4.
    """

    nside = config.Property(proptype=int)
    redshift = config.Property(proptype=lssutil.linspace, default=None)
    frequencies = config.Property(proptype=lssutil.linspace, default=None)
    num = config.Property(proptype=int, default=1)
    start_seed = config.Property(proptype=int, default=0)
    xromb = config.Property(proptype=int, default=2)
    leg_q = config.Property(proptype=int, default=4)

    def setup(self, correlation_functions: InterpolatedFunction):
        """Setup the task.

        Parameters
        ----------
        correlation_functions
            The pre-calculated correlation functions.
        """

        # TODO: get PS function in a better way
        # TODO: ensure we are scaled to the correct redshift
        from . import corr21cm

        cr = corr21cm.Corr21cm()
        self.cosmology = cr.cosmology
        self.correlation_functions = correlation_functions

        if self.redshift is None and self.frequencies is None:
            raise RuntimeError("Redshifts or frequencies must be specified!")

        self.seed = self.start_seed

    def process(self) -> InitialLSS:
        """Generate a realisation of the LSS initial conditions.

        Returns
        -------
        InitialLSS
            The LSS initial conditions.
        """

        # Stop if we've already generated enough simulated fields
        if self.num == 0:
            raise pipeline.PipelineStopIteration()
        self.num -= 1

        corr0 = self.correlation_functions.get_function("corr0")
        corr2 = self.correlation_functions.get_function("corr2")
        corr4 = self.correlation_functions.get_function("corr4")

        if self.frequencies is None:
            redshift = self.redshift
        else:
            redshift = units.nu21 / self.frequencies - 1.0

        xa = self.cosmology.comoving_distance(redshift)
        lmax = 3 * self.nside

        self.log.debug("Generating C_l(x, x')")
        cla0 = corrfunc.corr_to_clarray(corr0, lmax, xa, xromb=self.xromb, q=self.leg_q)
        cla2 = corrfunc.corr_to_clarray(corr2, lmax, xa, xromb=self.xromb, q=self.leg_q)
        cla4 = corrfunc.corr_to_clarray(corr4, lmax, xa, xromb=self.xromb, q=self.leg_q)

        nz = len(redshift)

        # Create an extended covariance matrix capturing the cross correlations between
        # the fields
        cla = np.zeros((lmax + 1, 2 * nz, 2 * nz))
        cla[:, :nz, :nz] = cla4
        cla[:, :nz, nz:] = cla2
        cla[:, nz:, :nz] = cla2
        cla[:, nz:, nz:] = cla0

        self.log.info("Generating realisation of fields using seed %d" % self.seed)
        np.random.seed(self.seed)
        sky = skysim.mkfullsky(cla, self.nside)

        if self.frequencies is not None:
            f = InitialLSS(
                cosmology=self.cosmology,
                nside=self.nside,
                freq=self.frequencies,
            )
        else:
            f = InitialLSS(
                cosmology=self.cosmology,
                nside=self.nside,
                redshift=redshift,
            )

        # Local shape and offset along chi axis
        loff = f.phi.local_offset[0]
        lshape = f.phi.local_shape[0]

        # f.phi[:] = sky[:nz]
        # f.delta[:] = sky[nz:]
        f.phi[:] = sky[loff : loff + lshape]
        f.delta[:] = sky[nz + loff : nz + loff + lshape]

        self.seed += 1

        return f


class GenerateBiasedFieldBase(task.SingleTask):
    r"""Take the initial field and generate a biased field.

    Note that this applies a bias in Lagrangian space.

    .. math::
        \delta_B = D(z) b_1(z) \delta_L
        + D(z)^2 b_2(z) (\delta_L^2 - \langle \delta_L^2 \rangle)

    Attributes
    ----------
    lightcone : bool
        Generate the field on the lightcone or not. If not, all slices will be at a
        fixed epoch specified by the `redshift` attribute.
    redshift : float
        The redshift of the field (if not on the `lightcone`).
    """

    lightcone = config.Property(proptype=bool, default=True)
    redshift = config.Property(proptype=float, default=None)

    lognormal = config.Property(proptype=bool, default=False)

    def _bias_1(self, z: FloatArrayLike) -> FloatArrayLike:
        """Get the first order Lagrangian bias as a function of redshift.

        Parameters
        ----------
        z
            Redshift.

        Returns
        -------
        bias
            The first order Lagrangian bias.
        """
        raise NotImplementedError("Must be overridden in subclass.")

    def _bias_2(self, z: FloatArrayLike) -> FloatArrayLike:
        """Get the second order Lagrangian bias as a function of redshift.

        Parameters
        ----------
        z
            Redshift.

        Returns
        -------
        bias
            The second order Lagrangian bias.
        """
        raise NotImplementedError("Must be overridden in subclass.")

    def process(self, f: InitialLSS) -> BiasedLSS:
        """Create a biased field."""

        biased_field = BiasedLSS(
            lightcone=self.lightcone,
            fixed_redshift=self.redshift,
            axes_from=f,
            attrs_from=f,
        )
        biased_field.delta[:] = 0.0

        z = f.redshift if self.lightcone else self.redshift * np.ones_like(f.chi)
        D = growth_factor(z, f.cosmology) / growth_factor(0, f.cosmology)

        # Apply any first order bias
        try:
            b1 = self._bias_1(z)

            # Local offset and shape along chi axis
            loff = f.delta.local_offset[0]
            lshape = f.delta.local_shape[0]

            biased_field.delta[:] += (D * b1)[
                loff : loff + lshape, np.newaxis
            ] * f.delta[:]
        except NotImplementedError:
            self.log.info("First order bias is not implemented. This is a bit odd.")

        # Apply any second order bias
        try:
            b2 = self._bias_2(z)
            # NOTE: should this average be over redshift shells, not the full volume?
            d2m = (f.delta[:] ** 2).mean()
            biased_field.delta[:] += (D ** 2 * b2)[:, np.newaxis] * (
                f.delta[:] ** 2 - d2m
            )
        except NotImplementedError:
            self.log.debug("No second order bias to apply.")

        if self.lognormal:
            lognormal_transform(
                biased_field.delta[:],
                out=biased_field.delta[:],
                axis=(1 if self.lightcone else None),
            )

        return biased_field

    def _crop_low(self, x, cut=0.0):

        # Clip values in place
        mask = x < cut
        x[mask] = cut

        self.log.debug(f"Fraction of pixels cropped {mask.mean()}.")


class GenerateConstantBias(GenerateBiasedFieldBase):
    """Generate a field with a constant linear bias.

    Attributes
    ----------
    bias : float
        The Lagrangian bias. At first order this is the related to the Eulerian bias
        by :math:`b_L = b_E - 1`.
    """

    bias = config.Property(proptype=float, default=0.0)

    def _bias_1(self, z: FloatArrayLike) -> FloatArrayLike:
        return np.ones_like(z) * self.bias


class DynamicsBase(task.SingleTask):
    """Base class for generating final fields from biased fields.

    Attributes
    ----------
    redshift_space : bool, optional
        Generate the final field in redshift space or not.
    """

    redshift_space = config.Property(proptype=bool, default=True)

    def _validate_fields(self, initial_field: InitialLSS, biased_field: BiasedLSS):
        """Check the field are consistent.

        Parameters
        ----------
        initial_field
            The initial field.
        biased_field
            The biased field.
        """

        if (initial_field.chi[:] != biased_field.chi[:]).any():
            raise ValueError(
                "Radial axes do not match between initial and biased fields."
            )

        if (biased_field.index_map["pixel"] != initial_field.index_map["pixel"]).any():
            raise ValueError(
                "Angular axes do not match between initial and biased fields."
            )

    def _get_props(
        self,
        biased_field: BiasedLSS,
    ) -> Tuple[Cosmology, int, bool, np.ndarray, np.ndarray]:
        """Get useful properties.

        Parameters
        ----------
        biased_field
            The biased field to fetch the params out of.

        Returns
        -------
        cosmology
            The Cosmology object.
        nside
            The NSIDE of the map axis.
        lightcone
            Is the field on the lightcone or not?
        chi
            The radial coordinate.
        z
            The redshift of each slice. This changes if the field is on the lightcone
            or not.
        """
        # redshift_array = units.nu21 / self.freqs_full - 1.
        c = biased_field.cosmology

        nside = healpy.npix2nside(biased_field.delta.shape[1])
        chi = biased_field.chi[:]

        # Calculate the redshift of each radial slice, this depends if we are
        # evalulating on the lightcone or not
        if biased_field.lightcone:
            if "redshift" not in biased_field.index_map:
                raise ValueError(
                    "Incoming biased field does not have a redshift label."
                )
            za = biased_field.redshift
        else:
            za = np.ones_like(chi) * biased_field.fixed_redshift

        return c, nside, biased_field.lightcone, chi, za


class ZeldovichDynamics(DynamicsBase):
    """Generate a simulated LSS field using Zel'dovich dynamics."""

    sph = config.Property(proptype=bool, default=True)

    def process(self, initial_field: InitialLSS, biased_field: BiasedLSS) -> BiasedLSS:
        """Apply Zel'dovich dynamics to the biased field to get the final field.

        Parameters
        ----------
        initial_field
            The initial conditions of the volume.
        biased_field
            The biased field in Lagrangian space.

        Returns
        -------
        BiasedLSS
            The field Eulerian space with Zel'dovich dynamics applied. This can be in
            redshift space or real space depending on the value of `redshift_space`.
        """
        self._validate_fields(initial_field, biased_field)
        c, nside, _, chi, za = self._get_props(biased_field)

        # Take the gradient of the Lagrangian potential to get the
        # displacements
        vpsi = lssutil.gradient(initial_field.phi[:], chi)

        # Apply growth factor:
        D = cosmology.growth_factor(za, c) / cosmology.growth_factor(0, c)
        vpsi *= D[np.newaxis, :, np.newaxis]

        theta, _ = hputil.ang_positions(nside).T

        # Divide by comoving distance to get angular displacements
        vpsi[1:3] /= chi[np.newaxis, :, np.newaxis]

        # Divide by sin(theta) to get phi displacements, not angular displacements
        vpsi[2] /= np.sin(theta[np.newaxis, :])

        # Add the extra radial displacement to get into redshift space
        if self.redshift_space:
            fr = cosmology.growth_rate(za, c)
            vpsi[0] *= (1 + fr)[:, np.newaxis]

        # Create the final field container
        final_field = BiasedLSS(axes_from=biased_field, attrs_from=biased_field)
        final_field.delta[:] = 0.0

        # Calculate the Zel'dovich displaced field and assign directly into the output
        # container
        # za_density(vpsi, biased_field.delta[:], nside, chi, out=final_field.delta[:])
        delta_m = (initial_field.delta[:] * D[:, np.newaxis]).view(np.ndarray)
        delta_bias = biased_field.delta[:].view(np.ndarray)

        if self.sph:
            za_density_sph(
                vpsi,
                delta_bias,
                delta_m,
                chi,
                final_field.delta[:],
            )
        else:
            za_density_grid(vpsi, delta_bias, delta_m, chi, final_field.delta[:])
        return final_field


class LinearDynamics(DynamicsBase):
    """Generate a simulated LSS field using first order standard perturbation theory."""

    def process(self, initial_field: InitialLSS, biased_field: BiasedLSS) -> BiasedLSS:
        """Apply Eulerian linear dynamics to the biased field to get the final field.

        This produces a realisation assuming first order Eulerian perturbation theory.

        Parameters
        ----------
        initial_field
            The initial conditions of the volume.
        biased_field
            The biased field in *Lagrangian* space. This is true even though we are
            eventually generating an Eulerian field.

        Returns
        -------
        BiasedLSS
            The field Eulerian space with linear dynamics applied. This can be in
            redshift space or real space depending on the value of `redshift_space`.
        """

        self._validate_fields(initial_field, biased_field)
        c, _, __, chi, za = self._get_props(biased_field)

        # Create the final field container
        final_field = BiasedLSS(axes_from=biased_field, attrs_from=biased_field)

        # Get growth factor:
        D = cosmology.growth_factor(za, c) / cosmology.growth_factor(0, c)

        # Copy over biased field
        final_field.delta[:] = biased_field.delta[:]

        # Add in standard linear term
        loff = final_field.delta.local_offset[0]
        lshape = final_field.delta.local_shape[0]
        final_field.delta[:] += (
            D[loff : loff + lshape, np.newaxis] * initial_field.delta[:]
        )

        if self.redshift_space:
            fr = cosmology.growth_rate(za, c)
            vterm = lssutil.diff2(
                initial_field.phi[:], chi[loff : loff + lshape], axis=0
            )
            vterm *= -(D * fr)[loff : loff + lshape, np.newaxis]

            final_field.delta[:] += vterm

        return final_field


class BiasedLSSToMap(task.SingleTask):
    """Convert a BiasedLSS object into a Map object.

    Attributes
    ----------
    use_mean_21cmT : bool
        Multiply map by mean 21cm temperature as a function of z (default: False).
    """

    use_mean_21cmT = config.Property(proptype=int, default=False)
    map_prefactor = config.Property(proptype=float, default=1.0)

    def process(self, biased_lss) -> Map:
        """Generate a realisation of the LSS initial conditions.

        Parameters
        ----------
        biased_lss : BiasedLSS
            Input BiasedLSS container.

        Returns
        -------
        out_map : Map
            Output Map container.
        """

        # Form frequency map in appropriate format to feed into Map container
        n_freq = len(biased_lss.freq)
        freqmap = np.zeros(
            n_freq, dtype=[("centre", np.float64), ("width", np.float64)]
        )
        freqmap["centre"][:] = biased_lss.freq[:]
        freqmap["width"][:] = np.abs(np.diff(biased_lss.freq[:])[0])

        # Make new map container and copy delta into Stokes I component
        m = Map(
            freq=freqmap, polarisation=True, axes_from=biased_lss, attrs_from=biased_lss
        )
        m.map[:, 0, :] = biased_lss.delta[:, :]

        # Multiply map by specific prefactor, if desired
        if self.map_prefactor != 1:
            self.log.info("Multiplying map by %g" % self.map_prefactor)
            m.map[:] *= self.map_prefactor

        # If desired, multiply by Tb(z)
        if self.use_mean_21cmT:
            from . import corr21cm

            cr = corr21cm.Corr21cm()

            loff = m.map.local_offset[0]
            lshape = m.map.local_shape[0]
            m.map[:, 0] *= cr.T_b(biased_lss.redshift)[loff : loff + lshape, np.newaxis]

        return m


def za_density_grid(
    psi: np.ndarray,
    delta_bias: np.ndarray,
    delta_m: np.ndarray,
    chi: np.ndarray,
    out: np.ndarray,
):
    """Calculate the density field under the Zel'dovich approximations.

    This treats the initial mass as a grid of cuboids which are moved to their final
    location and their mass allocate to pixels in the final grid according to their
    overlap.

    .. warning::
        Healpix actually has a very weird definition of the points for a bilinear
        interpolation, so this doesn't really work like it should.

        Also this does not apply any scaling based on the change in volume of the
        grid cell.

    Parameters
    ----------
    psi
        The vector displacement field.
    delta_bias
        The biased density field.
    delta_m
        The underlying matter density field. Used to calculate the density dependent
        particle smoothing scale.
    chi
        The comoving distance of each slice.
    out
        The array to save the output into.
    """
    nchi, npix = delta_bias.shape

    # Validate all the array shapes
    _assert_shape(psi, (3, nchi, npix), "psi")
    _assert_shape(delta_m, (nchi, npix), "delta_m")
    _assert_shape(chi, (nchi,), "chi")
    _assert_shape(out, (nchi, npix), "out")

    nside = healpy.npix2nside(npix)

    angpos = np.array(healpy.pix2ang(nside, np.arange(npix)))

    # Extend the chi bins to add another cell at each end
    # This makes it easier to deal with interpolation at the boundaries
    chi_ext = np.zeros(len(chi) + 2, dtype=chi.dtype)
    chi_ext[1:-1] = chi
    chi_ext[0] = chi[0] - (chi[1] - chi[0])
    chi_ext[-1] = chi[-1] + (chi[-1] - chi[-2])

    # Array to hold the density obtained from the ZA approximation
    if out is None:
        out = np.zeros((nchi, npix), dtype=delta_bias.dtype)

    angpos = np.array(healpy.pix2ang(nside, np.arange(npix)))

    for ii in range(nchi):

        # Initial density
        density_slice = 1 + delta_bias[ii]
        # Coordinate displacements
        psi_slc = psi[:, ii]

        # Final angles and radial coordinate
        new_angpos = calculate_positions(angpos, psi_slc[1:])
        new_chi = chi[ii] + psi_slc[0]

        # Get the weights for the surrounding pixels
        pixel_ind, pixel_weight = healpy.get_interp_weights(
            nside, new_angpos[0], new_angpos[1]
        )

        # Find
        chi_ext_ind = np.digitize(new_chi, chi_ext)
        chi0 = chi_ext[(chi_ext_ind - 1) % (nchi + 2)]
        chi1 = chi_ext[chi_ext_ind % (nchi + 2)]
        dchi = chi1 - chi0

        # Calculate the indices and weights in the radial direction
        w0 = np.abs((chi1 - new_chi) / dchi)
        w1 = np.abs((new_chi - chi0) / dchi)
        i0 = chi_ext_ind - 2
        i1 = chi_ext_ind - 1

        w0[np.where((i0 < 0) | (i0 >= nchi))] = -1
        w1[np.where((i1 < 0) | (i1 >= nchi))] = -1

        radial_ind = np.array([i0, i1])
        radial_weight = np.array([w0, w1])

        _bin_delta(
            density_slice,
            pixel_ind.T.astype(np.int32, order="C"),
            pixel_weight.T.copy(),
            radial_ind.T.astype(np.int32, order="C"),
            radial_weight.T.copy(),
            out,
        )

    out[:] -= 1.0

    return out


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


def _assert_shape(arr, shape, name):

    if arr.ndim != len(shape):
        raise ValueError(
            f"Array {name} has wrong number of dimensions (got {arr.ndim}, "
            f"expected {len(shape)}"
        )

    if arr.shape != shape:
        raise ValueError(
            f"Array {name} has the wrong shape (got {arr.shape}, expected {shape}"
        )


def za_density_sph(
    psi: np.ndarray,
    delta_bias: np.ndarray,
    delta_m: np.ndarray,
    chi: np.ndarray,
    out: np.ndarray,
):
    """Calculate the density field under the Zel'dovich approximations.

    This treats the initial mass as a grid of particles that are displaced to their
    final location. Their mass is allocated to the underlying final grid with a
    Gaussian profile that truncates beyond their immediate nearest neighbours.

    Parameters
    ----------
    psi
        The vector displacement field.
    delta_bias
        The biased density field.
    delta_m
        The underlying matter density field. Used to calculate the density dependent
        particle smoothing scale.
    chi
        The comoving distance of each slice.
    out
        The array to save the output into.
    """
    nchi, npix = delta_bias.shape
    nside = healpy.npix2nside(npix)

    # Validate all the array shapes
    _assert_shape(psi, (3, nchi, npix), "psi")
    _assert_shape(delta_m, (nchi, npix), "delta_m")
    _assert_shape(chi, (nchi,), "chi")
    _assert_shape(out, (nchi, npix), "out")

    # Set the nominal smoothing scales at mean density
    sigma_chi = np.mean(np.abs(np.diff(chi))) / 2
    sigma_ang = healpy.nside2resol(nside) / 2

    angpos = np.array(healpy.pix2ang(nside, np.arange(npix)))

    # For every pixel in the map figure out the indices of its nearest neighbours
    nn_ind = np.zeros((npix, 9), dtype=int)
    nn_ind[:, 0] = np.arange(npix)
    nn_ind[:, 1:] = healpy.get_all_neighbours(nside, nn_ind[:, 0]).T

    # For each neighbour calculate its 3D position vector
    nn_vec = np.array(healpy.pix2vec(nside, nn_ind.ravel())).T.reshape(npix, 9, 3)
    nn_vec = np.ascontiguousarray(nn_vec)

    # Pre-allocate arrays to save the indices and weights into
    pixel_ind = np.zeros((npix, 9), dtype=np.int32)
    pixel_weight = np.zeros((npix, 9), dtype=np.float64)
    radial_ind = np.zeros((npix, 3), dtype=np.int32)
    radial_weight = np.zeros((npix, 3), dtype=np.float64)

    for ii in range(nchi):

        # Calculate the biased mass associated with each particle
        density_slice = 1 + delta_bias[ii]

        # Coordinate displacements
        psi_slc = psi[:, ii]

        # Calculate the amount we expect each particle to change volume
        # NOTE: we clip at the low delta end to stop particles growing to enormous
        # sizes
        scaling = np.clip(1 + delta_m[ii], 0.1, 3.0) ** (-1.0 / 3)

        # Final angles and radial coordinate
        new_angpos = calculate_positions(angpos, psi_slc[1:])
        new_chi = chi[ii] + psi_slc[0]

        # Get the pixel index that each new angular position maps into
        new_ang_ind = healpy.ang2pix(nside, new_angpos[0], new_angpos[1])

        # Calculate the 3D vector for each angular position
        new_ang_vec = np.ascontiguousarray(healpy.ang2vec(new_angpos[0], new_angpos[1]))

        # Calculate the weights for each neighbour in the pixel direction, these should
        # add up to 1
        _pixel_weights(
            new_ang_ind,
            new_ang_vec,
            scaling,
            sigma_ang,
            nn_ind,
            nn_vec,
            pixel_ind,
            pixel_weight,
        )

        # Calculate the radial weights
        chi_ind = np.searchsorted(chi, new_chi)
        _radial_weights(
            chi_ind, new_chi, scaling, sigma_chi, 1, chi, radial_ind, radial_weight
        )

        # Loop over all particles and spread their mass into the final pixel locations
        _bin_delta(
            density_slice,
            pixel_ind,
            pixel_weight,
            radial_ind,
            radial_weight,
            out,
        )

    out[:] -= 1.0

    return out
