from typing import Tuple, Optional
from cora.core import skysim
from cora.util import cosmology
from cora.util import hputil
from cora.util.pmesh import calculate_positions, _bin_delta

import numpy as np

import healpy

from caput import config
from caput import pipeline

from cora.util.cosmology import Cosmology, growth_factor

from draco.core import task

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
        k0, c0 = corrfunc.ps_to_corr(self._ps_n(0), samples=True)
        self.log.debug("Generating C_dp(r)")
        k2, c2 = corrfunc.ps_to_corr(self._ps_n(2), h=3e-6, samples=True)
        self.log.debug("Generating C_pp(r)")
        k4, c4 = corrfunc.ps_to_corr(self._ps_n(4), h=2e-6, samples=True)

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
    num : int
        The number of simulations to generate.
    """

    nside = config.Property(proptype=int)
    redshift = config.Property(proptype=lssutil.linspace)
    num = config.Property(proptype=int, default=1)

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

        xa = self.cosmology.comoving_distance(self.redshift)
        lmax = 3 * self.nside

        # TODO: configurable accuracy parameters, xromb, q
        self.log.debug("Generating C_l(x, x')")
        cla0 = corrfunc.corr_to_clarray(corr0, lmax, xa, xromb=2, q=4)
        cla2 = corrfunc.corr_to_clarray(corr2, lmax, xa, xromb=2, q=4)
        cla4 = corrfunc.corr_to_clarray(corr4, lmax, xa, xromb=2, q=4)

        nz = len(self.redshift)

        # Create an extended covariance matrix capturing the cross correlations between
        # the fields
        cla = np.zeros((lmax + 1, 2 * nz, 2 * nz))
        cla[:, :nz, :nz] = cla4
        cla[:, :nz, nz:] = cla2
        cla[:, nz:, :nz] = cla2
        cla[:, nz:, nz:] = cla0

        # TODO: replace with a properly seeded MPI parallel routine
        self.log.debug("Generating realisation of fields")
        sky = skysim.mkfullsky(cla, self.nside)

        f = InitialLSS(
            cosmology=self.cosmology, nside=self.nside, redshift=self.redshift
        )
        f.phi[:] = sky[:nz]
        f.delta[:] = sky[nz:]

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
            biased_field.delta[:] += (D * b1)[:, np.newaxis] * f.delta
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
        za_density(vpsi, biased_field.delta[:], nside, chi, out=final_field.delta[:])
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
        final_field.delta[:] += D[:, np.newaxis] * initial_field.delta[:]

        if self.redshift_space:
            fr = cosmology.growth_rate(za, c)
            vterm = lssutil.diff2(initial_field.phi[:], chi, axis=0)
            vterm *= -(D * fr)[:, np.newaxis]

            final_field.delta[:] += vterm

        return final_field


def za_density(psi, delta, nside, chi, out=None):

    # Extend the chi bins to add another cell at each end
    # This makes it easier to deal with interpolation at the boundaries
    chi_ext = np.zeros(len(chi) + 2, dtype=chi.dtype)
    chi_ext[1:-1] = chi
    chi_ext[0] = chi[0] - (chi[1] - chi[0])
    chi_ext[-1] = chi[-1] + (chi[-1] - chi[-2])

    npix = healpy.pixelfunc.nside2npix(nside)  # In original size maps
    nz = len(chi)

    # Array to hold the density obtained from the ZA approximation
    if out is None:
        out = np.zeros((nz, npix), dtype=delta.dtype)

    angpos = np.array(healpy.pix2ang(nside, np.arange(npix)))

    for ii in range(nz):

        # Initial density
        density_slice = 1 + delta[ii]
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
        chi0 = chi_ext[(chi_ext_ind - 1) % (nz + 2)]
        chi1 = chi_ext[chi_ext_ind % (nz + 2)]
        dchi = chi1 - chi0

        # Calculate the indices and weights in the radial direction
        w0 = np.abs((chi1 - new_chi) / dchi)
        w1 = np.abs((new_chi - chi0) / dchi)
        i0 = chi_ext_ind - 2
        i1 = chi_ext_ind - 1

        w0[np.where((i0 < 0) | (i0 >= nz))] = -1
        w1[np.where((i1 < 0) | (i1 >= nz))] = -1

        radial_ind = np.array([i0, i1])
        radial_weight = np.array([w0, w1])

        _bin_delta(
            density_slice,
            pixel_ind.astype(np.int32),
            pixel_weight,
            radial_ind.astype(np.int32),
            radial_weight,
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