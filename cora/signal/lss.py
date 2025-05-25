from pathlib import Path
from typing import Optional, Tuple

import healpy
import numpy as np

from caput import config, mpiarray, pipeline
from cora.core import skysim
from cora.util import hputil, units
from cora.util.cosmology import Cosmology
from cora.util.pmesh import (
    _bin_delta,
    _pixel_weights,
    _radial_weights,
    calculate_positions,
)
from draco.core import task
from draco.util.random import RandomTask
from draco.core.containers import Map

from ..util.nputil import FloatArrayLike
from . import corrfunc, lssutil, lssmodels
from .lsscontainers import (
    BiasedLSS,
    CorrelationFunction,
    MultiFrequencyAngularPowerSpectrum,
    InitialLSS,
    MatterPowerSpectrum,
    _INTERP_TYPES,
)


# Power spectra that can be used
_POWERSPECTRA = [
    "cora-orig",
    "planck2018_z1.0_halofit-mead-feedback",
    "planck2018_z1.0_halofit-mead",
    "planck2018_z1.0_halofit-original",
    "planck2018_z1.0_halofit-takahashi",
    "planck2018_z1.0_linear",
]


class CalculateCorrelations(task.SingleTask):
    """Calculate the density and potential correlation functions from a power spectrum.

    To reproduce something close to the old cora results, set `powerspectrum` to be
    `cora-orig` and `ksmooth` as 5.

    If a precalculated power spectrum at z>0 is used as input, it will be re-scaled
    to z=0 using the linear growth factor, so the output correlation functions will
    also contain this scaling.

    Attributes
    ----------
    minlogr, maxlogr : float
        The minimum and maximum values of r to calculate (given as log_10(r h /
        Mpc)). Defaults: -1, 5
    switchlogr : float
        The scale below which to directly integrate. Above this an FFT log scheme is
        used (see `corrfunc.ps_to_corr` for details). Default: 1
    samples_per_decade : int
        How many samples per decade to calculate. Default: 1000
    powerspectrum : str
        If a power spectrum object is not passed directly to the setup, then this
        parameter is used to read a precalculated power spectrum. One of: cora-orig,
        planck2018_z1.0_halofit-mead-feedback, planck2018_z1.0_halofit-mead,
        planck2018_z1.0_halofit-original, planck2018_z1.0_halofit-takahashi,
        planck2018_z1.0_linear. The default is planck2018_z1.0_halofit-mead.
    ksmooth : float
        Apply a Gaussian suppression to the power spectrum with sigma=ksmooth. Default
        is `None`, i.e. no Gaussian suppression. To reproduce the old behaviour in cora
        set this to 5.0.
    logkcut_low, logkcut_high : float
        Apply power-law cutoffs at low and high-k to regularise the power spectrum.
        These values give the locations of the cutoffs in log_10(k Mpc / h). By default
        they are at -4 and 1.
    r_interp_type : str
        Default interpolation type for correlation functions. Default: "sinh".
    """

    minlogr = config.Property(proptype=float, default=-1)
    maxlogr = config.Property(proptype=float, default=5)
    switchlogr = config.Property(proptype=float, default=1)
    samples_per_decade = config.Property(proptype=int, default=1000)
    ksmooth = config.Property(proptype=float, default=None)
    logkcut_low = config.Property(proptype=float, default=-4)
    logkcut_high = config.Property(proptype=float, default=4)
    powerspectrum = config.enum(_POWERSPECTRA, default="planck2018_z1.0_halofit-mead")
    r_interp_type = config.enum(_INTERP_TYPES, default="sinh")

    def setup(self, powerspectrum: Optional[MatterPowerSpectrum] = None):

        if powerspectrum is None:
            fpath = Path(__file__).parent / "data" / f"ps_{self.powerspectrum}.h5"
            self.log.info(f"Loading power spectrum file {fpath}")
            powerspectrum = MatterPowerSpectrum.from_file(fpath)

        self._ps = powerspectrum

    def _ps_n(self, n):

        ks = 1e10 if self.ksmooth is None else self.ksmooth

        # Get the power spectrum multiplied by k**-n and a the low and high cutoffs,
        # plus a Gaussian suppression to be compatible with cora.
        # Note that the options for precalculated power spectra are all calculated
        # at z=1, but here they are rescaled to z=0 using the linear growth factor
        # (this occurs internally in self._ps.powerspectrum())
        def _ps(k):
            return (
                lssutil.cutoff(k, self.logkcut_low, 1, 0.5, 6)
                * lssutil.cutoff(k, self.logkcut_high, -1, 0.5, 4)
                * np.exp(-0.5 * (k / ks) ** 2)
                * self._ps.powerspectrum(k, 0.0)
                * k**-n
            )

        return _ps

    def process(self) -> CorrelationFunction:
        """Calculate the correlation functions.

        Returns
        -------
        corr
            An container holding the density correlation function (`corr0`), the
            potential correlation function (`corr4`) and their cross correlation
            (`corr2`).
        """

        # Calculate the correlation functions
        self.log.debug("Generating C_dd(r)")
        k0, c0 = corrfunc.ps_to_corr(
            self._ps_n(0),
            minlogr=self.minlogr,
            maxlogr=self.maxlogr,
            switchlogr=self.switchlogr,
            samples_per_decade=self.samples_per_decade,
            pad_low=4,
            pad_high=6,
            richardson_n=9,
        )
        self.log.debug("Generating C_dp(r)")
        k2, c2 = corrfunc.ps_to_corr(
            self._ps_n(2),
            minlogr=self.minlogr,
            maxlogr=self.maxlogr,
            switchlogr=self.switchlogr,
            samples_per_decade=self.samples_per_decade,
            pad_low=4,
            pad_high=6,
            richardson_n=9,
        )
        self.log.debug("Generating C_pp(r)")
        k4, c4 = corrfunc.ps_to_corr(
            self._ps_n(4),
            minlogr=self.minlogr,
            maxlogr=self.maxlogr,
            switchlogr=self.switchlogr,
            samples_per_decade=self.samples_per_decade,
            pad_low=4,
            pad_high=6,
            richardson_n=9,
        )

        func = CorrelationFunction(attrs_from=self._ps)

        func.add_function("corr0", k0, c0, type=self.r_interp_type, x_t=k0[1], f_t=1e-3)
        func.add_function("corr2", k2, c2, type=self.r_interp_type, x_t=k2[1], f_t=1e-6)
        func.add_function("corr4", k4, c4, type=self.r_interp_type, x_t=k4[1], f_t=1e2)

        self.done = True

        return func


class BlendNonLinearPowerSpectrum(task.SingleTask):
    """Generate a controllable mix between a linear and non-linear power spectrum.

    Blends a linear and non-linear power spectrum by linear interpolating in
    :math:`\log(P(k))`.

    Attributes
    ----------
    alpha_NL : float
        Controls the mix between the linear and non-linear PS. 0 is purely linear, 1
        (default) gives the non-linear power spectrum.
    powerspectrum_linear : str
        The name of the linear power spectrum.
    powerspectrum_nonlinear : str
        The name of the non-linear power spectrum.
    """

    alpha_NL = config.Property(proptype=float, default=1.0)
    powerspectrum_linear = config.enum(_POWERSPECTRA, default="planck2018_z1.0_linear")
    powerspectrum_nonlinear = config.enum(
        _POWERSPECTRA, default="planck2018_z1.0_halofit-mead"
    )

    def process(self) -> MatterPowerSpectrum:
        """Construct the blended power spectrum.

        Returns
        -------
        blended_ps
            The blended power spectrum.
        """

        basepath = Path(__file__).parent / "data"
        ps_linear = MatterPowerSpectrum.from_file(
            basepath / f"ps_{self.powerspectrum_linear}.h5"
        )
        ps_nonlinear = MatterPowerSpectrum.from_file(
            basepath / f"ps_{self.powerspectrum_nonlinear}.h5"
        )

        if ps_linear._ps_redshift != ps_nonlinear._ps_redshift:
            raise RuntimeError(
                "Linear and non-linear PS do not have matching redshifts."
            )

        if not np.array_equal(
            ps_linear.index_map["x_powerspectrum"][:],
            ps_nonlinear.index_map["x_powerspectrum"][:],
        ):
            raise RuntimeError("Linear and non-linear PS do not have matching k axes.")

        psl = ps_linear.datasets["powerspectrum"][:]
        psnl = ps_nonlinear.datasets["powerspectrum"][:]

        ps_linear.datasets["powerspectrum"][:] = (
            psl ** (1 - self.alpha_NL) * psnl**self.alpha_NL
        )
        ps_linear.attrs["tag"] = f"psblend_alphaNL_{self.alpha_NL}"

        self.done = True

        return ps_linear


class CalculateMultiFrequencyAngularPowerSpectrum(task.SingleTask):
    """Calculate C_l(chi,chi') from a real-space correlation function.

    The output will be evaluated at constant redshift, corresponding
    to the redshift of the input correlation functions.

    Attributes
    ----------
    nside : int
        The Healpix resolution corresponding to the desired ell_max.
    redshift : np.ndarray
        The redshifts to evaluate at.
    frequencies : np.ndarray
        The frequencies to evaulate at. Overrides redshift property if
        specified.
    xromb : int, optional
        Gauss-Legendre quadrature order for integrating C_ell over radial
        bins. (Used Romberg integration in a previous version, hence the
        name xromb.) xromb=0 turns off this integral. When dealing with
        nonlinear matter power spectrum, for sub-percent accuracy up to
        l ~ 1500, xromb = 3 is recommended. Default: 2.
    leg_q : int, optional
        Integration accuracy parameter for Legendre transform for C_ell
        computation. When dealing with nonlinear matter power spectrum,
        for sub-percent accuracy up to l ~ 1500, leg_q = 16 is recommended.
        Default: 4.
    leg_chunksize: int, optional
        Chunk size for evaluating samples of C_ell integrand. Changing
        from default value is unlikely to affect performance. Default: 50.
    corrfunc_interp_type: str, optional
        Interpolation method to use for correlation function in C_ell
        integrand. If None, the default method stored in the
        CorrelationFunction container is used. Default: None.
    """

    nside = config.Property(proptype=int)
    redshift = config.Property(proptype=lssutil.linspace, default=None)
    frequencies = config.Property(proptype=lssutil.linspace, default=None)
    xromb = config.Property(proptype=int, default=2)
    leg_q = config.Property(proptype=int, default=4)
    leg_chunksize = config.Property(proptype=int, default=50)
    corrfunc_interp_type = config.enum(_INTERP_TYPES, default=None)

    def process(
        self, correlation_functions: CorrelationFunction
    ) -> MultiFrequencyAngularPowerSpectrum:
        """Compute the angular power spectra.

        Returns
        -------
        out_cont
            Container with the output angular power spectra.
        """

        if self.redshift is None and self.frequencies is None:
            raise RuntimeError("Redshifts or frequencies must be specified!")

        cosmology = correlation_functions.cosmology

        corr0 = correlation_functions.get_function(
            "corr0", interp_type=self.corrfunc_interp_type
        )
        corr2 = correlation_functions.get_function(
            "corr2", interp_type=self.corrfunc_interp_type
        )
        corr4 = correlation_functions.get_function(
            "corr4", interp_type=self.corrfunc_interp_type
        )

        if self.frequencies is None:
            redshift = self.redshift
        else:
            redshift = units.nu21 / self.frequencies - 1.0

        xa = cosmology.comoving_distance(redshift)

        # NOTE: it is important not to set this any higher. Otherwise,
        # power will alias back down when the map is transformed later on
        lmax = 3 * self.nside - 1

        self.log.debug("Generating C_l(x, x') for delta-delta")
        cla0 = corrfunc.corr_to_clarray(
            corr0,
            lmax,
            xa,
            xromb=self.xromb,
            q=self.leg_q,
            chunksize=self.leg_chunksize,
        )

        self.log.debug("Generating C_l(x, x') for phi-delta")
        cla2 = corrfunc.corr_to_clarray(
            corr2,
            lmax,
            xa,
            xromb=self.xromb,
            q=self.leg_q,
            chunksize=self.leg_chunksize,
        )

        self.log.debug("Generating C_l(x, x') for phi-phi")
        cla4 = corrfunc.corr_to_clarray(
            corr4,
            lmax,
            xa,
            xromb=self.xromb,
            q=self.leg_q,
            chunksize=self.leg_chunksize,
        )

        if self.frequencies is not None:
            out_cont = MultiFrequencyAngularPowerSpectrum(
                cosmology=cosmology,
                freq=self.frequencies,
                lmax=lmax,
            )
        else:
            out_cont = MultiFrequencyAngularPowerSpectrum(
                cosmology=cosmology,
                redshift=redshift,
                lmax=lmax,
            )

        out_cont.Cl_delta_delta[:] = cla0
        out_cont.Cl_phi_delta[:] = cla2
        out_cont.Cl_phi_phi[:] = cla4

        return out_cont


class GenerateInitialLSSFromCl(task.SingleTask):
    """Generate initial LSS maps from input angular power spectrum.

    Attributes
    ----------
    nside : int, optional
        The Healpix resolution to use. Must not be higher than the nside
        corresponding to lmax of the input angular power spectrum. If not
        specified, will be determined from the input spectrum. Default: None.
    num : int
        The number of simulations to generate. Default: 1.
    start_seed : int
        The random seed to use for generating the first simulation.
        Sim i (indexed from 0) is generated using start_seed+i.
        Default: 0.
    """

    nside = config.Property(proptype=int, default=None)
    num_sims = config.Property(proptype=int, default=1)
    start_seed = config.Property(proptype=int, default=0)

    def setup(self, aps: MultiFrequencyAngularPowerSpectrum):
        """Set up the task.

        Parameters
        ----------
        aps
            The pre-calculated multi-frequency angular power spectrum.
        """
        self.aps = aps
        self.cosmology = aps.cosmology

        self.seed = self.start_seed

        nside_from_cl = hputil.nside_for_lmax(len(aps.ell) - 1, accuracy_boost=0)

        if self.nside is None:
            self.nside = nside_from_cl
            self.log.info(f"Set nside={self.nside} from input C_l container")
        else:
            if self.nside > nside_from_cl:
                raise RuntimeError(
                    f"Requested nside ({self.nside}) cannot exceed nside used for "
                    f"input C_l ({nside_from_cl})"
                )

        aps.redistribute("ell")

    def process(self) -> InitialLSS:
        """Generate a realisation of the LSS initial conditions.

        Returns
        -------
        f
            The LSS initial conditions.
        """
        # Stop if we've already generated enough realizations
        if self.num_sims == 0:
            raise pipeline.PipelineStopIteration()
        self.num_sims -= 1

        nz = len(self.aps.chi)

        # Create extended covariance matrix capturing cross-correlations
        # between the phi and delta fields
        cla = mpiarray.zeros((len(self.aps.ell), 2 * nz, 2 * nz), axis=0)
        cla[:, nz:, nz:] = self.aps.Cl_delta_delta[:]
        cla[:, :nz, nz:] = self.aps.Cl_phi_delta[:]
        cla[:, nz:, :nz] = self.aps.Cl_phi_delta[:]
        cla[:, :nz, :nz] = self.aps.Cl_phi_phi[:]

        # Generate map
        self.log.info(f"Generating realisation of fields using seed {self.seed}")
        rng = np.random.default_rng(self.seed)
        sky = skysim.mkfullsky(cla, self.nside, rng=rng)

        # Make container for output
        if self.aps.freq is not None:
            f = InitialLSS(
                cosmology=self.cosmology,
                nside=self.nside,
                freq=self.aps.freq,
            )
        else:
            f = InitialLSS(
                cosmology=self.cosmology,
                nside=self.nside,
                redshift=self.aps.redshift,
            )

        # Redistribute over the pixel axis to properly
        # index the sky nz axis
        f.redistribute("pixel")
        sky = sky.redistribute(axis=1)

        f.phi[:] = sky[:nz]
        f.delta[:] = sky[nz:]

        f.redistribute("chi")

        self.seed += 1

        return f


class GenerateInitialLSS(
    CalculateMultiFrequencyAngularPowerSpectrum,
    GenerateInitialLSSFromCl,
):
    """Generate initial LSS maps from a correlation function.

    This reproduces the behavior of the previous GenerateInitialLSS
    task, for legacy compatibility.
    """

    def setup(self, correlation_functions: CorrelationFunction):
        aps = CalculateMultiFrequencyAngularPowerSpectrum.process(
            self, correlation_functions
        )
        GenerateInitialLSSFromCl.setup(self, aps)

    def process(self):
        return GenerateInitialLSSFromCl.process(self)


class GenerateBiasedFieldBase(task.SingleTask):
    r"""Take the initial field and generate a biased field.

    Note that this applies a bias in Lagrangian space.

    .. math::
        \delta_B = D(z) b_1(z) \delta_L
        + D(z)^2 b_2(z) (\delta_L^2 - \langle \delta_L^2 \rangle)

    Attributes
    ----------
    lightcone : bool, optional
        Whether to generate the field on the lightcone, by re-scaling each
        map by the linear growth factor. If not, all slices will be at a
        fixed epoch specified by the `redshift` attribute. Default: True.
    redshift : float, optional
        The redshift of the field (if not on the `lightcone`). Default: None.
    lognormal : bool, optional
        Whether to apply a lognormal transform to the field. Default: False.
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

        # The input field has previously been scaled to z=0, so to re-scale to
        # another redshift, we need to multiply by D(z) / D(z=0)
        z = f.redshift if self.lightcone else self.redshift * np.ones_like(f.chi)
        D = f.cosmology.growth_factor(z) / f.cosmology.growth_factor(0)

        f.redistribute("chi")
        fd = f.delta[:].local_array

        # Apply any first order bias
        try:
            b1 = self._bias_1(z)

            # Local offset and shape along chi axis
            lsel = f.delta[:].local_bounds

            biased_field.delta[:].local_array[:] += (D * b1)[lsel, np.newaxis] * fd
        except NotImplementedError:
            self.log.info("First order bias is not implemented. This is a bit odd.")

        # Apply any second order bias
        try:
            b2 = self._bias_2(z)
            d2m = (fd**2).mean(axis=1)[:, np.newaxis]
            biased_field.delta[:].local_array[:] += (D**2 * b2)[:, np.newaxis] * (
                fd**2 - d2m
            )
        except NotImplementedError:
            self.log.debug("No second order bias to apply.")

        if self.lognormal:
            lssutil.lognormal_transform(
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
    """Generate a field with a constant linear Lagrangian bias.

    Attributes
    ----------
    bias : float
        The *Lagrangian* bias. At first order this is the related to the Eulerian bias
        by :math:`b_L = b_E - 1`.
    """

    bias_L = config.Property(proptype=float, default=0.0)

    def _bias_1(self, z: FloatArrayLike) -> FloatArrayLike:
        return np.ones_like(z) * self.bias_L


class GeneratePolynomialBias(GenerateBiasedFieldBase):
    r"""Generate a field with a polynomial bias model.

    .. math::
        b_1(z) = \sum_{n=0}^N c_n (z - z_{eff})^n

    Attributes
    ----------
    z_eff : float
        The effective redshift of the bias model for the field.
    bias_coeff : list
        The *Lagrangian* bias coefficients. At first order this is the related to the
        Eulerian bias by subtracting one from the Eulerian bias 0th coefficient.
    model : str, optional
        Pick a predetermined set of coefficients and z_eff. This is lower priority than
        explicitly setting the parameters above. See `lssmodels.bias` for available
        models and details.
    alpha_b : float
        Apply a scaling that increase the *Eulerian* bias by a factor `alpha_b`.
        Although that this scales the Eulerian bias may be confusing, this is useful to
        generating combinations that isolate certain terms in a cross power spectrum.
    """

    z_eff = config.Property(proptype=float, default=None)
    bias_coeff = config.list_type(type_=float, default=None)
    model = config.enum(lssmodels.bias.models(), default=None)
    alpha_b = config.Property(proptype=float, default=1.0)

    def setup(self):
        """Verify the config parameters."""

        if self.z_eff is not None and self.bias_coeff is not None:

            def b(z):
                return lssmodels.PolyModelSet.evaluate_poly(
                    z, self.z_eff, self.bias_coeff
                )

            self._bias = b

        elif self.model is not None:
            self._bias = lssmodels.bias[self.model]
        else:
            raise config.CaputConfigError(
                "Either `model` must be set, or `z_eff` and `bias_coeff`"
            )

    def _bias_1(self, z: FloatArrayLike) -> FloatArrayLike:

        # Evaluate the polynomial bias model to get the default bias
        bias = self._bias(z)

        # Modify the bias to account for the scaling coefficient. This is a no-op if
        # alpha=1 (default)
        bias = self.alpha_b * bias + self.alpha_b - 1.0

        return bias


class DynamicsBase(task.SingleTask):
    """Base class for generating final fields from biased fields.

    Attributes
    ----------
    redshift_space : bool, optional
        Generate the final field in redshift space or not. Default: True.
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
    """Generate a simulated LSS field using Zel'dovich dynamics.

    Attributes
    ----------
    sph : bool
        Use a Smoothed Particle Hydrodynamics (SPH) type scheme for computing the final
        field after applying the displacements. This treats the mass as compressible
        clouds which change size with the local density. See `za_density_sph` for
        details.
    """

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
        initial_field.redistribute("chi")
        biased_field.redistribute("chi")

        lsel = initial_field.phi[:].local_bounds

        self._validate_fields(initial_field, biased_field)
        c, nside, _, chi, za = self._get_props(biased_field)

        # Take the gradient of the Lagrangian potential to get the
        # displacements. Need the full array on each rank because
        # we want the gradient along both axes. This is a bit of an
        # awkward way to do this, but we have to do it this way to
        # avoid ballooning the memory
        vpsi = lssutil.gradient(
            initial_field.phi[:].local_array, chi[lsel], grad0=False
        )
        initial_field.redistribute("pixel")
        vpsi[0] = np.gradient(initial_field.phi[:].local_array, chi, axis=0)
        initial_field.redistribute("chi")

        # Apply growth factor:
        D = c.growth_factor(za) / c.growth_factor(0)
        vpsi *= D[np.newaxis, :, np.newaxis]

        theta, _ = hputil.ang_positions(nside).T

        # Divide by comoving distance to get angular displacements
        vpsi[1:3] /= chi[np.newaxis, :, np.newaxis]

        # Divide by sin(theta) to get phi displacements, not angular displacements
        vpsi[2] /= np.sin(theta[np.newaxis, :])

        # Add the extra radial displacement to get into redshift space
        if self.redshift_space:
            fr = c.growth_rate(za)
            vpsi[0] *= (1 + fr)[:, np.newaxis]

        # Create the final field container
        final_field = BiasedLSS(axes_from=biased_field, attrs_from=biased_field)
        final_field.delta[:] = 0.0

        # Calculate the Zel'dovich displaced field and assign directly into the output
        # container
        delta_m = initial_field.delta[:].local_array * D[lsel, np.newaxis]
        delta_bias = biased_field.delta[:].local_array
        fdelta = final_field.delta[:].local_array

        if self.sph:
            # Calculate sigma_chi for the full chi range, rather than
            # just the local selection
            sigma_chi = np.mean(abs(np.diff(chi))) / 2

            za_density_sph(
                vpsi[:, lsel],
                delta_bias,
                delta_m,
                chi[lsel],
                fdelta,
                sigma_chi=sigma_chi,
            )
        else:
            za_density_grid(vpsi[:, lsel], delta_bias, delta_m, chi[lsel], fdelta)

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

        # Distribute over the pixel axis for derivatives
        initial_field.redistribute("pixel")
        biased_field.redistribute("pixel")

        # Create the final field container
        final_field = BiasedLSS(axes_from=biased_field, attrs_from=biased_field)
        final_field.redistribute("pixel")

        fdelta = final_field.delta[:].local_array
        idelta = initial_field.delta[:].local_array
        iphi = initial_field.phi[:].local_array

        # The input fields have previously been scaled to z=0, so to re-scale to
        # another redshift, we need to multiply by D(z) / D(z=0)
        D = c.growth_factor(za) / c.growth_factor(0)

        # Copy over biased field
        fdelta[:] = biased_field.delta[:].local_array

        # Input biased field will have been multiplied by Lagrangian bias, which
        # is equal to Eulerian bias minus one, so we add the growth-factor-scaled
        # "initial" delta to obtain the Eulerian-bias-scaled field
        fdelta[:] += D[:, np.newaxis] * idelta

        if self.redshift_space:
            fr = c.growth_rate(za)
            vterm = lssutil.diff2(iphi, chi[:], axis=0)
            vterm *= -(D * fr)[:, np.newaxis]

            fdelta[:] += vterm

        final_field.redistribute("chi")

        return final_field


class BiasedLSSToMap(task.SingleTask):
    """Convert a BiasedLSS object into a Map object.

    Attributes
    ----------
    use_mean_21cmT : bool
        Multiply map by mean 21cm temperature as a function of z (default: False).
    use_prefactor : float
        Scale by an arbitrary prefactor.
    lognormal : bool
        Use a lognormal transform to guarantee that the density is always positive (i.e.
        delta > -1).
    omega_HI_model : str
        The model to use for Omega_HI when applying the mean 21cm temperature. See
        `lssmodels.omega_HI` for the list of valid models and details. Default is
        `Crighton2015`.
    """

    use_mean_21cmT = config.Property(proptype=int, default=False)
    map_prefactor = config.Property(proptype=float, default=1.0)
    lognormal = config.Property(proptype=bool, default=False)
    omega_HI_model = config.enum(lssmodels.omega_HI.models(), default="Crighton2015")

    def process(self, biased_lss: BiasedLSS) -> Map:
        """Generate a realisation of the LSS initial conditions.

        Parameters
        ----------
        biased_lss
            Input BiasedLSS container.

        Returns
        -------
        out_map
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

        if self.lognormal:
            lssutil.lognormal_transform(biased_lss.delta[:], out=m.map[:, 0], axis=1)
        else:
            m.map[:, 0, :] = biased_lss.delta[:, :]

        # Multiply map by specific prefactor, if desired
        if self.map_prefactor != 1:
            self.log.info(f"Multiplying map by {self.map_prefactor}")
            m.map[:] *= self.map_prefactor

        # If desired, multiply by Tb(z)
        if self.use_mean_21cmT:

            z = biased_lss.redshift
            omHI = lssmodels.omega_HI.evaluate(z, model=self.omega_HI_model)
            T_b = lssmodels.mean_21cm_temperature(biased_lss.cosmology, z, omHI)

            m.map[:].local_array[:, 0] *= T_b[m.map[:].local_bounds, np.newaxis]

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
    lssutil.assert_shape(psi, (3, nchi, npix), "psi")
    lssutil.assert_shape(delta_m, (nchi, npix), "delta_m")
    lssutil.assert_shape(chi, (nchi,), "chi")
    lssutil.assert_shape(out, (nchi, npix), "out")

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

        # Find the enclosing chi bins
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


class FingersOfGod(task.SingleTask):
    r"""Apply a smoothing to approximate the effects of Fingers of God.

    This applies an exponential smoothing kernel which is equivalent to a *squared
    Lorentzian* suppression in power spectrum space which is a reasonably common model.

    Attributes
    ----------
    alpha_FoG : float
        A parameter to control the strength of the effect by adjusting the smoothing
        scale. A value of 1 (default) applies the nominal smoothing, 0 turns the
        smoothing off entirely, and any other values adjust the scale appropriately.
    model : str, optional
        Use an inbuilt model for the Fingers of God smoothing. If `None` (default), a
        specific model is expected to be set via `FoG_coeff` and `z_eff`. See
        `lssmodels.sigma_P` for available models and details about them.
    FoG_coeff : list
        A list of coefficient in a polynomial of `FoG_coeff[i] * (z - z_eff)**i`. If
        `None` default, a `model` must be set.
    z_eff : float
        The effective redshift of the polynomial expansion.
    """

    model = config.enum(lssmodels.sigma_P.models(), default=None)

    alpha_FoG = config.Property(proptype=float, default=1.0)

    FoG_coeff = config.list_type(type_=float, default=None)
    z_eff = config.Property(proptype=float, default=None)

    def setup(self):
        """Verify the config parameters."""

        if self.z_eff is not None and self.FoG_coeff is not None:

            def s(z):
                return lssmodels.PolyModelSet.evaluate_poly(
                    z, self.z_eff, self.FoG_coeff
                )

            self._sigma_P = s

        elif self.model is not None:
            self._sigma_P = lssmodels.sigma_P[self.model]
        else:
            raise config.CaputConfigError(
                "Either `model` must be set, or `z_eff` and `FoG_coeff`"
            )

    def process(self, field: BiasedLSS) -> BiasedLSS:
        """Apply the FoG effect.

        Parameters
        ----------
        field
            The input field.

        Returns
        -------
        smoothed_field
            The field with the smoothing applied.
        """

        # Just return the input if FoGs are effectively turned off
        if self.alpha_FoG == 0.0:
            return field

        # Get the growth factor and smoothing scales
        D = field.cosmology.growth_factor(field.redshift)
        sigmaP = self._sigma_P(field.redshift)

        K = lssutil.exponential_FoG_kernel(field.chi, self.alpha_FoG * sigmaP, D)

        smoothed_field = BiasedLSS(axes_from=field, attrs_from=field)
        # Distribute over pixels to apply the smoothing kernel
        smoothed_field.redistribute("pixel")
        field.redistribute("pixel")

        # Apply the smoothing kernel
        np.matmul(
            K, field.delta[:].local_array, out=smoothed_field.delta[:].local_array
        )

        # distribute back to the chi axis
        smoothed_field.redistribute("chi")

        return smoothed_field


class AddCorrelatedShotNoise(RandomTask, task.SingleTask):
    """Add a correlated shot noise contribution to each input map.

    The shot noise realisation is deterministically generated from the LSS field, so all
    tasks receiving the same `InitialLSS` object as a requirement will generate the same
    shot noise realisation.

    Attributes
    ----------
    n_eff : float
        The effective number density in sources per (Mpc / h)^3.
    log_M_HI_g : float
        Set the shot noise level by specifying the log10 of the average HI mass per
        tracer galaxy in solar mass units. This is overriden if `n_eff` is set
        explicitly.
    omega_HI_model : str
        The model to use for Omega_HI when applying the mean 21cm temperature. See
        `lssmodels.omega_HI` for the list of valid models and details. Default is
        `Crighton2015`.
    """

    n_eff = config.Property(proptype=float, default=None)
    log_M_HI_g = config.Property(proptype=float, default=None)
    omega_HI_model = config.enum(lssmodels.omega_HI.models(), default="Crighton2015")

    def setup(self, lss: InitialLSS):
        """Set up a common seed from the LSS field.

        Parameters
        ----------
        lss
            Common InitialLSS field to use for seeding.
        """
        import zlib

        # Get a subset of the data in bytes
        lss_subset = lss.delta[:].local_array[:, :100].copy().tobytes()

        # If the seed was not set, hash the input LSS and use that
        if self.seed is None:
            self.seed = zlib.adler32(lss_subset)

        if self.n_eff is not None:
            self._n_eff_z = np.ones_like(lss.chi) * self.n_eff
        elif self.log_M_HI_g is not None:
            self._n_eff_z = lssmodels.log_M_HI_g_to_n_eff(
                self.log_M_HI_g, lss.cosmology, lss.redshift, self.omega_HI_model
            )
        else:
            raise RuntimeError("One of `n_eff` or `log_M_HI_g` must be set.")

    def process(self, input_field: BiasedLSS) -> BiasedLSS:
        """Add correlated shot noise to the input field.

        Paramters
        ---------
        input_field
            The incoming field to add shot noise to. Modified in-place.

        Returns
        -------
        output_field
            The field with shot noise. This is the same object as `input_field`.
        """
        pixarea = healpy.nside2pixarea(input_field.nside)
        ichi = input_field.chi[input_field.delta[:].local_bounds]

        volume = pixarea * (ichi**2) * lssutil.calculate_width(ichi)

        std = (volume * self._n_eff_z) ** -0.5
        shot_noise = self.rng.normal(
            scale=std[:, np.newaxis], size=input_field.delta.shape
        )
        input_field.delta[:].local_array[:] += shot_noise

        return input_field


def za_density_sph(
    psi: np.ndarray,
    delta_bias: np.ndarray,
    delta_m: np.ndarray,
    chi: np.ndarray,
    out: np.ndarray,
    sigma_chi: float = None,
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
    sigma_chi
        Smoothing scale at mean density
    """
    nchi, npix = delta_bias.shape
    nside = healpy.npix2nside(npix)

    # Validate all the array shapes
    lssutil.assert_shape(psi, (3, nchi, npix), "psi")
    lssutil.assert_shape(delta_m, (nchi, npix), "delta_m")
    lssutil.assert_shape(chi, (nchi,), "chi")
    lssutil.assert_shape(out, (nchi, npix), "out")

    # Set the nominal smoothing scales at mean density
    if sigma_chi is None:
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
