# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import numpy as np

from cora.core import maps, skysim
from cora.util import cubicspline as cs
from cora.util import units, nputil, halomodel
from cora.signal import corr


class CorrBiasedTracer(corr.RedshiftCorrelation, maps.Sky3d):
    r"""Correlation function of HI brightness temperature fluctuations.

    Incorporates reasonable approximations for the growth factor and
    growth rate.

    """

    add_mean = False

    _kstar = 5.0

    def __init__(self, ps=None, redshift=0.0, sigma_v=0.0, 
                 tracer_type='none', bias=1.0, lognorm=False, 
                 no_rsd=False, **kwargs):

        from os.path import join, dirname

        if ps is None:

            psfile = join(dirname(__file__), "data/ps_z1.5.dat")
            redshift = 1.5

            c1 = cs.LogInterpolater.fromfile(psfile)
            ps = lambda k: np.exp(-0.5 * k ** 2 / self._kstar ** 2) * c1(k)

        elif tracer_type == '21cm':
            # Compute the non-smoothed power spectrum from the one given.
            # Needed for the halo model.
            c1 = lambda k: ps(k) / np.exp(-0.5 * k ** 2 / self._kstar ** 2)

        self._sigma_v = sigma_v

        corr.RedshiftCorrelation.__init__(self, ps_vv=ps, redshift=redshift)
        self._load_cache(join(dirname(__file__), "data/corr_z1.5.dat"))
        # self.load_fft_cache(join(dirname(__file__),"data/fftcache.npz"))

        self.tracer_type = tracer_type  # Tracer type
        self.no_rsd = no_rsd
        self.lognorm = lognorm
        self.bias = bias
        if self.tracer_type == '21cm':  # Use Halo Model to compute bias.
            # Pass the full (not smoothed) Power Spectrum to the halo model
            # HaloModel uses powerspectrum normalized to z=0
            def hmps(k):
                return c1(k) * (self.growth_factor(0.)
                                / self.growth_factor(self.ps_redshift))**2

            # Initialize HaloModel
            self.hm = halomodel.HaloModel(gf=self.growth_factor, ps=hmps)
            self.hm.check_update()
            

    def T_b(self, z):
        r"""Mean 21cm brightness temperature at a given redshift.

        Temperature is in K.

        Parameters
        ----------
        z : array_like
            Redshift to calculate at.

        Returns
        -------
        T_b : array_like

        Notes: the prefactor used to be 0.3 mK, but Tzu-Ching pointed out that this
        was from and error in 2008PhRvL.100i1303C, Eric recalculated this to be
        0.39 mK (agrees with 0.4 mK quoted over phone from Tzu-Ching)
        """

        return (
            3.9e-4
            * ((self.cosmology.omega_m + self.cosmology.omega_l * (1 + z) ** -3) / 0.29)
            ** -0.5
            * ((1.0 + z) / 2.5) ** 0.5
            * (self.omega_HI(z) / 1e-3)
        )

    def mean(self, z):
        if self.add_mean:
            return self.T_b(z)
        else:
            return np.zeros_like(z)

    def omega_HI(self, z):
        """Neutral hydrogen fraction.

        Parameters
        ----------
        z : array_like
            Redshift to calculate at.

        Returns
        -------
        omega_HI : array_like


        Notes
        -----
        Best fit value from http://arxiv.org/pdf/1304.3712.pdf
        """
        return 6.2e-4

    def x_h(self, z):
        r"""Neutral hydrogen fraction at a given redshift.

        Just returns a constant at the moment. Need to add a more
        sophisticated model.

        Parameters
        ----------
        z : array_like
            Redshift to calculate at.

        Returns
        -------
        x_e : array_like
        """
        return 1e-3

    def prefactor(self, z):
        r"""The Brightness prefactors to be applied to C_l's.
        """
        if self.tracer_type == '21cm':
            return self.T_b(z)
        else:
            return np.ones_like(z)

    def growth_factor(self, z):
        r"""Approximation for the matter growth factor.

        Uses a Pade approximation.

        Parameters
        ----------
        z : array_like
            Redshift to calculate at.

        Returns
        -------
        growth_factor : array_like

        Notes
        -----
        See _[1].

        .. [1] http://arxiv.org/abs/1012.2671
        """

        x = ((1.0 / self.cosmology.omega_m) - 1.0) / (1.0 + z) ** 3

        num = 1.0 + 1.175 * x + 0.3064 * x ** 2 + 0.005355 * x ** 3
        den = 1.0 + 1.857 * x + 1.021 * x ** 2 + 0.1530 * x ** 3

        d = (1.0 + x) ** 0.5 / (1.0 + z) * num / den

        return d

    def growth_rate(self, z):
        r"""Approximation for the matter growth rate.

        From explicit differentiation of the Pade approximation for
        the growth factor.

        Parameters
        ----------
        z : array_like
            Redshift to calculate at.

        Returns
        -------
        growth_factor : array_like

        Notes
        -----
        See _[1].

        .. [1] http://arxiv.org/abs/1012.2671
        """

        x = ((1.0 / self.cosmology.omega_m) - 1.0) / (1.0 + z) ** 3

        dnum = 3.0 * x * (1.175 + 0.6127 * x + 0.01607 * x ** 2)
        dden = 3.0 * x * (1.857 + 2.042 * x + 0.4590 * x ** 2)

        num = 1.0 + 1.175 * x + 0.3064 * x ** 2 + 0.005355 * x ** 3
        den = 1.0 + 1.857 * x + 1.021 * x ** 2 + 0.1530 * x ** 3

        f = 1.0 + 1.5 * x / (1.0 + x) + dnum / num - dden / den

        return f

    def bias_z(self, z):
        r"""Use the halo model to determine the bias.
        Otherwise use ones. """

        if self.tracer_type == '21cm':
            bias = []
            z_shape = z.shape
            z = z.flatten()
            for zz in z:
                self.hm.redshift = zz
                bias.append(self.hm.lbias_h1() + 1)
            return np.array(bias).reshape(z_shape)

        if tracer_type == 'qso':
            # Quasars. Bias taken from arXiv:1705.04718
            alpha, beta = 0.278, 2.393
            return alpha*((1 + z)**2 - 6.565) + beta

        else:
            return np.ones_like(z) * self.bias

    # Override angular_power spectrum to switch to allow using frequency
    def angular_powerspectrum(self, l, nu1, nu2, redshift=False):
        """Calculate the angular powerspectrum.

        Parameters
        ----------
        l : np.ndarray
            Multipoles to calculate at.
        nu1, nu2 : np.ndarray
            Frequencies/redshifts to calculate at.
        redshift : boolean, optional
            If `False` (default) interperet `nu1`, `nu2` as frequencies, 
            otherwise they are redshifts (relative to the 21cm line).

        Returns
        -------
        aps : np.ndarray
        """

        if not redshift:
            z1 = units.nu21 / nu1 - 1.0
            z2 = units.nu21 / nu2 - 1.0
        else:
            z1 = nu1
            z2 = nu2

        return corr.RedshiftCorrelation.angular_powerspectrum(self, l, z1, z2)

    # Wrap angular_powerspectrum to switch to allow using frequency
    # and to remove bias, prefactors, evolution and RSD.
    def raw_angular_powerspectrum(self, l, nu1, nu2, redshift=False):
        """Calculate the angular powerspectrum of the matter over-density
        without time evolution, RSD or 21cm normalization.

        Parameters
        ----------
        l : np.ndarray
            Multipoles to calculate at.
        nu1, nu2 : np.ndarray
            Frequencies/redshifts to calculate at.
        redshift : boolean, optional
            If `False` (default) interperet `nu1`, `nu2` as frequencies,
            otherwise they are redshifts (relative to the 21cm line).

        Returns
        -------
        aps : np.ndarray
        """

        if not redshift:
            z1 = units.nu21 / nu1 - 1.0
            z2 = units.nu21 / nu2 - 1.0
        else:
            z1 = nu1
            z2 = nu2

        return corr.RedshiftCorrelation.angular_powerspectrum(
                            self, l, z1, z2,
                            include_prefactor=False, 
                            include_bias=False, include_rsd=False, 
                            include_evolution=False)

    # Override angular_power spectrum to switch to allow using frequency
    def angular_powerspectrum_full(self, l, nu1, nu2, redshift=False):
        """Calculate the angular powerspectrum by explicit integration.

        Parameters
        ----------
        l : np.ndarray
            Multipoles to calculate at.
        nu1, nu2 : np.ndarray
            Frequencies/redshifts to calculate at.
        redshift : boolean, optional
            If `False` (default) interperet `nu1`, `nu2` as frequencies, 
            otherwise they are redshifts (relative to the 21cm line).

        Returns
        -------
        aps : np.ndarray
        """

        if not redshift:
            z1 = units.nu21 / nu1 - 1.0
            z2 = units.nu21 / nu2 - 1.0
        else:
            z1 = nu1
            z2 = nu2

        return corr.RedshiftCorrelation.angular_powerspectrum_full(self, l, z1, z2)

    # Overwrite getsky() to account for specific options.
    def getsky(self):
        """Create a map of the unpolarised sky.
        """

        if (self.lognorm or self.no_rsd):

            lmax = 3 * self.nside - 1
            cla = skysim.clarray(
                self.raw_angular_powerspectrum, lmax, 
                self.frequencies, zromb=self.oversample
            )
            zero_mean_map = skysim.mkfullsky(cla, self.nside)

            if self.lognorm:
                # Log-normalization:
                zero_mean_map = np.exp(zero_mean_map) - 1.0

            # Apply prefactor, evolution and bias since 
            # they were ommited in the C_l's.
            z_pixels = units.nu21 / self.frequencies - 1.0
            zero_mean_map *= self.prefactor(z_pixels)[:, np.newaxis]
            zero_mean_map *= self.bias_z(z_pixels)[:, np.newaxis]
            zero_mean_map *= self.growth_factor(z_pixels)[:, np.newaxis]

            return self.mean_nu(self.frequencies)[:, np.newaxis] + zero_mean_map

        else:
            return super(CorrBiasedTracer, self).getsky()

    def mean_nu(self, freq):

        return self.mean(units.nu21 / freq - 1.0)

    def getfield(self):
        r"""Fetch a realisation of the 21cm signal.
        """
        z1 = units.nu21 / self.nu_upper - 1.0
        z2 = units.nu21 / self.nu_lower - 1.0

        cube = self.realisation(
            z1,
            z2,
            self.x_width,
            self.y_width,
            self.nu_num,
            self.x_num,
            self.y_num,
            zspace=False,
        )[::-1, :, :].copy()

        return cube

    def get_kiyo_field(self, refinement=1):
        r"""Fetch a realisation of the 21cm signal (NOTE: in K)
        """
        z1 = units.nu21 / self.nu_upper - 1.0
        z2 = units.nu21 / self.nu_lower - 1.0

        cube = self.realisation(
            z1,
            z2,
            self.x_width,
            self.y_width,
            self.nu_num,
            self.x_num,
            self.y_num,
            refinement=refinement,
            zspace=False,
        )

        return cube

    def get_pwrspec(self, k_vec):
        r"""Fetch the power spectrum of the signal
        The effective redshift is found by averaging over 256 redshifts...
        """
        z1 = units.nu21 / self.nu_upper - 1.0
        z2 = units.nu21 / self.nu_lower - 1.0

        return self.powerspectrum_1D(k_vec, z1, z2, 256)

    def get_kiyo_field_physical(
        self, refinement=1, density_only=False, no_mean=False, no_evolution=False
    ):
        r"""Fetch a realisation of the 21cm signal (NOTE: in K)
        """
        z1 = units.nu21 / self.nu_upper - 1.0
        z2 = units.nu21 / self.nu_lower - 1.0

        (cube, rsf, d) = self.realisation(
            z1,
            z2,
            self.x_width,
            self.y_width,
            self.nu_num,
            self.x_num,
            self.y_num,
            refinement=refinement,
            zspace=False,
            report_physical=True,
            density_only=density_only,
            no_mean=no_mean,
            no_evolution=no_evolution,
        )

        return (cube, rsf, d)


class CorrZA(CorrBiasedTracer):
    r"""Correlation function of (possibly) biased tracer of the
    matter field fluctuations using the Zeldovich approximation.

    Incorporates reasonable approximations for the growth factor and
    growth rate.

    """

    def __init__(self, ps=None, nside_factor=4, ndiv_radial=4,
                 ps_redshift=0.0, sigma_v=0.0, twostep=False, **kwargs):

        # Wether to compute the ZA field in two steps
        self.twostep = twostep
        # Factor to increase nside for the particle mesh
        self.nside_factor = nside_factor
        # Factor to increase radial resolution for the particle mesh 
        self.ndiv_radial = ndiv_radial

        super(CorrZA, self).__init__(ps, ps_redshift, sigma_v, **kwargs)

        # Simpler alias for Power spectrum
        self.ps = self.ps_vv

    # Wrap angular_powerspectrum to gen the PS of the Newtonian potential
    def potential_angular_powerspectrum(self, l, nu1, nu2, redshift=False):
        """Calculate the angular powerspectrum of the inverse Laplacian of delta.

        Parameters
        ----------
        l : np.ndarray
            Multipoles to calculate at.
        nu1, nu2 : np.ndarray
            Frequencies/redshifts to calculate at.
        redshift : boolean, optional
            If `False` (default) interperet `nu1`, `nu2` as frequencies,
            otherwise they are redshifts (relative to the 21cm line).

        Returns
        -------
        aps : np.ndarray
        """

        if not redshift:
            z1 = units.nu21 / nu1 - 1.0
            z2 = units.nu21 / nu2 - 1.0
        else:
            z1 = nu1
            z2 = nu2

        return corr.RedshiftCorrelation.angular_powerspectrum(
                            self, l, z1, z2,
                            include_prefactor=False, 
                            include_bias=False, include_rsd=False, 
                            include_evolution=False, potential=True)

    # Za in two steps: displace - bias - displace RSD
    def getsky_2s(self):
        """Create a map of the unpolarised sky using the
        zeldovich approximation.
        """

        import healpy as hp
        from cora.util import pmesh as pm

        lmax = 3 * self.nside - 1

        # Angular power spectrum for the Newtonian potential
        cla_pot = skysim.clarray(self.potential_angular_powerspectrum, lmax,
                                 self.freqs_full, zromb=self.oversample)

        redshift_array = units.nu21 / self.freqs_full - 1.
        comovd = self.cosmology.comoving_distance(redshift_array)

        # Generate potential field maps and their spacial derivatives:
        maps_der1 = skysim.mkfullsky_der1(cla_pot, self.nside, comovd)

        # Apply growth factor:
        D = self.growth_factor(redshift_array) / self.growth_factor(self.ps_redshift)
        maps_der1 *= D[np.newaxis, :, np.newaxis]

        npix = hp.pixelfunc.nside2npix(self.nside)
        base_coordinates = np.array(hp.pixelfunc.pix2ang(
                                        self.nside, np.arange(npix)))

        # Divide by comoving distance to get angular displacements:
        maps_der1[1:3] /= comovd[np.newaxis, :, np.newaxis]
        # Divide by sin(theta) to get phi displacements, 
        # not angular displacements:
        maps_der1[2] /= np.sin(base_coordinates[0][np.newaxis, :])

        # Compute extra displacement due to redshift space distortions.
        disp_rsd = np.zeros(maps_der1[1:].shape, dtype=maps_der1.dtype)
        disp_rsd[2] = maps_der1[3] * self.growth_rate(redshift_array)[:, np.newaxis]

        # Compute density in the Zeldovich approximation:
        maps0 = np.ones(maps_der1[0].shape, dtype=maps_der1.dtype)
        delta_za = pm.za_density(maps_der1[1:], maps0, self.nside,
                                 comovd, self.nside_factor, self.ndiv_radial,
                                 nslices=2)
        # Apply Eulerian bias
        delta_za *= self.bias_z(redshift_array)[:, np.newaxis]
        # Add mean (needed for pm.za_density())
        delta_za += 1.0
        # Crop negative densities
        print("Fraction of pixels croped: {0:0.3e}, shape: {1}".format(
              np.sum(delta_za<0.)/np.prod(delta_za.shape), delta_za.shape))
        delta_za = np.where(delta_za<0., 0., delta_za)

        # Displace for RSD:
        delta_za = pm.za_density(disp_rsd, delta_za, self.nside,
                                 comovd, self.nside_factor, self.ndiv_radial,
                                 nslices=2) 

        # Recover original frequency range:
        delta_za = delta_za[self.freq_slice]

        # Apply prefactor
        redshift_array = units.nu21 / self.frequencies - 1.
        pref = self.prefactor(redshift_array)[:, np.newaxis]

        return delta_za * pref

    @CorrBiasedTracer.frequencies.setter
    def frequencies(self,freq):
        """Overrides the frequencies setter to also
        define freqs_full and freq_slice.
        """
        self._frequencies = freq
        self.freqs_full, self.freq_slice = self._pad_freqs(self.frequencies)

    # Overwrite getsky to implement the steps necessary for the
    # Zeldovich approximation case
    def getsky_1s(self, gaussvars_list=None):
        """Create a map of the unpolarised sky using the
        zeldovich approximation.
        """

        import healpy as hp
        from cora.util import pmesh as pm

        lmax = 3 * self.nside - 1
        nz_full = len(self.freqs_full)

        if gaussvars_list is None:
            # Generate gaussian random numbers for realization
            gaussvars_list = [nputil.complex_std_normal((nz_full, l + 1))
                              for l in range(lmax+1)]

        # Angular power spectrum for the matter field
        cla = skysim.clarray(self.raw_angular_powerspectrum, lmax,
                             self.freqs_full, zromb=self.oversample)
        # Angular power spectrum for the Newtonian potential
        cla_pot = skysim.clarray(self.potential_angular_powerspectrum, lmax,
                                 self.freqs_full, zromb=self.oversample)

        redshift_array = units.nu21 / self.freqs_full - 1.
        comovd = self.cosmology.comoving_distance(redshift_array)

        # Generate matter field maps
        maps = skysim.mkfullsky(cla, self.nside, 
                                gaussvars_list=gaussvars_list)
        # Generate potential field maps and their spacial derivatives:
        maps_der1 = skysim.mkfullsky_der1(cla_pot, self.nside, comovd, 
                                          gaussvars_list=gaussvars_list)

        # Apply growth factor:
        D = (self.growth_factor(redshift_array) 
             / self.growth_factor(self.ps_redshift))
        maps *= D[:, np.newaxis]
        maps_der1 *= D[np.newaxis, :, np.newaxis]

        npix = hp.pixelfunc.nside2npix(self.nside)
        base_coordinates = np.array(hp.pixelfunc.pix2ang(
                                        self.nside, np.arange(npix)))

        # Divide by comoving distance to get angular displacements:
        maps_der1[1:3] /= comovd[np.newaxis, :, np.newaxis]
        # Divide by sin(theta) to get phi displacements, 
        # not angular displacements:
        maps_der1[2] /= np.sin(base_coordinates[0][np.newaxis, :])
        # Add extra displacement due to redshift space distortions.
        maps_der1[3] *= (1. + self.growth_rate(redshift_array))[:, np.newaxis]

        # Apply a lognormal transformation on the biased Lagrangian field
        # to get rid of negative densities. Pass filed with mean 1, not 0.
#        maps = np.exp(maps*self.lagbias_z(redshift_array)[:, np.newaxis])
        maps = maps*self.lagbias_z(redshift_array)[:, np.newaxis] + 1.
        print("Fraction of pixels croped: {0:0.3e}, shape: {1}".format(
            np.sum(maps<0.)/np.prod(maps.shape), maps.shape))
        maps = np.where(maps<0.,0.,maps)

        # Compute density in the Zeldovich approximation:
        # Multiply linear density maps by lagrangian bias of the tracer.
        delta_za = pm.za_density(
                        maps_der1[1:], maps, self.nside, comovd,
                        self.nside_factor, self.ndiv_radial, nslices=2)

        # Recover original frequency range:
        delta_za = delta_za[self.freq_slice]

        # Apply prefactor
        redshift_array = units.nu21 / self.frequencies - 1.
        pref = self.prefactor(redshift_array)[:, np.newaxis]

        return delta_za * pref

    def getsky(self):
        if self.twostep:
            return self.getsky_2s()
        else:
            return self.getsky_1s()

    def lagbias_z(self, z):
        """Linear Lagrangian bias from the Eulerian counterpart
        """
        return self.bias_z(z) - 1.

    def _pad_freqs(self, freqs):
        """ Add frequency padding for correct particle displacement at
        the edges.
        """

        def get_zpad(ff):
            # Pad with 3 bins at 400MHz and 6 bins ad 800MHz.
            # Linear (integer) fill in between. Ensures maps average is
            # less than 1% of RMS at all frequencies at frequency edge bins.
            return int(np.around(3./400.*(ff-400.)+3.))

        zpad = (get_zpad(freqs[0]), get_zpad(freqs[1]))
        # Assume constant linear spacing in freqsuency:
        df = freqs[1]-freqs[0]
        # Overwrite frequency axis with padded one:
        lower_freq_pad = freqs[0]+np.arange(-zpad[0], 0)*df
        upper_freq_pad = freqs[-1]+np.arange(1, 1+zpad[1])*df
        freqs = np.concatenate((lower_freq_pad, freqs, upper_freq_pad))
        # Frequency slice to recover original axis:
        frq_slc = np.ones(freqs.shape, dtype=bool)
        frq_slc[:zpad[0]] = False
        frq_slc[-zpad[1]:] = False

        return freqs, frq_slc


# TODO: this was moved from elsewhere and needs to be tested again
def theory_power_spectrum(
    redshift_filekey, bin_centers, unitless=True, fileout="theory.dat"
):
    r"""simple caller to output a power spectrum"""
    zfilename = datapath_db.fetch(redshift_filekey, intend_read=True, pick="1")

    zspace_cube = algebra.make_vect(algebra.load(zfilename))
    simobj = corr21cm.CorrBiasedTracer.like_kiyo_map(zspace_cube)
    pwrspec_input = simobj.get_pwrspec(bin_center)
    if unitless:
        pwrspec_input *= bin_center ** 3.0 / 2.0 / math.pi / math.pi

    outfile = open(fileout, "w")
    for specdata in zip(bin_left, bin_center, bin_right, pwrspec_input):
        outfile.write(("%10.15g " * 4 + "\n") % specdata)

    outfile.close()


class EoR21cm(CorrBiasedTracer):
    def T_b(self, z):

        r"""Mean 21cm brightness temperature at a given redshift.
        
        According to Eq.(4) of M. G. Santos, L. Ferramacho and M. B. Silva, 2009, Fast Large Volume Simulations of the Epoch of Reionization.

        Temperature is in K.

        Parameters
        ----------
        z : array_like
            Redshift to calculate at.

        Returns
        -------
        T_b : array_like
        """
        return (
            23e-3
            * (self.cosmology.omega_b * (self.cosmology.H0 / 100.0) ** 2 / 0.02)
            * (
                0.15
                / (self.cosmology.omega_m * (self.cosmology.H0 / 100.0) ** 2)
                * ((1.0 + z) / 10)
            )
            ** 0.5
            * ((self.cosmology.H0 / 100.0) / 0.7) ** -1
        )

    def omega_HI(self, z):
        return 5e-3

    def x_h(self, z):
        r"""Neutral hydrogen fraction at a given redshift.

        Just returns a constant at the moment. Need to add a more
        sophisticated model.

        Parameters
        ----------
        z : array_like
            Redshift to calculate at.

        Returns
        -------
        x_e : array_like
        """
        return 0.25

    def bias_z(self, z):
        r"""Changing the bias from 1 to 3, based on EoR bias estimates in Santos 2004 (arXiv: 0408515)"""

        return np.ones_like(z) * 3.0
