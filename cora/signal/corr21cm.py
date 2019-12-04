# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import numpy as np

from cora.core import maps
from cora.util import cubicspline as cs
from cora.util import units, nputil, halomodel
from cora.signal import corr


class Corr21cm(corr.RedshiftCorrelation, maps.Sky3d):
    r"""Correlation function of HI brightness temperature fluctuations.

    Incorporates reasonable approximations for the growth factor and
    growth rate.

    """

    add_mean = False

    _kstar = 5.0

    def __init__(self, ps = None, redshift = 0.0, sigma_v=0.0, **kwargs):

        from os.path import join, dirname
        if ps == None:

            psfile = join(dirname(__file__),"data/ps_z1.5.dat")
            redshift = 1.5

            c1 = cs.LogInterpolater.fromfile(psfile)
            ps = lambda k: np.exp(-0.5 * k**2 / self._kstar**2) * c1(k)

        self._sigma_v = sigma_v

        corr.RedshiftCorrelation.__init__(self, ps_vv=ps, redshift=redshift)
        self._load_cache(join(dirname(__file__),"data/corr_z1.5.dat"))
        #self.load_fft_cache(join(dirname(__file__),"data/fftcache.npz"))


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

        return (3.9e-4 * ((self.cosmology.omega_m + self.cosmology.omega_l * (1+z)**-3) / 0.29)**-0.5
                * ((1.0 + z) / 2.5)**0.5 * (self.omega_HI(z) / 1e-3))


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
        return self.T_b(z)


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

        x = ((1.0 / self.cosmology.omega_m) - 1.0) / (1.0 + z)**3

        num = 1.0 + 1.175*x + 0.3064*x**2 + 0.005355*x**3
        den = 1.0 + 1.857*x + 1.021 *x**2 + 0.1530  *x**3

        d = (1.0 + x)**0.5 / (1.0 + z) * num / den

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

        x = ((1.0 / self.cosmology.omega_m) - 1.0) / (1.0 + z)**3

        dnum = 3.0*x*(1.175 + 0.6127*x + 0.01607*x**2)
        dden = 3.0*x*(1.857 + 2.042 *x + 0.4590 *x**2)

        num = 1.0 + 1.175*x + 0.3064*x**2 + 0.005355*x**3
        den = 1.0 + 1.857*x + 1.021 *x**2 + 0.1530  *x**3

        f = 1.0 + 1.5 * x / (1.0 + x) + dnum / num - dden / den

        return f


    def bias_z(self, z):
        r"""It's unclear what the bias should be. Using 1 for the moment. """

        return np.ones_like(z) * 1.0

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


    def mean_nu(self, freq):

        return self.mean(units.nu21 / freq - 1.0)


    def getfield(self):
        r"""Fetch a realisation of the 21cm signal.
        """
        z1 = units.nu21 / self.nu_upper - 1.0
        z2 = units.nu21 / self.nu_lower - 1.0

        cube = self.realisation(z1, z2, self.x_width, self.y_width, self.nu_num, self.x_num, self.y_num, zspace = False)[::-1,:,:].copy()

        return cube


    def get_kiyo_field(self, refinement=1):
        r"""Fetch a realisation of the 21cm signal (NOTE: in K)
        """
        z1 = units.nu21 / self.nu_upper - 1.0
        z2 = units.nu21 / self.nu_lower - 1.0

        cube = self.realisation(z1, z2, self.x_width, self.y_width,
                                self.nu_num, self.x_num, self.y_num,
                                refinement=refinement, zspace = False)

        return cube

    def get_pwrspec(self, k_vec):
        r"""Fetch the power spectrum of the signal
        The effective redshift is found by averaging over 256 redshifts...
        """
        z1 = units.nu21 / self.nu_upper - 1.0
        z2 = units.nu21 / self.nu_lower - 1.0

        return self.powerspectrum_1D(k_vec, z1, z2, 256)

    def get_kiyo_field_physical(self, refinement=1, density_only=False,
                                no_mean=False, no_evolution=False):
        r"""Fetch a realisation of the 21cm signal (NOTE: in K)
        """
        z1 = units.nu21 / self.nu_upper - 1.0
        z2 = units.nu21 / self.nu_lower - 1.0

        (cube, rsf, d) = self.realisation(z1, z2, self.x_width, self.y_width,
                                self.nu_num, self.x_num, self.y_num,
                                refinement=refinement, zspace = False,
                                report_physical=True, density_only=density_only,
                                no_mean=no_mean, no_evolution=no_evolution)

        return (cube, rsf, d)


class CorrZA(Corr21cm):
    r"""Correlation function of (possibly) biased tracer of the
    matter field fluctuations using the Zeldovich approximation.

    Incorporates reasonable approximations for the growth factor and
    growth rate.

    """

    def __init__(self, ps=None, nside_factor=4, ndiv_radial=4,
                 ps_redshift=0.0, sigma_v=0.0, **kwargs):

        # Factor to increase nside for the particle mesh
        self.nside_factor = nside_factor
        # Factor to increase radial resolution for the particle mesh 
        self.ndiv_radial = ndiv_radial

        super(CorrZA, self).__init__(ps, ps_redshift, sigma_v, **kwargs)

        # Simpler alias for Power spectrum
        self.ps = self.ps_vv

    # Override raw_angular_powerspectrum to switch to allow using frequency
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

        return corr.RedshiftCorrelation.raw_angular_powerspectrum(
                                                        self, l, z1, z2)

    # Override raw_angular_powerspectrum to gen the PS of the Newtonian potential
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

        return corr.RedshiftCorrelation.raw_angular_powerspectrum(
                                        self, l, z1, z2, potential=True)


    # Za in two steps: displace - bias - displace RSD
    def getsky_2s(self):
        """Create a map of the unpolarised sky using the
        zeldovich approximation.
        """

        from cora.core import skysim
        import healpy as hp
        from cora.util import pmesh as pm

        lmax = 3 * self.nside - 1

        # Add frequency padding for correct particle displacement at edges
        frequencies, freq_slice = self._pad_freqs(self._frequencies)

#        # Angular power spectrum for the matter field
#        cla = skysim.clarray(self.raw_angular_powerspectrum, lmax,
#                             frequencies, zromb=self.oversample)

        # Angular power spectrum for the Newtonian potential
        cla_pot = skysim.clarray(self.potential_angular_powerspectrum, lmax,
                                 frequencies, zromb=self.oversample)

        redshift_array = units.nu21 / frequencies - 1.
        comovd = self.cosmology.comoving_distance(redshift_array)

#        # Generate gaussian random numbers for realization
#        gaussvars_list = [nputil.complex_std_normal((cla_pot.shape[1], l + 1)) 
#                          for l in range(cla_pot.shape[0])]
#        # Generate matter field maps
#        maps = skysim.mkfullsky(cla, self.nside, 
#                                gaussvars_list=gaussvars_list)

        # Generate potential field maps and their spacial derivatives:
        maps_der1 = skysim.mkfullsky_der1(cla_pot, self.nside, comovd)

        # Apply growth factor:
        D = self.growth_factor(redshift_array) / self.growth_factor(self.ps_redshift)
#        maps *= D[:, np.newaxis]
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
        # TODO: Multiply linear density maps_der1[0] by lagrangean bias of
        # the tracers involved.
        maps0 = np.zeros(maps_der1[0].shape, dtype=maps_der1.dtype) # TODO: delete
#        delta_za = pm.za_density(maps_der1[1:], maps, self.nside,
        delta_za = pm.za_density(maps_der1[1:], maps0, self.nside,
                                 comovd, self.nside_factor, self.ndiv_radial,
                                 nslices=2)
        # Apply simple Eulerian bias
        delta_za = delta_za * 2.
        # Displace for RSD:
        delta_za = pm.za_density(disp_rsd, delta_za, self.nside,
                                 comovd, self.nside_factor, self.ndiv_radial,
                                 nslices=2) 

        # Recover original frequency range:
        delta_za = delta_za[freq_slice]

        # Multiply by prefactor (21cm brightness)
        pref = self.prefactor(redshift_array[freq_slice])[:, np.newaxis]
#        bias = self.bias_z(redshift_array[freq_slice])[:, np.newaxis]

        # TODO: Add mean value: self.mean_nu(self.nu_pixels) ?
        #return delta_za * pref * bias # TODO: remove. Bias done in Lagrangian space.
        #return delta_za * pref
        return delta_za, maps_der1

    @Corr21cm.frequencies.setter
    def frequencies(self,freq):
        """Overrides the frequencies setter to also
        define freqs_full and freq_slice.
        """
        self._frequencies = freq
        self.freqs_full, self.freq_slice = self._pad_freqs(self.frequencies)

    # Overwrite getsky to implement the steps necessary for the
    # Zeldovich approximation case
    def getsky(self, gaussvars_list=None):
        """Create a map of the unpolarised sky using the
        zeldovich approximation.
        """

        from cora.core import skysim
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
        maps = np.where(maps<0.,0.,maps)

        # Compute density in the Zeldovich approximation:
        # Multiply linear density maps by lagrangian bias of the tracer.
        delta_za = pm.za_density(
                        maps_der1[1:], maps, self.nside, comovd,
                        self.nside_factor, self.ndiv_radial, nslices=2)

        # Recover original frequency range:
        delta_za = delta_za[self.freq_slice]

        # TODO: Add mean value: self.mean_nu(self.nu_pixels) ?
        #return delta_za, maps, maps_der1
        return delta_za

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
def theory_power_spectrum(redshift_filekey, bin_centers,
                          unitless=True, fileout="theory.dat"):
    r"""simple caller to output a power spectrum"""
    zfilename = datapath_db.fetch(redshift_filekey, intend_read=True,
                                      pick='1')

    zspace_cube = algebra.make_vect(algebra.load(zfilename))
    simobj = corr21cm.Corr21cm.like_kiyo_map(zspace_cube)
    pwrspec_input = simobj.get_pwrspec(bin_center)
    if unitless:
        pwrspec_input *= bin_center ** 3. / 2. / math.pi / math.pi

    outfile = open(fileout, "w")
    for specdata in zip(bin_left, bin_center, bin_right, pwrspec_input):
        outfile.write(("%10.15g " * 4 + "\n") % specdata)

    outfile.close()


class CorrQuasarZA(CorrZA):
    r"""Correlation function of the Quasar number density fluctuations
    using the Zeldovich approximation.

    Overwrites CorrZA to include the bias taken from arXiv:1705.04718 

    """

    def bias_z(self, z):
        """Linear Eulerian bias taken from arXiv:1705.04718
        """
        alpha, beta = 0.278, 2.393
        return alpha*((1 + z)**2 - 6.565) + beta


class Corr21cmZA(CorrZA):
    r"""Correlation function of HI brightness temperature fluctuations
    using the Zeldovich approximation.

    Incorporates reasonable approximations for the growth factor and
    growth rate.

    """

    def __init__(self, ps=None, nside_factor=4, ndiv_radial=4,
                 ps_redshift=0.0, sigma_v=0.0, **kwargs):
        """
        """

        from os.path import join, dirname
        if ps is None:
            psfile = join(dirname(__file__),"data/ps_z1.5.dat")
            ps_redshift = 1.5
            ps_full = cs.LogInterpolater.fromfile(psfile)
            ps = lambda k: np.exp(-0.5 * k**2 / self._kstar**2) * ps_full(k)
        else:
            # This might mean passing a smoothed PS to the halo model. 
            # TODO: This is probably a bug. Maybe I should not accept ps as
            # an input parameter in Corr21cmZA.
            ps_full = ps

        super(Corr21cmZA, self).__init__(ps, nside_factor, ndiv_radial,
                                         ps_redshift, sigma_v, **kwargs)

        # Pass the full (not smoothed) Power Spectrum to the halo model
        # HaloModel uses powerspectrum normalized to z=0
        def hmps(k):
            return ps_full(k) * (self.growth_factor(0.)
                            / self.growth_factor(self.ps_redshift))**2

        # Initialize HaloModel
        self.hm = halomodel.HaloModel(gf=self.growth_factor, ps=hmps)
        self.hm.check_update()

    def lagbias_z(self, z):
        """ Overwrites lagbias_z to use a halo model derived HI bias.
        """
        bias = []
        for zz in z:
            self.hm.redshift = zz
            bias.append(self.hm.lbias_h1())

        return np.array(bias)

    def bias_z(self, z):
        """Linear Eulerian bias from the Lagrangian counterpart
        """
        return self.lagbias_z(z) + 1.

    def getsky(self,gaussvars_list=None):
        """ Overwrites getsky to add correct pre-factors
        """
        # Get sky
        delta_za = super(Corr21cmZA, self).getsky(gaussvars_list)
        # Apply prefactor
        redshift_array = units.nu21 / self.frequencies - 1.
        pref = self.prefactor(redshift_array)[:, np.newaxis]

        # TODO: add prefactor
        #print pref
        return delta_za * pref
        #return delta_za


class EoR21cm(Corr21cm):

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
        """

        return (23e-3 * (self.cosmology.omega_b * (self.cosmology.H0 / 100.0)**2 / 0.02) * ((0.15 / self.cosmology.omega_m * (self.cosmology.H0 / 100.0)**2) * ((1.0 + z)/10))**0.5 * ((self.cosmology.H0 / 100.0) / 0.7)**-1)


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
