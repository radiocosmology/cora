
import numpy as np

from cora.core import maps
from cora.util import cubicspline as cs
from cora.util import units
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


class Corr21cmZA(Corr21cm):
    r"""Correlation function of HI brightness temperature fluctuations
    using the Zeldovich approximation.

    Incorporates reasonable approximations for the growth factor and
    growth rate.

    """

    def __init__(self, ps=None, nside_factor=4, ndiv_radial=4,
                 redshift=0.0, sigma_v=0.0, **kwargs):

        from os.path import join, dirname
        if ps is None:

            psfile = join(dirname(__file__), "data/ps_z1.5.dat")
            redshift = 1.5

            c1 = cs.LogInterpolater.fromfile(psfile)
            # Divide power spectrum by k^4 to get a field proportional to
            # the Newtonian potential. Usefull to generate the displacement
            # field Psi.
            ps = lambda k: np.exp(-0.5 * k**2 / self._kstar**2) * c1(k) / k**4

        self._sigma_v = sigma_v
        # Factor to increase nside for the particle mesh
        self.nside_factor = nside_factor
        # Factor to increase radial resolution for the particle mesh 
        self.ndiv_radial = ndiv_radial

        corr.RedshiftCorrelation.__init__(self, ps_vv=ps, redshift=redshift)
        self._load_cache(join(dirname(__file__), "data/corr_z1.5.dat"))
        # self.load_fft_cache(join(dirname(__file__),"data/fftcache.npz"))

    # Remove growth factor and allow using frequency.
    # TODO: This function was used for testing, should be deleted.
    def angular_powerspectrum_nogrowth(self, l, nu1, nu2, redshift=False):
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

        D1 = self.growth_factor(z1) / self.growth_factor(self.ps_redshift)
        D2 = self.growth_factor(z2) / self.growth_factor(self.ps_redshift)
        return corr.RedshiftCorrelation.angular_powerspectrum(
                                                    self, l, z1, z2)/(D1*D2)

    # Override angular_powerspectrum_realspace to switch to allow using frequency
    def angular_powerspectrum_realspace(self, l, nu1, nu2, redshift=False):
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

        return corr.RedshiftCorrelation.angular_powerspectrum_realspace(
                                                            self, l, z1, z2)

    # Overwrite getsky to implement the steps necessary for the
    # Zeldovich approximation case
    # TODO: Questio: Maybe I should have put this functionality in
    # cora.core.maps ?
    def getsky(self):
        """Create a map of the unpolarised sky using the
        zeldovich approximation.
        """

        from cora.core import skysim
        import healpy as hp
        from cora.util import pmesh as pm

        lmax = 3 * self.nside - 1

        # Add frequency padding for correct particle displacement at edges
        frequencies, freq_slice = self._pad_freqs(self._frequencies)

        cla = skysim.clarray(self.angular_powerspectrum_nogrowth, lmax,
                             frequencies, zromb=self.oversample)

#        cla = skysim.clarray(self.angular_powerspectrum, lmax, self.nu_pixels,
#                             zromb=self.oversample)

        redshift_array = units.nu21 / frequencies - 1.
        comovd = self.cosmology.comoving_distance(redshift_array)
        # Generate maps and their spacial derivatives:
        maps_der1 = skysim.mkfullsky_der1(cla, self.nside, comovd)

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
        # Add extra displacement due to redshift space distortions.
        maps_der1[3] *= (1. + self.growth_rate(redshift_array))[:, np.newaxis]

        # Compute density in the Zeldovich approximation:
        delta_za = pm.za_density(maps_der1[1:], self.nside, comovd,
                                 self.nside_factor, self.ndiv_radial,
                                 nslices=2)
        # Recover original frequency range:
        delta_za = delta_za[freq_slice]

        # TODO: Before returning should I apply:
        # prefactors (to get HI brightness)
        # bias
        # Add mean value: self.mean_nu(self.nu_pixels)
        # ?
        return delta_za

    def _pad_freqs(self, freqs):
        """ Add frequency padding for correct particle displacement at
        the edges.
        """

        def get_zpad(ff):
            # Pad with 2 bins at 400MHz and 5 bins ad 800MHz.
            # Linear (integer) fill in between.
            return int(np.around(3./400.*(ff-400.)+2.))

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
