import scipy
import scipy.ndimage
import scipy.fftpack
import scipy.special
import numpy as np
import math

from os.path import dirname, join, exists

from utils import sphbessel, integrate, units, fftutil
from utils import cubicspline as cs

from utils.cosmology import Cosmology
from gaussianfield import RandomField

#ps = cs.LogInterpolater.fromfile( join(dirname(__file__),"data/ps.dat")).value

_feedback = False



class RedshiftCorrelation(object):
    r"""A class for calculating redshift-space correlations.

    The mapping from real to redshift space produces anisotropic
    correlations, this class calculates them within the linear
    regime. As a minimum the velocity power spectrum `ps_vv` must be
    specified, the statistics of the observable can be specified
    explicitly (in `ps_dd` and `ps_dv`), or as a `bias` relative to
    the velocity spectrum.

    As the integrals to calculate the correlations can be slow, this
    class can construct a table for interpolating them. This table can
    be saved to a file, and reloaded as need.

    Parameters
    ----------
    ps_vv : function, optional
        A function which gives the velocity power spectrum at a
        wavenumber k (in units of h Mpc^{-1}).

    ps_dd : function, optional
        A function which gives the power spectrum of the observable.

    ps_dv : function, optional
        A function which gives the cross power spectrum of the
        observable and the velocity.

    redshift : scalar, optional
        The redshift at which the power spectra are
        calculated. Defaults to a redshift, z = 0.

    bias : scalar, optional
        The bias between the observable and the velocities (if the
        statistics are not specified directly). Defaults to a bias of
        1.0.

    Attributes
    ----------
    ps_vv, ps_dd, ps_dv : function
        The statistics of the obserables and velocities (see Parameters).

    ps_redshift : scalar
        Redshift of the power spectra (see Parameters).

    bias : scalar
        Bias of the observable (see Parameters).

    cosmology : instance of Cosmology()
        An instance of the Cosmology class to allow mapping of
        redshifts to Cosmological distances.

    Notes
    -----
    To allow more sophisticated behaviour the four methods
    `growth_factor`, `growth_rate`, `bias_z` and `prefactor` may be
    replaced. These return their respective quantities as functions of
    redshift. See their method documentation for details. This only
    really makes sense when using `_vv_only`, though the functions can
    be still be used to allow some redshift scaling of individual
    terms.
    """

    ps_vv = None
    ps_dd = None
    ps_dv = None


    ps_redshift = 0.0
    bias = 1.0

    _vv_only = False

    _cached = False
    _vv0i = None
    _vv2i = None
    _vv4i = None

    _dd0i = None
    _dv0i = None
    _dv2i = None

    cosmology = Cosmology()

    def __init__(self, ps_vv = None, ps_dd = None, ps_dv = None,
                 redshift = 0.0, bias = 1.0):
        self.ps_vv = ps_vv
        self.ps_dd = ps_dd
        self.ps_dv = ps_dv
        self.ps_redshift = redshift
        self.bias = bias
        self._vv_only = False if ps_dd and ps_dv else True


    @classmethod
    def from_file_matterps(cls, fname, redshift = 0.0, bias = 1.0):
        r"""Initialise from a cached, single power spectrum, file.

        Parameters
        ----------
        fname : string
            Name of the cache file.
        redshift : scalar, optional
            Redshift that the power spectrum is defined at (default z = 0).
        bias : scalar, optional
            The bias of the observable relative to the velocity field.
        """

        rc = cls(redshift=redshift, bias=bias)
        rc._vv_only = True
        rc._load_cache(fname)

        return rc

    @classmethod
    def from_file_fullps(cls, fname, redshift = 0.0):
        r"""Initialise from a cached, multi power spectrum, file.

        Parameters
        ----------
        fname : string
            Name of the cache file.
        redshift : scalar, optional
            Redshift that the power spectra are defined at (default z = 0).
        """

        rc = cls(redshift=redshift)
        rc._vv_only = False
        rc._load_cache(fname)

        return rc


    def powerspectrum(self, kpar, kperp, z1 = None, z2 = None):
        r"""A vectorized routine for calculating the redshift space powerspectrum.

        Parameters
        ----------
        kpar : array_like
            The parallel component of the k-vector.
        kperp : array_like
            The perpendicular component of the k-vector.
        z1, z2 : array_like, optional
            The redshifts of the wavevectors to correlate. If either
            is None, use the default redshift `ps_redshift`.

        Returns
        -------
        ps : array_like
            The redshift space power spectrum at the given k-vector and redshift.
        """
        if z1 == None:
            z1 = self.ps_redshift
        if z2 == None:
            z2 = self.ps_redshift

        b1 = self.bias_z(z1)
        b2 = self.bias_z(z2)
        f1 = self.growth_rate(z1)
        f2 = self.growth_rate(z2)
        D1 = self.growth_factor(z1) / self.growth_factor(self.ps_redshift)
        D2 = self.growth_factor(z2) / self.growth_factor(self.ps_redshift)
        pf1 = self.prefactor(z1)
        pf2 = self.prefactor(z2)

        k2 = kpar**2 + kperp**2
        k = k2**0.5
        mu2 = kpar**2 / k2

        if self._vv_only:
            ps = self.ps_vv(k) * (b1 + mu2 * f1) * (b2 + mu2 * f2)
        else:
            ps = (b1*b2*self.ps_dd(k) + mu2 * self.ps_dv(k) * (f1*b2 + f2*b1) + mu2**2 * f1*f2 * self.ps_vv(k))


        return D1*D2*pf1*pf2*ps



    def redshiftspace_correlation(self, pi, sigma, z1 = None, z2 = None):
        """The correlation function in the flat-sky approximation.

        This is a vectorized function. The inputs `pi` and `sigma`
        must have the same shape (if arrays), or be scalars. This
        function will perform redshift dependent scaling for the
        growth and biases.

        Parameters
        ----------
        pi : array_like
            The separation in the radial direction (in units of h^{-1} Mpc).
        sigma : array_like
            The separation in the transverse direction (in units of h^{-1} Mpc).
        z1 : array_like, optional
            The redshift corresponding to the first point.
        z2 : array_like, optional
            The redshift corresponding to the first point.

        Returns
        -------
        corr : array_like
            The correlation function evaluated at the input values.

        Notes
        -----
        If neither `z1` or `z2` are provided assume that both points
        are at the same redshift as the power spectrum. If only `z1`
        is provided assume the second point is at the same redshift as
        the first.
        """

        r = (pi**2 + sigma**2)**0.5

        # Calculate mu with a small constant added to regularise cases
        # of pi = sigma = 0
        mu = pi / (r + 1e-100)

        if z1 == None and z2 == None:
            z1 = self.ps_redshift
            z2 = self.ps_redshift
        elif z2 == None:
            z2 = z1

        if self._cached:
            xvv_0 = self._vv0i(r)
            xvv_2 = self._vv2i(r)
            xvv_4 = self._vv4i(r)

            if self._vv_only:
                xdd_0 = xvv_0
                xdv_0 = xvv_0
                xdv_2 = xvv_2
            else:
                xdd_0 = self._dd0i(r)
                xdv_0 = self._dv0i(r)
                xdv_2 = self._dv2i(r)

        else:
            xvv_0 = _integrate(r, 0, self.ps_vv)
            xvv_2 = _integrate(r, 2, self.ps_vv)
            xvv_4 = _integrate(r, 4, self.ps_vv)

            if self._vv_only:
                xdd_0 = xvv_0
                xdv_0 = xvv_0
                xdv_2 = xvv_2
            else:
                xdd_0 = _integrate(r, 0, self.ps_dd)
                xdv_0 = _integrate(r, 0, self.ps_dv)
                xdv_2 = _integrate(r, 2, self.ps_dv)

        #if self._vv_only:
        b1 = self.bias_z(z1)
        b2 = self.bias_z(z2)
        f1 = self.growth_rate(z1)
        f2 = self.growth_rate(z2)

        xdd_0 *= b1*b2
        xdv_0 *= 0.5*(b1*f2 + b2*f1)
        xdv_2 *= 0.5*(b1*f2 + b2*f1)

        xvv_0 *= f1*f2
        xvv_2 *= f1*f2
        xvv_4 *= f1*f2

        D1 = self.growth_factor(z1) / self.growth_factor(self.ps_redshift)
        D2 = self.growth_factor(z2) / self.growth_factor(self.ps_redshift)

        pf1 = self.prefactor(z1)
        pf2 = self.prefactor(z2)

        pl0 = 1.0
        pl2 = _pl(2, mu)
        pl4 = _pl(4, mu)

        return ((xdd_0 + 2.0 / 3.0 * xdv_0 + 1.0 / 5.0 * xvv_0) * pl0 -
                (4.0 / 3.0 * xdv_2 + 4.0 / 7.0 * xvv_2) * pl2 +
                8.0 / 35.0 * xvv_4 * pl4) * D1 * D2 * pf1 * pf2



    def angular_correlation(self, theta, z1, z2):
        r"""Angular correlation function (in a flat-sky approximation).

        Parameters
        ----------
        theta : array_like
            The angle between the two points in radians.
        z1, z2 : array_like
            The redshift of the points.

        Returns
        -------
        corr : array_like
            The correlation function at the points.
        """


        za = (z1+z2) / 2.0
        sigma = theta * self.cosmology.proper_distance(za)
        pi = self.cosmology.comoving_distance(z2) - self.cosmology.comoving_distance(z1)

        return self.redshiftspace_correlation(pi, sigma, z1, z2)



    def _load_cache(self, fname):

        if not exists(fname):
            raise Exception("Cache file does not exist.")

        a = np.loadtxt(fname)
        ra = a[:,0]
        vv0 = a[:,1]
        vv2 = a[:,2]
        vv4 = a[:,3]
        if not self._vv_only:
            if(a.shape[1] != 7):
                raise Exception("Cache file has wrong number of columns.")
            dd0 = a[:,4]
            dv0 = a[:,5]
            dv2 = a[:,6]

        self._vv0i = cs.Interpolater(ra, vv0)
        self._vv2i = cs.Interpolater(ra, vv2)
        self._vv4i = cs.Interpolater(ra, vv4)

        if not self._vv_only:
            self._dd0i = cs.Interpolater(ra, dd0)
            self._dv0i = cs.Interpolater(ra, dv0)
            self._dv2i = cs.Interpolater(ra, dv2)

        self._cached = True


    def gen_cache(self, fname = None, rmin = 1e-3, rmax = 1e4, rnum = 1000):
        r"""Generate the cache.

        Calculate a table of the integrals required for the
        correlation functions, and save them to a named file (if
        given).

        Parameters
        ----------
        fname : filename, optional
            The file to save the cache into. If not set, the cache is
            generated but not saved.
        rmin : scalar
            The minimum r-value at which to generate the integrals (in
            h^{-1} Mpc)
        rmax : scalar
            The maximum r-value at which to generate the integrals (in
            h^{-1} Mpc)
        rnum : integer
            The number of points to generate (using a log spacing).
        """


        ra  = np.logspace(np.log10(rmin), np.log10(rmax), rnum)

        vv0 = _integrate(ra, 0, self.ps_vv)
        vv2 = _integrate(ra, 2, self.ps_vv)
        vv4 = _integrate(ra, 4, self.ps_vv)

        if not self._vv_only:
            dd0 = _integrate(ra, 0, self.ps_dd)
            dv0 = _integrate(ra, 0, self.ps_dv)
            dv2 = _integrate(ra, 2, self.ps_dv)

        if fname and not exists(fname):
            if self._vv_only:
                np.savetxt(fname, np.dstack([ra, vv0, vv2, vv4])[0])
            else:
                np.savetxt(fname, np.dstack([ra, vv0, vv2, vv4, dd0, dv0, dv2])[0])

        self._cached = True

        self._vv0i = cs.Interpolater(ra, vv0)
        self._vv2i = cs.Interpolater(ra, vv2)
        self._vv4i = cs.Interpolater(ra, vv4)

        if not self._vv_only:
            self._dd0i = cs.Interpolater(ra, dd0)
            self._dv0i = cs.Interpolater(ra, dv0)
            self._dv2i = cs.Interpolater(ra, dv2)


    def bias_z(self, z):
        r"""The linear bias at redshift z.

        The bias relative to the matter as a function of
        redshift. In this simple version the bias is assumed
        constant. Inherit, and override to use a more complicated
        model.

        Parameters
        ----------
        z : array_like
            The redshift to calculate at.

        Returns
        -------
        bias : array_like
            The bias at `z`.
        """
        return self.bias * np.ones_like(z)


    def growth_factor(self, z):
        r"""The growth factor D_+ as a function of redshift.

        The linear growth factor at a particular
        redshift, defined as:
        .. math:: \delta(k; z) = D_+(z; z_0) \delta(k; z_0)

        Normalisation can be arbitrary. For the moment
        assume that \Omega_m ~ 1, and thus the growth is linear in the
        scale factor.

        Parameters
        ----------
        z : array_like
            The redshift to calculate at.

        Returns
        -------
        growth_factor : array_like
            The growth factor at `z`.
        """
        return 1.0 / (1.0 + z)


    def growth_rate(self, z):
        r"""The growth rate f as a function of redshift.

        The linear growth rate at a particular redshift defined as:
        .. math:: f = \frac{d\ln{D_+}}{d\ln{a}}

        For the
        moment assume that \Omega_m ~ 1, and thus the growth rate is
        unity.

        Parameters
        ----------
        z : array_like
            The redshift to calculate at.

        Returns
        -------
        growth_rate : array_like
            The growth factor at `z`.
        """
        return 1.0 * np.ones_like(z)

    def prefactor(self, z):
        r"""An arbitrary scaling multiplying on each perturbation.

        This factor can be redshift dependent. It results in scaling
        the entire correlation function by prefactor(z1) *
        prefactor(z2).

        Parameters
        ----------
         z : array_like
            The redshift to calculate at.

        Returns
        -------
        prefactor : array_like
            The prefactor at `z`.
        """

        return 1.0 * np.ones_like(z)



    def mean(self, z):
        r"""Mean value of the field at a given redshift.

        Parameters
        ----------
        z : array_like
            redshift to calculate at.

        Returns
        -------
        mean : array_like
            the mean value of the field at each redshift.
        """
        return np.ones_like(z) * 0.0

    _sigma_v = 0.0
    def sigma_v(self, z):
        """Return the pairwise velocity dispersion at a given redshift.
        This is stored internally as `self._sigma_v` in units of km/s
        Note that e.g. WiggleZ reports sigma_v in h km/s
        """
        print "using sigma_v (km/s): " + repr(self._sigma_v)
        sigma_v_hinvMpc = (self._sigma_v / 100.)
        return np.ones_like(z) * sigma_v_hinvMpc


    def velocity_damping(self, kpar):
        """The velocity damping term for the non-linear power spectrum.
        """
        return (1.0 + (kpar * self.sigma_v(self.ps_redshift))**2.)**-1.


    def _realisation_dv(self, d, n):
        """Generate the density and line of sight velocity fields in a
        3d cube.
        """

        if not self._vv_only:
            raise Exception("Doesn't work for independent fields, I need to think a bit more first.")

        def psv(karray):
            """Assume k0 is line of sight"""
            k = (karray**2).sum(axis=3)**0.5
            return self.ps_vv(k) * self.velocity_damping(karray[..., 0])

        # Generate an underlying random field realisation of the
        # matter distribution.
        rfv = RandomField(npix = n, wsize = d)
        rfv.powerspectrum = psv

        vf0 = rfv.getfield()

        # Construct an array of \mu^2 for each Fourier mode.
        spacing = rfv._w / rfv._n
        kvec = fftutil.rfftfreqn(rfv._n, spacing / (2*math.pi))
        mu2arr = kvec[...,0]**2 / (kvec**2).sum(axis=3)
        mu2arr.flat[0] = 0.0
        del kvec

        df = vf0

        # Construct the line of sight velocity field.
        vf = np.fft.irfftn(mu2arr * np.fft.rfftn(vf0))

        #return (df, vf, rfv, kvec)
        return (df, vf) #, rfv)



    def realisation(self, z1, z2, thetax, thetay, numz, numx, numy, zspace = True):
        r"""Simulate a redshift-space volume.

        Generates a 3D (angle-angle-redshift) volume from the given
        power spectrum. Currently only works with simply biased power
        spectra (i.e. vv_only). This routine uses a flat sky
        approximation, and so becomes inaccurate when a large volume
        of the sky is simulated.

        Parameters
        ----------
        z1, z2 : scalar
            Lower and upper redshifts of the box.
        thetax, thetay : scalar
            The angular size (in degrees) of the box.
        numz : integer
            The number of bins in redshift.
        numx, numy : integer
            The number of angular pixels along each side.
        zspace : boolean, optional
            If True (default) redshift bins are equally spaced in
            redshift. Otherwise space equally in the scale factor
            (useful for generating an equal range in frequency).

        Returns
        -------
        cube : np.ndarray
            The volume cube.

        """
        d1 = self.cosmology.proper_distance(z1)
        d2 = self.cosmology.proper_distance(z2)
        c1 = self.cosmology.comoving_distance(z1)
        c2 = self.cosmology.comoving_distance(z2)

        # Make cube pixelisation finer, such that angular cube will
        # have sufficient resolution on the closest face.
        d = np.array([c2-c1, thetax * d2 * units.degree, thetay * d2 * units.degree])
        #n = np.array([numz, int(c2 / c1 * numx), int(c2 / c1 * numy)])
        n = np.array([numz, int(d2 / d1 * numx), int(d2 / d1 * numy)])

        # Enlarge cube size by 1 in each dimension, so raytraced cube
        # sits exactly within the gridded points.
        d = d * (n + 1) / n
        n = n + 1

        print "Generating cube: %f x %f x %f Mpc^3" % (d[0], d[1], d[2])

        cube = self._realisation_dv(d, n)


        # Construct an array of the redshifts on each slice of the cube.
        comoving_inv = inverse_approx(self.cosmology.comoving_distance, z1, z2)
        da = np.linspace(c1, c2, n[0], endpoint=True)
        za = comoving_inv(da)

        # Calculate the bias and growth factors for each slice of the cube.
        mz = self.mean(za)
        bz = self.bias_z(za)
        fz = self.growth_rate(za)
        Dz = self.growth_factor(za) / self.growth_factor(self.ps_redshift)
        pz = self.prefactor(za)

        # Construct the observable and velocity fields.
        df = cube[0] * (Dz * pz * bz)[:,np.newaxis,np.newaxis]
        vf = cube[1] * (Dz * pz * fz)[:,np.newaxis,np.newaxis]

        # Construct the redshift space cube.
        rsf = (df + vf) + mz[:,np.newaxis,np.newaxis]

        # Find the distances that correspond to a regular redshift
        # spacing (or regular spacing in a).
        if zspace:
            za = np.linspace(z1, z2, numz, endpoint = False)
        else:
            za = 1.0 / np.linspace(1.0 / (1+z2), 1.0 / (1+z1), numz, endpoint = False)[::-1] - 1.0

        da = self.cosmology.proper_distance(za)
        xa = self.cosmology.comoving_distance(za)

        # Construct the angular offsets into cube
        tx = np.linspace(-thetax / 2, thetax / 2, numx) * units.degree
        ty = np.linspace(-thetay / 2, thetay / 2, numy) * units.degree

        #tgridx, tgridy = np.meshgrid(tx, ty)
        tgridy, tgridx = np.meshgrid(ty, tx)
        tgrid2 = np.zeros((3, numx, numy))
        acube = np.zeros((numz, numx, numy))

        # Iterate over redshift slices, constructing the coordinates
        # and interpolating into the 3d cube.
        for i in range(numz):
            zi = (xa[i] - c1) / (c2-c1) * numz
            tgrid2[0,:,:] = zi
            tgrid2[1,:,:] = (tgridx * da[i])  / d[1] * numx + 0.5*n[1]
            tgrid2[2,:,:] = (tgridy * da[i])  / d[2] * numy + 0.5*n[2]

            #if(zi > numz - 2):
            acube[i,:,:] = scipy.ndimage.map_coordinates(rsf, tgrid2, order=2)


        return acube #, rsf


    def angular_powerspectrum(self, la, za1, za2):
        r"""The angular powerspectrum C_l(z1, z2).

        Parameters
        ----------
        l : array_like
            The multipole moments to return at.
        z1, z2 : array_like
            The redshift slices to correlate.

        Returns
        -------
        arr : array_like
            The values of C_l(z1, z2)
        """

        def _ps_single(l, z1, z2):
            if not self._vv_only:
                raise Exception("Only works for vv_only at the moment.")

            b1 = self.bias_z(z1)
            b2 = self.bias_z(z2)
            f1 = self.growth_rate(z1)
            f2 = self.growth_rate(z2)
            D1 = self.growth_factor(z1) / self.growth_factor(self.ps_redshift)
            D2 = self.growth_factor(z2) / self.growth_factor(self.ps_redshift)
            pf1 = self.prefactor(z1)
            pf2 = self.prefactor(z2)

            x1 = self.cosmology.comoving_distance(z1)
            x2 = self.cosmology.comoving_distance(z2)

            d1 = math.pi / (x1 + x2)

            def _int_lin(k):
                return (k**2 * self.ps_vv(k) * (b1 * sphbessel.jl(l, k*x1) - f1 * sphbessel.jl_d2(l, k*x1)) *
                       (b2 * sphbessel.jl(l, k*x2) - f2 * sphbessel.jl_d2(l, k*x2)))
                #return (k**2 * self.ps_vv(k) * b1 * b2 * sphbessel.jl(l, k*x1) * sphbessel.jl(l, k*x2))

            def _int_log(lk):
                k = np.exp(lk)
                return k*_int_lin(k)

            def _int_offset(k):
                return (_int_lin(k) + 4*_int_lin(k + d1) + 6*_int_lin(k + 2*d1)
                        + 4*_int_lin(k + 3*d1) + _int_lin(k + 4*d1)) / 16.0

            def _int_taper(k):
                return (15.0*_int_lin(k) + 11.0*_int_lin(k + d1) +
                        5.0*_int_lin(k + 2*d1) + _int_lin(k + 3*d1)) / 16.0

            def _integrator(f, a, b):
                #return quad(f, a, b, limit=1000, epsrel = 1e-7, full_output=(0 if _feedback else 1))[0]
                #return Integrate_Patterson(f, a, b, eps = 1e-7, abs=1e-10)[0]
                return integrate.chebyshev(f, a, b, epsrel = 1e-7, epsabs=1e-10)[0]
                #return romberg(f, a, b, eps = 1e-6)
            #full_output=(0 if _feedback else 1))[0]
            mink = 1e-2 * l / (x1 + x2)
            cutk = 2e0 * l / (x1 + x2)
            cutk2 = 20.0  * l / (x1 + x2)
            maxk = 1e2 * l / (x1 + x2)

            #ka = np.logspace(np.log(mink), np.log(maxk), 1000)

            i1 = _integrator(_int_log, np.log(mink), np.log(cutk) )
            #print i1
            i2 = _integrator(_int_taper, cutk, cutk + d1)
            #print i2
            i3 = _integrator(_int_offset, cutk, cutk2)

            i4 = _integrator(_int_offset, cutk2, maxk)
            #print i3
            print i1, i2, i3, i4, i1+i2+i3

            #return ka, _int_lin(ka)

            #cl = (_integrator(_int_log, np.log(mink), np.log(cutk) ) +
            #      _integrator(_int_lin, cutk, maxk))

            cl = (i1 + i2 + i3 + i4) * D1 * D2 * pf1 * pf2 * (2 / np.pi)

            return cl

        bobj = np.broadcast(la, za1, za2)

        if not bobj.shape:
            # Broadcast from scalars
            return _ps_single(la, za1, za2)
        else:
            # Broadcast from arrays
            cla = np.empty(bobj.shape)
            cla.flat = [ _ps_single(l, z1, z2) for (l, z1, z2) in bobj ]

            return cla


    def angular_powerspectrum_flat(self, la, za1, za2):
        r"""The angular powerspectrum C_l(z1, z2) in a flat-sky limit.

        Parameters
        ----------
        l : array_like
            The multipole moments to return at.
        z1, z2 : array_like
            The redshift slices to correlate.

        Returns
        -------
        arr : array_like
            The values of C_l(z1, z2)
        """

        def _ps_single(l, z1, z2):

            b1 = self.bias_z(z1)
            b2 = self.bias_z(z2)
            f1 = self.growth_rate(z1)
            f2 = self.growth_rate(z2)
            D1 = self.growth_factor(z1) / self.growth_factor(self.ps_redshift)
            D2 = self.growth_factor(z2) / self.growth_factor(self.ps_redshift)
            pf1 = self.prefactor(z1)
            pf2 = self.prefactor(z2)

            x1 = self.cosmology.comoving_distance(z1)
            x2 = self.cosmology.comoving_distance(z2)

            dx = x1 - x2
            xc = 0.5*(x1 + x2)

            def _int_lin(kp):
                return np.cos(kp * dx) * self.powerspectrum(kp, l / xc, z1, z2) / (xc**2 * np.pi)

            def _int_log(lkp):
                kp = np.exp(lkp)
                return _int_lin(kp) * kp

            def _integrator(f, a, b):
                #return quad(f, a, b, limit=1000, epsrel = 1e-10, full_output=(0 if _feedback else 1))[0]
                #return Integrate_Patterson(f, a, b, eps = 1e-7, abs=1e-10)[0]
                i = integrate.chebyshev(f, a, b, epsrel = 1e-12, epsabs=1e-16)
                #i = integrate.romberg(f, a, b, epsrel = 1e-15)
                #print i
                return i
                #return romberg(f, a, b, eps = 1e-8)

            i1 = _integrator(_int_lin, 0.0, 2e1)

            return i1


        bobj = np.broadcast(la, za1, za2)

        if not bobj.shape:
            # Broadcast from scalars
            return _ps_single(la, za1, za2)
        else:
            # Broadcast from arrays
            cla = np.empty(bobj.shape)
            cla.flat = [ _ps_single(l, z1, z2) for (l, z1, z2) in bobj ]

            return cla


    _aps_cache = False

    def save_fft_cache(self, fname):
        if not self._aps_cache:
            self.angular_powerspectrum_fft(np.array([100]), np.array([1.0]), np.array([1.0]))

        np.savez(fname, dd=self._aps_dd, dv=self._aps_dv, vv=self._aps_vv)


    def load_fft_cache(self, fname):

        a = np.load(fname)

        self._aps_dd = a['dd']
        self._aps_dv = a['dv']
        self._aps_vv = a['vv']

        self._aps_cache = True


    def angular_powerspectrum_fft(self, la, za1, za2):

        kperpmin = 1e-4
        kperpmax = 40.0
        nkperp = 500
        kparmax = 40.0
        nkpar = 16384

        if not self._aps_cache:

            kperp = np.logspace(np.log10(kperpmin), np.log10(kperpmax), nkperp)[:,np.newaxis]

            #kpar = (np.fft.fftfreq(nkpar) * 2 * nkpar)[np.newaxis,:]
            kpar = np.linspace(0, kparmax, nkpar)[np.newaxis,:]

            k = (kpar**2 + kperp**2)**0.5
            mu2 = kpar**2 / k**2

            self._dd = self.ps_vv(k)
            self._dv = self._dd * mu2
            self._vv = self._dd * mu2**2

            self._aps_dd = scipy.fftpack.dct(self._dd, type=1) * kparmax / (2*nkpar)
            self._aps_dv = scipy.fftpack.dct(self._dv, type=1) * kparmax / (2*nkpar)
            self._aps_vv = scipy.fftpack.dct(self._vv, type=1) * kparmax / (2*nkpar)

            self._aps_cache = True

        xa1 = self.cosmology.comoving_distance(za1)
        xa2 = self.cosmology.comoving_distance(za2)

        b1 = self.bias_z(za1)
        b2 = self.bias_z(za2)
        f1 = self.growth_rate(za1)
        f2 = self.growth_rate(za2)
        D1 = self.growth_factor(za1) / self.growth_factor(self.ps_redshift)
        D2 = self.growth_factor(za2) / self.growth_factor(self.ps_redshift)
        pf1 = self.prefactor(za1)
        pf2 = self.prefactor(za2)

        xc = 0.5 * (xa1 + xa2)

        kperp = la / xc

        rpar = np.abs(xa2 - xa1)

        coords = np.empty((2,) + kperp.shape)
        coords[0,...] = np.log10(kperp / kperpmin) / np.log10(kperpmax / kperpmin) * (nkperp-1)
        coords[1,...] = rpar / (math.pi / kparmax)

        psdd = scipy.ndimage.map_coordinates(self._aps_dd, coords, order=2)
        psdv = scipy.ndimage.map_coordinates(self._aps_dv, coords, order=2)
        psvv = scipy.ndimage.map_coordinates(self._aps_vv, coords, order=2)

        return D1*D2*pf1*pf2*(b1*b2*psdd + (f1*b2 + f2*b1)*psdv + f1*f2*psvv) / (xc**2 * np.pi)


def inverse_approx(f, x1, x2):
    r"""Generate the inverse function on the interval x1 to x2.

    Periodically sample a function and use interpolation to construct
    its inverse. Function must be monotonic on the given interval.

    Parameters
    ----------
    f : callable
        The function to invert, must accept a single argument.
    x1, x2 : scalar
        The lower and upper bounds of the interval on which to
        construct the inverse.

    Returns
    -------
    inv : cubicspline.Interpolater
        A callable function holding the inverse.
    """

    xa = np.linspace(x1, x2, 1000)
    fa = f(xa)

    return cs.Interpolater(fa, xa)


@np.vectorize
def _pl(l, x):
    return scipy.special.lpn(l, x)[0][l]


def _integrand_linear(k, r, l, psfunc):

    return 1.0 / (2*math.pi**2) * k**2 * sphbessel.jl(l, k*r) * psfunc(k)


def _integrand_log(lk, r, l, psfunc):

    k = np.exp(lk)
    return k * _integrand_linear(k, r, l, psfunc)
    #return 1.0 / (2*math.pi**2) * k**3 * sphbessel.jl(l, k*r) * psfunc(k)


def _integrand_offset(k, *args):

    d1 = math.fabs(math.pi / args[0])
    return (_integrand_linear(k, *args) + 4*_integrand_linear(k + d1, *args) +
            6*_integrand_linear(k + 2*d1, *args) + 4*_integrand_linear(k + 3*d1, *args) +
            _integrand_linear(k + 4*d1, *args)) / 16.0


def _integrand_taper(k, *args):

    d1 = math.fabs(math.pi / args[0])
    return (15.0*_integrand_linear(k, *args) + 11.0*_integrand_linear(k + d1, *args) +
            5.0*_integrand_linear(k + 2*d1, *args) + _integrand_linear(k + 3*d1, *args)) / 16.0


@np.vectorize
def _integrate(r, l, psfunc):

    def _int(f, a, b, args=()):
        #return quad(f, a, b, args=args, limit=1000, epsrel = 1e-7,
        #            full_output=(0 if _feedback else 1))[0]
        #return Integrate_Patterson(f, a, b, eps = 1e-7, abs=1e-10, args=args)[0]
        return integrate.chebyshev(f, a, b, epsrel = 1e-7, epsabs=1e-9, args=args)
    mink = 1e-4
    maxk = 1e3
    cutk = 5e1
    d = math.pi / r

    argv = (r, l, psfunc)

    r1 = _int(_integrand_log, math.log(mink*d), math.log(cutk*d), args = argv)
    r2 = _int(_integrand_taper, cutk*d, (cutk+1.0)*d, args = argv)
    r3 = _int(_integrand_offset, cutk*d, maxk*d, args = argv)

    if _feedback:
        print r1, r2, r3

    return r1+r3+r3
