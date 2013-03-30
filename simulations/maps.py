import numpy as np

from cosmoutils import units

import skysim


class Map2d(object):
    r"""A 2-d sky map.
    
    Attributes
    ----------
    x_width, y_width : float
        Angular size along each axis (in degrees).
    x_num, y_num : int
        Number of pixels along each angular axis.
    nside : int
        Resolution of Healpix map (must be power of 2).
    """
    x_width = 5.0
    y_width = 5.0

    x_num = 128
    y_num = 128

    _nside = 128
    
    @classmethod
    def like_map(cls, mapobj, *args, **kwargs):
        r"""Create a Map2d (or subclassed) object the same shape as a given object.
        """
        
        c = cls(*args, **kwargs)
        c.x_width = mapobj.x_width
        c.y_width = mapobj.y_width
        c.x_num = mapobj.x_num
        c.y_num = mapobj.y_num
        c._nside = mapobj._nside

        return c

    def _width_array(self):
        return np.array([self.x_width, self.y_width], dtype=np.float64)*units.degree

    def _num_array(self):
        return np.array([self.x_num, self.y_num], dtype=np.int)

    @property
    def x_pixels(self):
        return ((np.arange(self.x_num) + 0.5) * (self.x_width / self.x_num))

    @property
    def y_pixels(self):
        return ((np.arange(self.y_num) + 0.5) * (self.y_width / self.y_num))

    @property
    def nside(self):
        """The resolution of the Healpix map."""
        return self._nside

    @nside.setter
    def nside(self, value):
        ns = int(value)
        lns = np.log2(ns)

        if int(lns) != lns or lns < 0:
            raise Exception("Not a valid value of nside.")

        self._nside = ns




class Map3d(Map2d):
    r"""A 3-d sky map.

    Attributes
    ----------
    x_width, y_width : float
        Angular size along each axis (in degrees).
    nu_upper, nu_lower : float
        Range of frequencies (in Mhz).
    x_num, y_num : int
        Number of pixels along each angular axis.
    nu_num : int
        Number of frequency bins.
    nside : int
        Resolution of Healpix map (must be power of 2).
        
    """
    nu_num = 128

    nu_lower = 500.0
    nu_upper = 900.0

    @classmethod
    def like_map(cls, mapobj, *args, **kwargs):
        r"""Create a Map3d (or subclassed) object the same shape as a given object.
        """
        #c = Map2d.like_map(cls, mapobj, *args, **kwargs)

        c = cls(*args, **kwargs)
        c.x_width = mapobj.x_width
        c.y_width = mapobj.y_width
        c.x_num = mapobj.x_num
        c.y_num = mapobj.y_num
        c._nside = mapobj._nside

        c.nu_upper = mapobj.nu_upper
        c.nu_lower = mapobj.nu_lower
        c.nu_num = mapobj.nu_num

        return c

    def _width_array(self):
        return np.array([self.nu_upper - self.nu_lower, self.x_width*units.degree, self.widthy*units.degree], dtype=np.float64)

    def _num_array(self):
        return np.array([self.nu_num, self.x_num, self.y_num], dtype=np.int)


    @property
    def nu_pixels(self):
        return (self.nu_lower + (np.arange(self.nu_num) + 0.5) * ((self.nu_upper - self.nu_lower) / self.nu_num))


    @classmethod
    def like_kiyo_map(cls, mapobj, *args, **kwargs):
        r"""Crease a Map3d (or subclassed) object based on one of kiyo's map
        objects
        """
        c = cls(*args, **kwargs)

        freq_axis = mapobj.get_axis('freq')
        ra_axis = mapobj.get_axis('ra')
        dec_axis = mapobj.get_axis('dec')

        ra_fact = sp.cos(sp.pi * mapobj.info['dec_centre'] / 180.0)
        c.x_width = (max(ra_axis) - min(ra_axis)) * ra_fact
        c.y_width = max(dec_axis) - min(dec_axis)
        (c.x_num, c.y_num) = (len(ra_axis), len(dec_axis))

        c.nu_lower = min(freq_axis)/1.e6
        c.nu_upper = max(freq_axis)/1.e6
        c.nu_num = len(freq_axis)

        print "Map3D: %dx%d field (%fx%f deg) from nu=%f to nu=%f (%d bins)" % \
                 (c.x_num, c.y_num, c.x_width, c.y_width,
                  c.nu_lower, c.nu_upper, c.nu_num)

        return c




class Sky3d(Map3d):



    def angular_powerspectrum(self, l, nu1, nu2):
        """C_l(nu1, nu2) for the given map."""

        raise Exception("Not implemented in base class.")


    def mean_nu(self, freq):
        return np.zeros_like(freq)



    def getfield(self):

        raise Exception("Not implemented in base class.")


    def getsky(self):

        lmax = 3 * self.nside - 1
        cla = skysim.clarray(self.angular_powerspectrum, lmax, self.nu_pixels)
        
        return (self.mean_nu(self.nu_pixels)[:, np.newaxis]
                + skysim.mkfullsky(cla, self.nside))


    def getpolsky(self):

        sky_I = self.getsky()

        sky_IQU = np.zeros((sky_I.shape[0], 3, sky_I.shape[1]), dtype=sky_I.dtype)

        sky_IQU[:, 0] = sky_I
        
        return sky_IQU


    def getalms(self, lmax):

        cla = skysim.clarray(self.angular_powerspectrum, lmax, self.nu_pixels)
        
        return skysim.mkfullsky(cla, self.nside, alms=True)




