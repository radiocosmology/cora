import numpy as np

from utils import units


class Map2d(object):
    r"""A 2-d sky map.
    
    Attributes
    ----------
    x_width, y_width : float
        Angular size along each axis (in degrees).
    x_num, y_num : int
        Number of pixels along each angular axis.

        
    """
    x_width = 5.0
    y_width = 5.0

    x_num = 128
    y_num = 128

    
    @classmethod
    def like_map(cls, mapobj, *args, **kwargs):
        r"""Create a Map2d (or subclassed) object the same shape as a given object.
        """
        
        c = cls(*args, **kwargs)
        c.x_width = mapobj.x_width
        c.y_width = mapobj.y_width
        c.x_num = mapobj.x_num
        c.y_num = mapobj.y_num

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


class Map3d(object):
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
        
    """
    x_width = 5.0
    y_width = 5.0
    
    x_num = 128
    y_num = 128

    nu_num = 128

    nu_lower = 500.0
    nu_upper = 900.0

    @classmethod
    def like_map(cls, mapobj, *args, **kwargs):
        r"""Create a Map3d (or subclassed) object the same shape as a given object.
        """
        c = cls(*args, **kwargs)
        c.x_width = mapobj.x_width
        c.y_width = mapobj.y_width
        c.nu_upper = mapobj.nu_upper
        c.nu_lower = mapobj.nu_lower
        c.x_num = mapobj.x_num
        c.y_num = mapobj.y_num
        c.nu_num = mapobj.nu_num

        return c

    def _width_array(self):
        return np.array([self.nu_upper - self.nu_lower, self.x_width*units.degree, self.widthy*units.degree], dtype=np.float64)

    def _num_array(self):
        return np.array([self.nu_num, self.x_num, self.y_num], dtype=np.int)


    @property
    def x_pixels(self):
        return ((np.arange(self.x_num) + 0.5) * (self.x_width / self.x_num))

    @property
    def y_pixels(self):
        return ((np.arange(self.y_num) + 0.5) * (self.y_width / self.y_num))

    @property
    def nu_pixels(self):
        return (self.nu_lower + (np.arange(self.nu_num) + 0.5) * ((self.nu_upper - self.nu_lower) / self.nu_num))


