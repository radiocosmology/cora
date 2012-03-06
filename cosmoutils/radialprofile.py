"""Calculate the radial average of an image. Code stolen from:

http://www.astrobetter.com/wiki/tiki-index.php?page=python_radial_profiles

and modified to crudely have a controllable bin width, and work in n-dimensions.
"""

import numpy as np




def azimuthalAverage(image, center=None, bw=3):
    """
    Calculate the azimuthally averaged radial profile.
    
    Works with n-dimensional images.
    
    Parameters
    ----------
    image : np.ndarray
        The image.
    center : array_like, optional
        The [x,y] pixel coordinates used as the center. The default is
        None, which then uses the center of the image (including
        fractional pixels).

    Returns
    -------
    bl : np.ndarray
        The lower limit of the bin radius.
    radial_prof : np.ndarray
        The radial averaged profile.
    
    """
    # Calculate the indices from the image
    ia = np.indices(image.shape)
    ia = np.rollaxis(ia, 0, ia.ndim)

    if not center:
        center = (np.array(image.shape) - 1.0) / 2.0

    r = np.sum((ia - center)**2, axis=-1)**0.5

    # Get sorted radii
    #maxr = np.array([image.shape[0] - center[0], image.shape[1] - center[1]]).min()
    maxr = (np.array(image.shape) - center).min()

    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    maxind = np.searchsorted(r_sorted, maxr+0.00001)
    r_sorted = r_sorted[:maxind]
    i_sorted = i_sorted[:maxind]

    numbins = int(maxr / bw)
    maxr = numbins * bw

    bn = np.linspace(0.0, maxr, numbins+1)

    bc = 0.5*(bn[1:] + bn[:-1])
    bl = bn[:-1]

    # Get the integer part of the radii (bin size = 1)
    #r_int = r_sorted.astype(int)
    r_int = np.digitize(r_sorted, bn)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    #pdb.set_trace()
    rind = np.where(deltar)[0]       # location of changed radius
    rind = np.insert(rind, 0, -1)
    nr = rind[1:] - rind[:-1]        # number of radius bin


    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=image.dtype)
    csim = np.insert(csim, 0, 0.0)
    tbin = csim[rind[1:]+1] - csim[rind[:-1]+1]

    radial_prof = tbin / nr

    return bl, radial_prof
