import numpy as np

from utils import radialprofile


def make_nd_at_ind(arr, nd, ind):
    sl = ([np.newaxis]*ind) + [slice(None)] + ([np.newaxis]) * (nd - ind - 1)
    return arr[sl]


def blackman_nd(shape):
    
    nd = len(shape)
    b = np.ones(shape)
    
    for i, l in enumerate(shape):
        b *= make_nd_at_ind(np.blackman(l), nd, i)

    return b


def ps(arr, width = None, bw = 3, kmodes = True, window = True):
    """Calculate the radially average power spectrum of an nD fields.
    
    The array must have the same length (physically
    and in pixel number) along each axis.
    
    Parameters
    ----------
    arr : np.ndarray
        The cube to calculate the power spectrum of.
    width : scalar, optional
        The dimensional length of each cube axis (each axis must be
        the same length). If None (default), assume that each side is
        of length 1.
    bw : integer, optional
        The width of the binning in the radial direction. Default is 3.
    kmodes : boolean, optional
        If True (default) return the k-values of the binning.
    window : boolean, optional
        If True (default), apply a Blackman window to account for
        the non-periodicity.

    Returns
    -------
    pps : np.ndarray
        A 1D array containing the power spectrum P(k)
    kvals: np.ndarray
        1D arrays containing the lower bounds of the bins in k.
    """
    
    return crossps(arr, None, width=width, bw=bw, kmodes=kmodes, window=window)


def crossps(arr1, arr2, width = None, bw = 3, kmodes = True, window = True):
    """Calculate the radially average cross-power spectrum of a two nD fields.
    
    The arrays must be identical and have the same length (physically
    and in pixel number) along each axis.
    
    Parameters
    ----------
    arr1, arr2 : np.ndarray
        The cubes to calculate the cross-power spectrum of. If arr2 is
        None then return the standard (auto-)power spectrum.
    width : scalar, optional
        The dimensional length of each cube axis (each axis must be
        the same length). If None (default), assume that each side is
        of length 1.
    bw : integer, optional
        The width of the binning in the radial direction. Default is 3.
    kmodes : boolean, optional
        If True (default) return the k-values of the binning.
    window : boolean, optional
        If True (default), apply a Blackman window to account for
        the non-periodicity.

    Returns
    -------
    xps : np.ndarray
        A 1D array containing the cross-power spectrum P(k)
    kvals: np.ndarray
        1D arrays containing the lower bounds of the bins in k.
    """
    
    if window:
        w = blackman_nd(arr1.shape)

        arr1 = arr1 * w
        if arr2 != None:
            arr2 = arr2 * w

    if arr2 != None:
        fq1 = np.fft.fftshift(np.fft.fftn(arr1))
        fq2 = np.fft.fftshift(np.fft.fftn(arr2))
        fq = fq1 * fq2.conj()
    else:
        fq = np.abs(np.fft.fftshift(np.fft.fftn(arr1)))**2

    kv, ps = radialprofile.azimuthalAverage_nd(fq)


    if width != None:
        tp *= np.array(width)**arr1.ndim
        kv /= width

    if arr2 == None:
        ps = np.abs(ps)

    if kmodes:
        return ps, kv
    else:
        return ps
    


def ps_azimuth(img, width = None, bwperp = 3, bwpar = 3, kmodes = True, window = True):
    r"""Calculate the 2-D azimuthally averaged power spectrum of a 3D field.
    
    The azimuthal axis is assumed to be axis 0. Generally, this is useful when
    looking at redshift distorted fields, as it gives P(kpar, kperp).

    Parameters
    ----------
    img : np.ndarray (3D)
        The cube to calculate the power spectrum of.
    width : array_like, optional
        The dimensional length of each cube axis.
    bwpar, bwperp : integer, optional
        The width of the bins in the radial and transverse
        directions. If None (default), assume that each side is of
        length 1.
    kmodes : boolean, optional
        If True (default) return the k-values of the binning in the
        parallel and perpendicular directions.
    window : boolean, optional
        If True (default), apply a 3D Blackman window to account for
        the non-periodicity.

    Returns
    -------
    ps : np.ndarray
        A 2D array containing the power spectrum P(kpar, kperp)
    kpar, kperp : np.ndarray
        1D arrays containing the lower bounds of the bins in each
        direction,
    """
    
    return (crossps_azimuth(img, None, width = width, bwperp = bwperp,
                           bwpar = bwpar, kmodes = kmodes, window = window))


def crossps_azimuth(img1, img2, width = None, bwperp = 3, bwpar = 3, kmodes = True, window = True):
    """Calculate the 2-D azimuthally averaged cross-power spectrum of a two 3D fields.
    
    The azimuthal axis is assumed to be axis 0. Generally, this is useful when
    looking at redshift distorted fields, as it gives P(kpar, kperp).

    Parameters
    ----------
    img1, img2 : np.ndarray (3D)
        The cubes to calculate the cross-power spectrum of. If img2 is
        None then return the standard (auto-)power spectrum.
    width : array_like, optional
        The dimensional length of each cube axis.
    bwpar, bwperp : integer, optional
        The width of the bins in the radial and transverse
        directions. If None (default), assume that each side is of
        length 1.
    kmodes : boolean, optional
        If True (default) return the k-values of the binning in the
        parallel and perpendicular directions.
    window : boolean, optional
        If True (default), apply a 3D Blackman window to account for
        the non-periodicity.

    Returns
    -------
    xps : np.ndarray
        A 2D array containing the cross-power spectrum P(kpar, kperp)
    kpar, kperp : np.ndarray
        1D arrays containing the lower bounds of the bins in each
        direction,
    """
    
    if window:
        w0 = np.blackman(img1.shape[0])[:,np.newaxis,np.newaxis]
        w1 = np.blackman(img1.shape[1])[np.newaxis,:,np.newaxis]
        w2 = np.blackman(img1.shape[2])[np.newaxis,np.newaxis,:]

        img1 = img1 * w0 * w1 * w2
        if img2 != None:
            img2 = img2 * w0 * w1 * w2

    if img2 != None:
        fq1 = np.fft.fftshift(np.fft.fftn(img1))
        fq2 = np.fft.fftshift(np.fft.fftn(img2))
        fq = fq1 * fq2.conj()
    else:
        fq = np.abs(np.fft.fftshift(np.fft.fftn(img1)))**2

    hpar = fq.shape[0] / 2
    fqh = fq[hpar:]

    binsperp = (fq.shape[1] / 2 -1) / bwperp
    binspar = (fqh.shape[0] - 1)/ bwpar

    ta = np.zeros((binspar, fq.shape[1], fq.shape[2]), dtype=np.complex128)
    tp = np.zeros((binspar, binsperp), dtype=np.complex128)

    for i in range(binspar):
        sl = np.mean(fqh[bwpar*i:bwpar*(i+1),:,:], axis=0)
        ta[i,:,:] = sl

        kperp, tp[i,:] = radialprofile.azimuthalAverage(sl, bw = bwperp)

    kpar = (np.arange(binspar)) * bwpar * 2*np.pi
    kperp *= 2*np.pi
    tp /= np.array(fq.shape).prod()**2
    if width != None:
        tp *= np.array(width).prod()
        kpar /= width[0]
        kperp /= width[1]

    if img2 == None:
        tp = np.abs(tp)

    #return ta, tp, kpar, kperp
    if kmodes:
        return tp, kpar, kperp
    else:
        return tp
    

    

    
