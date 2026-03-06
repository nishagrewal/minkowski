import numpy as np


def map_derivatives(m, pixelsize):
        
    """
    Calculates the first and second derivatives of a 2D map

    Parameters
    ----------
    m : 2D numpy array
        2D map to calculate derivatives of
    pixelsize : float
        Pixel size of the map in radians

    Returns
    -------
    dx : 2D numpy array
        First derivative in x direction
    dy : 2D numpy array
        First derivative in y direction
    dxx : 2D numpy array
        Second derivative in x direction
    dxy : 2D numpy array
        Mixed derivative of x and y
    dyy : 2D numpy array
        Second derivative in y direction
    """
    
    # calculate first derivatives
    dx,dy = np.gradient(m, pixelsize, axis = (0,1))
    
    # caluclate second derivatives
    dxx,dxy = np.gradient(dx, pixelsize, axis = (0,1))
    dyy = np.gradient(dy, pixelsize,axis=1)
    
    return [dx,dy,dxx,dxy,dyy]


def V_012(k, thr_ct, k_std, pixelsize):

    """
    Calculates the Minkowski Functionals (MF) V0, V1, and V2 for a 2D flat sky square map

    Parameters
    ----------
    k : 2D numpy array
        2D map to calculate MFs of
    thr_ct : int
        Number of threshold values to calculate
    pixelsize : float
        Pixel size of the map in radians
    
    Returns
    -------
    V0 : numpy.ndarray
        Cumulative fraction of pixels above each threshold (normalised).
    V1 : numpy.ndarray
        Perimeter-related Minkowski functional (normalised).
    V2 : numpy.ndarray
        Curvature-related Minkowski functional (normalised).
    """
    
    # calculate derivatives
    kx,ky,kxx,kxy,kyy = map_derivatives(k, pixelsize)
  
    # define MF functions
    sq = np.sqrt(kx**2 + ky**2)
    frac = (2*kx*ky*kxy - (kx**2)*kyy - (ky**2)*kxx)/(kx**2 + ky**2)
        
    # get threshold array bin siz
    vmin = v.min()     
    vmax = v.max()                           
    vspace = (vmax - vmin) / (thr_ct - 1)            

    # flatten arrays for MF calculation
    k = k.flatten()
    sq = sq.flatten()
    frac = frac.flatten()

    # get threshold bin index for each pixel
    indices = np.floor((k - vmin) / vspace)     
    valid = (indices >= 0) & (indices < thr_ct)
    V1 = np.bincount(indices[valid].astype(int), weights=sq[valid], minlength=thr_ct)
    V2 = np.bincount(indices[valid].astype(int), weights=frac[valid], minlength=thr_ct)
    
    # normalise
    V0 = (k[None, :] > v[:, None]).sum(axis=1) / k.size
    V1 = V1 / (4 * k.size)
    V2 = V2 / (2 * np.pi * k.size)
    
    return V0, V1, V2



