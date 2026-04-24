import healpy as hp
import numpy as np


def map_derivatives_curved(m):
    """
    Compute first and second derivatives of a HEALPix map on the sphere.

    Parameters
    ----------
    m : numpy.ndarray
        1D array of map (pixel) values in HEALPix ordering.

    Returns
    -------
    der_theta : numpy.ndarray
        First derivative of the map with respect to theta.
    der_phi : numpy.ndarray
        First derivative of the map with respect to phi.
    der2_theta2 : numpy.ndarray
        Second derivative with respect to theta.
    der2_phi2 : numpy.ndarray
        Second derivative with respect to phi.
    der2_theta_phi : numpy.ndarray
        Mixed second derivative with respect to theta and phi.
    """
    npix = m.size
    nside = hp.npix2nside(npix)
    
    # take the spherical Fourier transform of the map
    alm = hp.map2alm(m,iter=1)

    # convert back to real space and also get the first derivatives
    _, der_theta_map, der_phi_map = hp.alm2map_der1(alm, nside)

    # Fourier transform the first derivatives
    der_theta_alm = hp.map2alm(der_theta_map,iter=1)
    der_phi_alm = hp.map2alm(der_phi_map,iter=1)

    # convert the first derivatives back to real space and also get the second derivatives
    _, der2_theta2_map, der2_theta_phi_map = hp.alm2map_der1(der_theta_alm, nside)
    _, _, der2_phi2_map = hp.alm2map_der1(der_phi_alm, nside)

    # return all
    return [der_theta_map, der_phi_map, der2_theta2_map, der2_phi2_map, der2_theta_phi_map]



def map_derivatives_flat(m, pixelsize):
        
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
    
    # calculate second derivatives
    dxx,dxy = np.gradient(dx, pixelsize, axis = (0,1))
    dyy = np.gradient(dy, pixelsize, axis=1)
    
    return [dx,dy,dxx,dxy,dyy]



def V_012(m, v=None, geometry="curved", pixelsize=None, thr_ct=20, t_min=-3, t_max=3):
    """
    Compute Minkowski Functionals V0, V1, and V2 for a scalar field on either Euclidean (flat-sky) or spherical (HEALPix) geometries.

    This function evaluates morphological statistics of a 2D field by computing:
        - V0: area 
        - V1: perimeter
        - V2: curvature

    Parameters
    ----------
    m : numpy.ndarray
        Input scalar field.
        - Flat case: 2D array (image/map).
        - Curved case: 1D HEALPix map (RING ordering).

    v : numpy.ndarray, optional
        Array of threshold values at which Minkowski functionals are evaluated.
        If None, thresholds are generated automatically using the standard deviation of the map.

    geometry : str, default="curved"
        Geometry of the input field:
        - "flat": Euclidean 2D grid using finite differences
        - "curved": spherical HEALPix map using harmonic derivatives

    pixelsize : float, optional
        Pixel size in radians (required if geometry="flat").

    thr_ct : int, default=20
        Number of threshold bins (used only if v=None).

    t_min : float, default=-3
        Minimum threshold in units of map standard deviation (used only if v=None).

    t_max : float, default=3
        Maximum threshold in units of map standard deviation (used only if v=None).

    Returns
    -------
    V0 : numpy.ndarray
        Fraction of pixels above each threshold.

    V1 : numpy.ndarray
        Normalised boundary length functional.

    V2 : numpy.ndarray
        Normalised curvature-related functional.

    Notes
    -----
    - It is recommended to use the same threshold range if you are comparing MFs for different scenarios to a base truth.
    - HEALPix transforms in curved mode can be computationally expensive for high NSIDE maps.
    - The curvature estimator (V2) assumes a smooth field; noisy maps may require smoothing.

    """
    # calculate first and second derivatives of the field
    if geometry == "flat":
        if pixelsize is None:
            raise ValueError("pixel size must be specified")
        dx, dy, dxx, dyy, dxy = map_derivatives_flat(m, pixelsize)
        m = m.flatten()
        dx = dx.flatten()
        dy = dy.flatten()
        dxx = dxx.flatten()
        dyy = dyy.flatten()
        dxy = dxy.flatten()
    
    elif geometry == "curved":
        dx, dy, dxx, dyy, dxy = map_derivatives_curved(m)
        
    else:
        raise ValueError("geometry must be 'flat' or 'curved'")

    # compute per-pixel quantities for V1 and V2
    sq = np.sqrt(dx*dx + dy*dy)
    grad2 = dx*dx + dy*dy
    grad2 = np.maximum(grad2, 1e-16)
    frac = (2*dx*dy*dxy - (dx**2)*dyy - (dy**2)*dxx) / grad2
    
   # linear binning based on thresholds
    if v is None:
        k_std = np.std(m)
        v = np.linspace(t_min * k_std, t_max * k_std, thr_ct)
        print('Using default thresholds: -3*map_std to 3*map_std with 20 bins.')
    
    vmin = v.min()                                  
    vmax = v.max()  
    thr_ct = len(v)                     
    vspace = (vmax - vmin) / (thr_ct - 1)
    indices = np.floor((m - vmin) / vspace)     
    valid = (indices >= 0) & (indices < thr_ct)
    
    # sum V1 and V2 per threshold
    V1 = np.bincount(indices[valid].astype(int), weights=sq[valid], minlength=thr_ct)
    V2 = np.bincount(indices[valid].astype(int), weights=frac[valid], minlength=thr_ct)

    # normalise
    V0 = (m[None, :] > v[:, None]).sum(axis=1) / m.size
    V1 = V1 / (4 * m.size)
    V2 = V2 / (2 * np.pi * m.size)
    
    return V0, V1, V2


