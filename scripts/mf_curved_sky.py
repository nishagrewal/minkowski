import healpy as hp
import numpy as np


def map_derivatives(m):
    """
    Compute first and second derivatives of a HEALPix map on the sphere.

    Parameters
    ----------
    m : numpy.ndarray
        1D array of map values (pixel heights) in HEALPix ordering.

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
    
    # Take the spherical Fourier transform of the map
    alm = hp.map2alm(m,iter=1)

    # Convert back to real space and also get the first derivatives
    _, der_theta_map, der_phi_map = hp.alm2map_der1(alm, nside)

    # Fourier transform the first derivatives
    der_theta_alm = hp.map2alm(der_theta_map,iter=1)
    der_phi_alm = hp.map2alm(der_phi_map,iter=1)

    # Convert the first derivatives back to real space and also get the second derivatives
    _, der2_theta2_map, der2_theta_phi_map = hp.alm2map_der1(der_theta_alm, nside)
    _, _, der2_phi2_map = hp.alm2map_der1(der_phi_alm, nside)

    # return all
    return [der_theta_map, der_phi_map, der2_theta2_map, der2_phi2_map, der2_theta_phi_map]



def V_012_curved(v, k):
    """
    Calculate Minkowski Functionals V0, V1, and V2 for a curved sky healpy map.

    Parameters
    ----------
    v : numpy.ndarray
        1D array of threshold values. These define the levels at which the functionals are evaluated.
    k : numpy.ndarray
        1D array of map values (pixel heights).

    Returns
    -------
    V0 : numpy.ndarray
        Cumulative fraction of pixels above each threshold (normalised).
    V1 : numpy.ndarray
        Perimeter-related Minkowski functional (normalised).
    V2 : numpy.ndarray
        Curvature-related Minkowski functional (normalised).
    """

    # calculate first and second derivatives
    dx, dy, dxx, dyy, dxy = map_derivatives(k)
    
    # compute per-pixel quantities for V1 and V2
    sq = np.sqrt(dx**2 + dy**2)
    grad2 = dx**2 + dy**2
    grad2[grad2 == 0] = 1e-16  # avoid division by zero
    frac = (2*dx*dy*dxy - (dx**2)*dyy - (dy**2)*dxx) / grad2
        
    # linear binning based on thresholds
    vmin = v.min()                                  
    vmax = v.max()  
    thr_ct = len(v)                     
    vspace = (vmax - vmin) / (thr_ct - 1)
    indices = np.floor((k - vmin) / vspace)     
    valid = (indices >= 0) & (indices < thr_ct)
    
    # sum V1 and V2 per threshold
    V1 = np.bincount(indices[valid].astype(int), weights=sq[valid], minlength=thr_ct)
    V2 = np.bincount(indices[valid].astype(int), weights=frac[valid], minlength=thr_ct)

    # normalise
    V0 = (k[None, :] > v[:, None]).sum(axis=1) / k.size
    V1 = V1 / (4 * k.size)
    V2 = V2 / (2 * np.pi * k.size)
    
    return V0, V1, V2


