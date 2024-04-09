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
    dyy = np.gradient(dy,pixelsize,axis=1)
    
    return [dx,dy,dxx,dxy,dyy]


def V_012(k, thr_ct, k_std, pixelsize, t_min=-3, t_max=3):

    """
    Calculates the Minkowski Functionals (MF) V0, V1, and V2 for a 2D map

    Parameters
    ----------
    k : 2D numpy array
        2D map to calculate MFs of
    thr_ct : int
        Number of threshold values to calculate
    k_std : float
        Standard deviation of the map (for an individual realisation or a cosmology)
    t_min : float
        Minimum threshold value
    t_max : float
        Maximum threshold value
    
    Returns
    -------
    output : 1D numpy array
        Concatenated array containing the Minkowski Functionals V0, V1, and V2
    """
    
    # calculate derivatives
    kx,ky,kxx,kxy,kyy = map_derivatives(k, pixelsize)
  
    # define MF functions
    sq = np.sqrt(kx**2 + ky**2)
    frac = (2*kx*ky*kxy - (kx**2)*kyy - (ky**2)*kxx)/(kx**2 + ky**2)
    
    N = k.size                                                    # pixel count
    v = np.linspace(t_min*k_std,t_max*k_std,thr_ct)               # threshold values (assumes maps are centered around 0)
    vmin = v.min()                                                # threshold min
    vmax = v.max()                                                # threshold max
    vspace = (vmax-vmin)/thr_ct                                   # threshold array bin size

    output0 = np.zeros(thr_ct)          # V0
    output1 = np.zeros(thr_ct)          # V1
    output2 = np.zeros(thr_ct)          # V2
    
    ## V0 ##
    for index,i in enumerate(v):

        # find the counts of pixels where the pixel height is greater than that threshold value
        output0[index] = (k > i).sum()
    
    # divide by pixel count
    output0 = output0/N   

    ## V1, V2 ##
    
    # flatten arrays for MF calculation
    k = k.flatten()
    sq = sq.flatten()
    frac = frac.flatten()

    # get threshold bin index for each pixel
    indices = np.floor((k-vmin)/vspace)
    
    # find the closest threshold value for every pixel    
    for i,index in enumerate(indices):
        
        # filter out values outside valid indeces        
        if 0 <= index < thr_ct: 
            output1[int(index)] += sq[int(i)]
            output2[int(index)] += frac[int(i)] 
 
    output1 = output1 / (4*N)
    output2 = output2 / (2*np.pi*N)
    
    return np.concatenate((output0.flatten(),output1.flatten(),output2.flatten()))



