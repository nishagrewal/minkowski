import healpy as hp
import numpy as np
import math
import os


''' Importing V12 function from C code ''' 
# C imports
import ctypes

# define v12 funtion from C code
lib = ctypes.cdll.LoadLibrary("./mf_c.so")
calculate_v12_c = lib.calculate_v12
c_double_p = ctypes.POINTER(ctypes.c_double)
# int n, double * v, double * sq, double * frac, int nt, double vmin, double vspace, double * v1, double * v2

# define type for each arguement
calculate_v12_c.argtypes = [
    ctypes.c_int,     # n
    c_double_p,       # v
    c_double_p,       # sq
    c_double_p,       # frac
    ctypes.c_int,     # nt
    ctypes.c_double,  # vmin
    ctypes.c_double,  # vspace
    c_double_p,       # v1
    c_double_p,       # v2 
]

# no output from the function - instead inputs are filled
calculate_v12_c.restype = None



''' Minkowski Functional Calculations '''

# calculate pixel derivatives

def map_derivatives(m):
    """
    Compute the derivatives of a map
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


### MF functions

def V_0(v,k):

    output = []
    for i in v:

        # find the counts of pixels where the pixel height is greater than that threshold value
        count = (k > i).sum()
        output = np.append(output,count)
    
    # divide by pixel count
    output = output/(k.size)   
    return output


def V_12(v,k,kx,ky,kxx,kxy,kyy):
    
    vmin = v.min()                      # threshold min
    vmax = v.max()                      # threshold max
    vspace = (vmax-vmin)/len(v)         # threshold array bin size
    N = k.size                          # pixel count
    nt = v.size                         # threshold count

    output1 = np.zeros(len(v))
    output2 = np.zeros(len(v))
  
    # define MF functions
    sq = np.sqrt(kx**2 + ky**2)
    frac = (2*kx*ky*kxy - (kx**2)*kyy - (ky**2)*kxx)/(kx**2 + ky**2)
    
    # use C function to calculate V1 and V2 (using only the values of the objects)
    calculate_v12_c(N, 
                    k.ctypes.data_as(c_double_p),
                    sq.ctypes.data_as(c_double_p),
                    frac.ctypes.data_as(c_double_p),
                    nt,
                    vmin,
                    vspace,
                    output1.ctypes.data_as(c_double_p),
                    output2.ctypes.data_as(c_double_p),
    )

    output1 = output1 / (4*N)
    output2 = output2 / (2*np.pi*N)    
    return output1,output2
