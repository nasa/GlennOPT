# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 12:45:36 2020

author: Trey Harrison (username: hmharri3)

Library of signal processing functions.

List of functions:
    signalFromSparseFrequency
"""

#%% import packages
import numpy as np

#%% specify frequency domain characteristics of signal

def signalFromSparseFrequency(fs, N, fKnown, XKnown):
    """
    reconstruct a signal in the time domain from sparse frequency content.
    useful for getting time signal of (without phase info) from digitized
    frequency plots.
    
    
    <333 T. Harrison 09/2020
    
    Inputs:
        1) fs - sample frequency
        2) N - number of points
        3) fKnown - frequency data at known values
        4) XKnown - amplitude data corresponding to fKnown
            
    Outputs:
        1) t - time domain data
        2) x - x(t)
    
    Notes:
        1) Will you shut up, man?
    """
    ##########################################################################
    # reconstruct full frequency content of signal
    # we assume frequency content not given is zero
    k  = np.arange(N)                       # number of frequencies in spectrum
    fk = k*fs/N                             # actual frequencies in spectrum
    X  = np.zeros(len(fk))                  # initialize amplitude of spectrum
    
    # assign values at known frequencies
    for i,f in enumerate(fk):
        if f in fKnown:
            X[i] = XKnown[f == fKnown]
    
    ##########################################################################
    # calculate inverse discrete fourier transform
    
    x = N*np.real(np.fft.ifft(X))
    dt = 1/fs
    t = np.arange(N)*dt
       
    return t, x, fk, X
