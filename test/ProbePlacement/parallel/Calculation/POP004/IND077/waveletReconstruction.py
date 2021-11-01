# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:10:42 2020

author: Trey Harrison (username: hmharri3)

Library of functions and classes used to reconstruct nonuniform circumferential
signals from spatially undersampled data. Based on methodoly published by
Lou and Key (and Matthews and Kormanik) in
RECONSTRUCTING COMPRESSOR NON-UNIFORM CIRCUMFERENTIAL FLOW FIELD FROM
SPATIALLY UNDERSAMPLED DATA (2020).
One of the few good things to come out of 2020. This and Eleazar.

List of functions:
    buildDesignMatrix
    solveForWaveletCoefficients
"""

#%% import packages

import numpy as np

#%% buildDesignMatrix

def buildDesignMatrix(probeTheta, waveNumber):
    """
    Build the design matrix (A) from probe positions and the wavenumbers of
    interest. Use the design matrix and signal from the probes, x, to solve
    for the wavelet coefficients.
    
    <333 T. Harrison 10/2020
    
    Input:
        1) probeTheta - angular position of probes between 0 and 360
        2) waveNumber - waveNumbers of itnerest
    
    Output:
        1) design matrix A
    """
    ##########################################################################
    m          = len(probeTheta)
    N          = len(waveNumber)*2 + 1
    A          = np.zeros([m,N])
    for i,pt in enumerate(probeTheta):
        for j in range(len(waveNumber)):
            A[i,2*j]   = np.sin(waveNumber[j]*pt*np.pi/180)
            A[i,2*j+1] = np.cos(waveNumber[j]*pt*np.pi/180)
    
    # last column equals 1
    A[:,-1] = 1
    
    return A

#%% solveForWaveletCoefficients

def solveForWaveletCoefficients(probeTheta, x, waveNumber):
    """
   Use the design matrix, A, and signal from the probes, x, to solve
    for the wavelet coefficients (output in F).
    
    <333 T. Harrison 10/2020
    
    Input:
        1) probeTheta - angular position of probes between 0 and 360
        2) x - signal from probes corresponding to probeTheta
        3) waveNumber - waveNumbers of itnerest
    
    Output:
        1) design matrix A
        2) wavelet coefficients in array F
    """
    ##########################################################################
    
    A = buildDesignMatrix(probeTheta, waveNumber)
    F = np.linalg.lstsq(A,x,rcond=None)[0]
    
    return F, A