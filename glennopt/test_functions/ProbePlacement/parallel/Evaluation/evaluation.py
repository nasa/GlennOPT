
import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
'''
This example comes from the paper of 
RECONSTRUCTING COMPRESSOR NON-UNIFORM CIRCUMFERENTIAL FLOW FIELD
FROM SPATIALLY UNDERSAMPLED DATA 
GT2020-15465
'''
def read_input(input_filename):
    x = []
    with open(input_filename, "r") as f: 
        for line in f:
            split_val = line.split('=')
            if len(split_val)==2: # x1 = 2 # Grab the 2
                x.append(float(split_val[1]))
    return x
 
def print_output(y,perf=None):
    with open("output.txt", "w") as f:        
        f.write('objective1 = {0:.6f}\n'.format(y)) # Output should contain [Name of the Objective/Parameter] = [value] This is read by the optimizer 

def true_signal_construction():
    # define signal to reconstruct
    fs        = 3600
    fLo       = 0
    fHi       = fs/2
    N         = 3600
    # fKnown = np.asarray([10])
    # XKnown = np.asarray([10])
    dcOffset    = 0
    fKnown = np.asarray([8,48,72,80,88,96,104])
    XKnown   = np.asarray([0.6229356,0.085519664,0.18917666,0.4036676,
                            0.15955658,2.217998,0.49061242,])

    # construct frequency content of signal

    k  = np.arange(N)
    fk = k*fs/N
    X  = np.zeros(len(fk))

    for i,f in enumerate(fk):
        if f in fKnown:
            X[i] = XKnown[f == fKnown]

    # plt.figure()
    # plt.plot(fk,X)
    # plt.xlim(0,120)
    # plt.savefig('true_signal.png')

    #%% calculate inverse discrete fourier transform
    dt = 1/fs * 360
    x = N*np.real(np.fft.ifft(X))
    x = (x - np.mean(x))
    t = np.arange(N)*dt

    # convert to degrees and only take one revolution
    theta = np.append(t[t<360],360)
    trueSignal  = np.append(x[t<360],x[0]) + 9

    # solve for coefficients

    # change variable names to make them spatial instead of temporal
    waveNumber = np.atleast_1d([8,96,104])
    return theta,trueSignal, waveNumber

def solveForWaveletCoefficients(probeTheta, waveNumber, theta, trueSignal):
    m          = len(probeTheta)
    N          = len(waveNumber)*2 + 1
    A          = np.zeros([m,N])
    x          = np.zeros([m])
    tsInterp = interp1d(theta, trueSignal, kind='linear')
    for i,pt in enumerate(probeTheta):
        x[i] = tsInterp(pt)
        for j in np.arange((N-1)//2):
            A[i,2*j]   = np.sin(waveNumber[j]*pt*np.pi/180)
            A[i,2*j+1] = np.cos(waveNumber[j]*pt*np.pi/180)
    # last column equals 1
    A[:,-1] = 1
    F = np.linalg.lstsq(A,x,rcond=None)[0]
    
    return A, F


#* define objective function

# goal is to minimize condition number of design matrix
# low condition numbers give lowest error metrics

def opt_func(probeTheta, waveNumber, theta, trueSignal):
    '''
        Vary the location of probeTheta location so that we pick up the true signal
    '''  
    A, F = solveForWaveletCoefficients(probeTheta, waveNumber, theta, trueSignal)
    cost = cost_function(probeTheta, waveNumber, A)    
    return cost
    
    
    

def cost_function(probeTheta, waveNumber, A):
    if len(probeTheta) == len(waveNumber)*2+1:
        k = np.linalg.norm(A)*np.linalg.norm(np.linalg.inv(A))
    else:
        k = np.linalg.norm(A)*np.linalg.norm(np.linalg.pinv(A))
    fcost = np.sum((1*k)**2)
    return fcost

if __name__ == '__main__':
    x = read_input("input.dat") 
    theta,trueSignal, waveNumber = true_signal_construction()
    y = opt_func(x,waveNumber, theta, trueSignal)
    print_output(y)   