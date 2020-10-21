import time

# Load external modules 
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import signalProcessing as sig
import plotParty as pp
import waveletReconstruction as fLou
from scipy import stats
import itertools


'''
This example comes from the paper of 
RECONSTRUCTING COMPRESSOR NON-UNIFORM CIRCUMFERENTIAL FLOW FIELD
FROM SPATIALLY UNDERSAMPLED DATA 
GT2020-15465
'''

'''
Globals
'''
singleOrCombos = 'combos' # only one wavenumber combination or try multiple combos
waveNumberGuesses = [[8,96,104],[8,96,80]] # guesses of wave numbers

wnCombos = waveNumberGuesses


#* Start of Trey's code
def true_signal_construction():
    # define signal to reconstruct    
    fs        = 3600
    N         = 3600
    fKnown = np.asarray([8,48,72,80,88,96,104])
    XKnown   = np.asarray([0.7,0.05,0.2,0.3,0.1,2.45,0.5])

    t, x, f, X = sig.signalFromSparseFrequency(fs, N, fKnown, XKnown)


    # convert to degrees and only take one revolution
    theta = t * 360
    theta = np.append(theta[theta<360],360)
    trueSignal  = np.append(x,x[0])
    tsInterp = interp1d(theta, trueSignal, kind='linear')

    step = max(theta)/(100*360)
    fHR = np.arange(0,max(theta)+step,step)
    XHR = np.zeros(len(fHR))
    for i,fhr in enumerate(fHR):
        for j,fk in enumerate(fKnown):
            if abs(fk-fhr) < step/10:
                XHR[i] = XKnown[j]

    # # Plot the true signal 
    # pp.setFigFormat('pres')
    # # frequency plot of true signal
    # _, ax = plt.subplots(1,1)
    # ax.plot(fHR,XHR)
    # ax.set_xlim(0,120)
    # ax.set_xlabel('Wavenumber')
    # ax.set_ylabel('Signal Units')
    # yMax = round(max(XHR)*2)/2
    # ax.set_ylim(0,yMax)
    # ax.grid(False)
    # plt.tight_layout()
    # plt.savefig('circumferentialReconstructionTestSingalFrequency.png')

    # # time domain plot of true signal
    # _, ax = plt.subplots(1,1)
    # ax.plot(theta,trueSignal)
    # ax.set_xlim(0,360)
    # ax.set_xlabel('Circumferential Position [$^\circ$]')
    # ax.set_ylabel('Signal Units')
    # yMin = round(min(trueSignal)*2)/2
    # yMax = round(max(trueSignal)*2)/2
    # ax.set_ylim(yMin, yMax)
    # plt.tight_layout()
    # plt.savefig('circumferentialReconstructionTestSingalTime.png')

    return theta,trueSignal, tsInterp

#* define objective function

# goal is to minimize condition number of design matrix
# low condition numbers give lowest error metrics

def objective_function(probeTheta, tsInterp, theta, trueSignal):
    '''
        Vary the location of probeTheta location so that we pick up the true signal
        Inputs
            probeTheta - this is what we are solving for
    '''  
    k = np.zeros(len(wnCombos))
    for j,wn in enumerate(wnCombos):
        A = fLou.buildDesignMatrix(probeTheta, wn)
        k[j] = np.linalg.cond(A)
    cost = k

    # Plot code 
    A = [None]*len(wnCombos)
    F = [None]*len(wnCombos)
    N = [None]*len(wnCombos)
    xR = [None]*len(wnCombos)
    signal = tsInterp(probeTheta)
    # do for best of each combination
    for i,wn in enumerate(wnCombos):
        F[i], A[i] = fLou.solveForWaveletCoefficients(probeTheta, signal, wn)
        xR[i] = np.zeros(len(theta))
        # for j,t in enumerate(theta):
        xRtemp = 0
        for j in np.arange(len(wn)):
            xRtemp = xRtemp + (F[i][2*j]*np.sin(wn[j]*theta*np.pi/180) +
                            F[i][2*j+1]*np.cos(wn[j]*theta*np.pi/180))
        xR[i] = xRtemp + F[i][-1]
        
    #%% check error in reconstructed signal

    pearsonR = np.zeros(len(wnCombos))
    rmsR     = np.zeros(len(wnCombos))
    for i in range(len(wnCombos)):
        pearsonR[i] = stats.pearsonr(trueSignal, xR[i])[0]
        rmsR[i]     = np.sqrt(1/len(trueSignal)*(np.sum((xR[i]-trueSignal)**2)))

    #%% visual evaluations for goodness of fit

    # bar chart of error metric
    xLabel = [str(wn) for wn in wnCombos]
    x      = np.arange(len(xLabel))
    width  = 0.35

    # make plot
    _, ax = plt.subplots(2, 1)
    # rects1 = ax[0].bar(x - width/2, pearsonR)
    # rects2 = ax[1].bar(x-width/2, rmsR)

    # format plot
    ax[0].set_ylabel('Pearson R')
    ax[1].set_ylabel('RMS Error')
    ax[0].set_ylim(0.75,1)
    for i in range(len(ax)):
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(xLabel)


    # line plot of true signal vs best reconstructed signal
    bestIndex     = np.argmax(pearsonR)
        
    plt.figure()
    plt.plot(theta,trueSignal)
    plt.plot(theta,xR[bestIndex],'--')
    plt.xlim((0,60))
    plt.savefig('signal_comparison.png')

    bestIndex     = np.argmax(pearsonR)

    return cost, pearsonR[bestIndex], rmsR[bestIndex]
        

#* GlennOPT helper code. Do not modify 
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
        f.write('objective1 = {0:.6f}\n'.format(y[0])) # Output should contain [Name of the Objective/Parameter] = [value] This is read by the optimizer 
        f.write('objective2 = {0:.6f}\n'.format(y[1]))
        if perf is not None:
            f.write('PearsonR = {0:.6f}\n'.format(perf[0]))
            f.write('RMS_Error = {0:.6f}\n'.format(perf[1]))

if __name__ == '__main__':
    '''
        Main Execution code. Reads the inputs, finds the cost, outputs the cost.
        
    '''
    x = read_input("input.dat") 
    theta,trueSignal, tsInterp = true_signal_construction()
    y,pearson_r,rms_error = objective_function(x,tsInterp, theta, trueSignal)
    print_output(y,[pearson_r,rms_error])

    