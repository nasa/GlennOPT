# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:18:36 2020

@author: hmharri3
probePlacementOptimizationPlusWN.py

Optmize probe placement around circumference of annulus for given combinations
of wavenumbers to best evauluate the circumferential variation of quantities.

The number of probes (nProbes) must be
    nProbes >= 2*N+2
where N is the number of wavenumbers of interest.

The minimimum spacing is minimimum number of degrees between probes and can be
defined by the user. The code is set up to distribute the probes around the
circumference of the annulus.

Singleor


<333 Trey 09/2020
"""

#%% import packages

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pyswarms as ps
import signalProcessing as sig
from scipy import stats
import itertools
import waveletReconstruction as fLou
import plotParty as pp

#%% inputs

# signal reconstruction parameters
nProbes           = 10                               # numbers of probes
waveNumberGuesses = [[8,96,104]]                # guesses of wave numbers
singleOrCombos    = 'single'                        # only one wavenumber combination or try multiple combos
maxCombo          = 3          # maximum number of wavenumbers to combine

# swarm parameters
minSpacing = 3      # minimum spacing between probes in degrees
swarmSize  = 500
iterations = 500

# plot options
saveFig = True

#%% define signal to reconstruct

fs        = 3600
N         = 3600
fKnown = np.asarray([8,48,72,80,88,96,104])
XKnown   = np.asarray([0.7,0.05,0.2,0.3,
                            0.1,2.45,0.5,])
# fKnown = np.asarray([1])
# XKnown = np.asarray([10])

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
        
import plotParty as pp
pp.setFigFormat('pres')
# frequency plot of true signal
fig, ax = plt.subplots(1,1)
ax.plot(fHR,XHR)
ax.set_xlim(0,120)
ax.set_xlabel('Wavenumber')
ax.set_ylabel('Signal Units')
yMax = round(max(XHR)*2)/2
ax.set_ylim(0,yMax)
ax.grid(False)
plt.tight_layout()
if saveFig:
    plt.savefig('circumferentialReconstructionTestSingalFrequency.png')

# time domain plot of true signal
fig, ax = plt.subplots(1,1)
ax.plot(theta,trueSignal)
ax.set_xlim(0,360)
ax.set_xlabel('Circumferential Position [$^\circ$]')
ax.set_ylabel('Signal Units')
yMin = round(min(trueSignal)*2)/2
yMax = round(max(trueSignal)*2)/2
ax.set_ylim(yMin, yMax)
plt.tight_layout()
if saveFig:
    plt.savefig('circumferentialReconstructionTestSingalTime.png')



#%% get all possible combinations of wave number guesses

if singleOrCombos == 'combos':
    wnCombos = []
    for i in range(1,len(waveNumberGuesses)+1):
        subCombos = [np.asarray(x) for x in itertools.combinations(waveNumberGuesses,i)]
        wnCombos.extend(subCombos)
else:
    wnCombos = waveNumberGuesses

#%% define objective function

def objectiveFunction(probeTheta):
    n_particles = probeTheta.shape[0]
    cost = np.zeros(n_particles)
    for i in range(n_particles):
        k = np.zeros(len(wnCombos))
        for j,wn in enumerate(wnCombos):
            signal = tsInterp(probeTheta[i])
            A, F = fLou.solveForWaveletCoefficients(probeTheta[i], signal, wn)
            k[j] = np.linalg.cond(A)
        cost[i] = np.sum(k**2)    
    return cost

#%% define swarm parameters

dim = nProbes
options = {'c1':1.5, 'c2':1.5, 'w':0.5}
probeSpacing = 360/nProbes
tLo     = np.zeros(nProbes)
tHi     = np.zeros(nProbes)
for i in range(nProbes):
    tLo[i] = probeSpacing*i
    if i != nProbes-1:
        tHi[i] = probeSpacing*(i+1) - minSpacing
    else:
        tHi[-1] = probeSpacing*(i+1)
constraints = (tLo,tHi)

#%%  do the optimization

# call optimizer instance of pso
optimizer = ps.single.GlobalBestPSO(n_particles=swarmSize, dimensions=dim, 
                                    options=options, bounds=constraints)
# optimize
cost, joint_vars = optimizer.optimize(objectiveFunction, iters=iterations)

from pyswarms.utils.plotters import plot_cost_history
plt.figure()
plot_cost_history(optimizer.cost_history)

probeTheta  = joint_vars

#%% reconstruct signal from optimization output for each combo

A = [None]*len(wnCombos)
F = [None]*len(wnCombos)
N = [None]*len(wnCombos)
xR = [None]*len(wnCombos)
signal = tsInterp(probeTheta)
# do for best of each combination
for i,wn in enumerate(wnCombos):
    A[i], F[i] = fLou.solveForWaveletCoefficients(probeTheta, signal, wn)
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
fig, ax = plt.subplots(2, 1)
rects1 = ax[0].bar(x - width/2, pearsonR)
rects2 = ax[1].bar(x-width/2, rmsR)

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

#%% check error

print('Pearson R:', pearsonR[bestIndex])
print('rmsR:     ', rmsR[bestIndex])
