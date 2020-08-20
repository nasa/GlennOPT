'''
    Defines mutation and crossover 
'''
import random
import math
import copy
import numpy as np
from numpy import ndarray
from typing import List
npList = List[np.ndarray]


def mutate(x1:np.ndarray,mu:float=0.02,sigma:float=0.2):
        '''
            Mutate the evaluation parameters
            Simple mutate
            Inputs:
                x1 - array of evaluation parameters
                mu - percentage of population to mutate
                sigma - mutation scale 
        
        '''
        nMu = math.ceil(mu*len(x1))
        j = np.random.randint(0,len(x1)-1,size=nMu)
        y = x1 
        for k in range(len(j)):
            if (j[k]>0):
                y[j[k]] = x1[j[k]]+sigma*np.random.randn(1)
        return y

def crossover(x1:np.ndarray,x2:np.ndarray):
    '''
        Simple crossover
        Inputs:
            x1 - array of evaluation parameters
            x2 - array of evaluation parameters
    '''
    alpha = np.random.rand(len(x1))
    y1=alpha*x1+(1.0-alpha)*x2
    y2=alpha*x2+(1.0-alpha)*x1
    return y1,y2
        
# Mutate x1 using a,b,c
def mutate_crossover_de(x:ndarray,xp:npList,F=0.6,C=0.8):
    '''
        Differential evolution mutation de/1/rand/bin
        Inputs: 
            x - vector of design variables 
            xp - List of parents (min parents = 3)
            xa - random vector of design variables
            xb - random vector of design variables
            xc - random vector of design variables
            F - Amplification Factor
            C - Crossover factor
        returns z mutated vector
    '''
    
    y = x*0 # Creates a copy
    z = y*0   
    for i in range(0,len(x)): # Mutate each index of the inputs
        temp = 0
        for j in range(1,len(xp)-1): # Iterates for the number of parents 
            temp += xp[j+1][i]-xp[j][i]        
        y[i] = (xp[0][i] + F*temp) 
        if (random.random() <= C):
            z[i] =  y[i]
        else:
            z[i] =  x[i]
    return z

def mutate_crossover_de_best_2_bin(x_best:ndarray,x_r1:ndarray,x_r2:ndarray,x_r3:ndarray,x_r4:ndarray,F=0.5,C=0.):
    '''
        Differential evolution mutation de/1/rand/bin
        Inputs: 
            x_best - vector of design variables 
            x_r1 - random vector of design variables
            x_r2 - random vector of design variables
            x_r3 - random vector of design variables
            x_r4 - random vector of design variables
            F - Amplification Factor
            C - Crossover factor
    '''
    y = x_best*0
    z = x_best*0
    for i in range(len(y)):
        y[i] = x_best[i]+F*(x_r1[i]-x_r2[i])+F*(x_r3[i]-x_r4[i])
        if (random.random() <= C):
            z[i] =  y[i]
        else:
            z[i] =  x_best[i]
    return z
        