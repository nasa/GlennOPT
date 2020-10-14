'''
    Defines mutation and crossover 
'''
from dataclasses import dataclass, field
from enum import Enum
import random
import math
import copy
import numpy as np
from numpy import ndarray
from typing import List
from ..helpers import Parameter
from ..base_classes import Individual


class de_mutation_type(Enum):
    """
        differential evolution mutation type. users can select what kind of mutation type to use 
    """
    de_1_rand_bin = 1
    de_best_2_bin = 2
    simple = 3

@dataclass
class mutation_parameters:
    """
        Data class for storing the mutation parameters used for NSGA and differential evolution problems 

        Properties:
            mutation_type:
            sigma: 
            mu: 
            F:
            C:
            min_parents = 0.1
            max_parents = 0.9
    """
    mutation_type: de_mutation_type = field(repr=True,default=de_mutation_type.de_1_rand_bin)
    sigma: float = field(repr=True,default=0.2)
    mu: float = field(repr=True,default=0.02)
    F: float = field(repr=True,default=0.6)
    C: float = field(repr=True,default=0.8)
    min_parents:int = field(repr=True,default=2)
    max_parents:int = field(repr=True,default=10)

def mutation_de_1_rand_bin(individuals:List[Individual],objectives:List[Parameter],eval_parameters:List[Parameter],performance_parameters:List[Parameter],min_parents:int,max_parents:int,F:float, C:float):
    """
        Applies mutation and crossover using de_1_rand_bin to a list of individuals 
        Inputs:
            individuals - list of individuals. Takes the best individual[0] (sorted lowest to highest)
            objectives - list of objectives List[Parameter]
            performance_parameters - list of parameters List[parameter]
            F - Amplification Factor [0,2]
            C - Crossover factor [0,1]
    """
    ind_best = individuals[0]
    x1 = ind_best.eval_parameters
    newIndividuals=list()        
    for i in range(len(individuals)):   # Loop for every individual        
        nParents = random.randint(min_parents,max_parents) 
        z = copy.deepcopy(x1)
        for p in range(nParents): # Call the mutation             
            r1 = random.randint(0,len(individuals)-1)            
            r2 = random.randint(0,len(individuals)-1)
            while r2 == r1:
                r2 = random.randint(0,len(individuals)-1)            
            xp = list()                       # gets a list of evaluation parameters
            for indx in [r1,r2]:
                xp.append(individuals[indx].eval_parameters)
            z = mutate_crossover_de(z,xp,ind_best.eval_parameter_min, ind_best.eval_parameter_max,F,C)
        newIndividuals.append(Individual(eval_parameters=set_eval_parameters(eval_parameters,z),objectives=objectives,performance_parameters=performance_parameters))
    return newIndividuals 

def mutation_de_best_2_bin(best_indx:int,individuals:List[Individual],objectives:List[Parameter],eval_parameters:List[Parameter],performance_parameters:List[Parameter],F:float,C:float):
    """
        mutation and crossover using de_best_2_bin (single objective only)
        Inputs:
            individuals - list of individuals
            objectives - list of objectives List[Parameter]
            performance_parameters - list of parameters List[parameter]
            F - Amplification Factor [0,2]
            C - Crossover factor [0,1]
    """
    newIndividuals=[]
    x_best = individuals[best_indx].eval_parameters
    for i in range(len(individuals)): # Loop for every individual            
        parent_indicies = get_pairs(len(individuals),4,[best_indx]) # pre-populate with the best index
        parent_indicies.remove(best_indx)            
        x1 = individuals[parent_indicies[0]].eval_parameters
        x2 = individuals[parent_indicies[1]].eval_parameters
        x3 = individuals[parent_indicies[2]].eval_parameters
        x4 = individuals[parent_indicies[3]].eval_parameters
        
        x = mutate_crossover_de_best_2_bin(x_best,x1,x2,x3,x4,ind1.eval_parameter_min, ind1.eval_parameter_max,F,C)

        newIndividuals.append(Individual(eval_parameters=set_eval_parameters(eval_parameters,x),objectives=objectives,performance_parameters=performance_parameters))
                
    return newIndividuals 

def mutation_simple(self,individuals:List[Individual],nCrossover:int,nMutation:int,objectives:List[Parameter],eval_parameters:List[Parameter],performance_parameters:List[Parameter],mu:float,sigma:float):
    """
        Performs a simple mutation and crossover on the individuals
        Inputs:
            individuals - list of individuals
            objectives - list of objectives List[Parameter]
            eval_parameters
            performance_parameters - list of parameters List[parameter]
            mu - mutation rate 0.2
            sigma - mutation step size 0.1
    """
    nIndividuals = len(individuals)
    # Perform Crossover
    crossover_individuals = []
    for k in range(int(nCrossover/2)):
        rand_indx = np.random.randint(0,nIndividuals-1)
        y1 = individuals[rand_indx].eval_parameters

        rand_indx2 = np.random.randint(0,nIndividuals-1)
        y2 = individuals[rand_indx2].eval_parameters
        [y1_new, y2_new] = crossover(y1, y2)
        
        crossover_individuals.append(Individual(eval_parameters=set_eval_parameters(eval_parameters,y1_new),objectives=objectives,performance_parameters=performance_parameters))        
        crossover_individuals.append(Individual(eval_parameters=set_eval_parameters(eval_parameters,y2_new),objectives=objectives,performance_parameters=performance_parameters))

    # Perform Mutation
    mutation_individuals = []
    for k in range(nMutation):
        rand_indx = np.random.randint(0,nIndividuals-1)
        y1 = individuals[rand_indx].eval_parameters
        ymin = individuals[rand_indx].eval_parameter_min
        ymax = individuals[rand_indx].eval_parameter_max

        y1_new = mutate(y1,ymin,ymax,mu,sigma)

        mutation_individuals.append(Individual(eval_parameters=set_eval_parameters(eval_parameters,y1_new),objectives=objectives,performance_parameters=performance_parameters))
    crossover_individuals.extend(mutation_individuals)
    return crossover_individuals

# Core functions 
def mutate(x1:np.ndarray,xmin:ndarray,xmax:ndarray,mu:float=0.02,sigma:float=0.2):
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
        indx = j[k]
        if (indx>0):
            y[indx] = x1[indx]+sigma*np.random.randn(1)

            if (xmin is not None) and (y[indx]<xmin[indx]):
                y[indx]=xmin[indx]
            if (xmax is not None) and (y[indx]>xmax[indx]):
                y[indx]=xmax[indx]
    return ys

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
def mutate_crossover_de(x:ndarray,xp:List[np.array],xmin:ndarray=None,xmax:ndarray=None,F=0.6,C=0.8):
    '''
        Differential evolution mutation de/1/rand/bin
        Inputs: 
            x - vector of design variables 
            xp - List of parents (min parents = 3)
            xa - random vector of design variables
            xb - random vector of design variables
            xc - random vector of design variables
            F - Amplification Factor [0,2]
            C - Crossover factor [0,1]
        returns z mutated vector
    '''
    
    z = x*0 # Creates a copy    
    
    for i in range(len(x)): # Mutate each index of the inputs
        temp = 0
        for j in range(len(xp)-1): # Iterates for the number of parents 
            temp += (xp[j+1][i]-xp[j][i])
        y = x[i] + F*temp        
        
        if (random.random() <= C or i==random.randint(0,len(x))):
            z[i] =  y
        else:
            z[i] =  x[i]
        
        if (xmin is not None) and (z[i]<xmin[i]):
            z[i]=xmin[i]    
        if (xmax is not None) and (z[i]>xmax[i]):
            z[i]=xmax[i]
    return z

def mutate_crossover_de_best_2_bin(x_best:ndarray,x_r1:ndarray,x_r2:ndarray,x_r3:ndarray,x_r4:ndarray,xmin:ndarray,xmax:ndarray,F=0.5,C=0.8):
    '''
        Differential evolution mutation de/1/rand/bin, Used for single objective not multi
        Inputs: 
            x_best - vector of design variables 
            x_r1 - random vector of design variables
            x_r2 - random vector of design variables
            x_r3 - random vector of design variables
            x_r4 - random vector of design variables
            F - Amplification Factor [0,2]
            C - Crossover factor [0,1]
    '''
    y = x_best*0
    z = x_best*0
    for i in range(len(y)):
        y[i] = x_best[i]+F*(x_r1[i]-x_r2[i])+F*(x_r3[i]-x_r4[i])
        
        if (xmin is not None) and (y[i]<xmin[i]):
            y[i]=xmin[i]
        if (xmax is not None) and (y[i]>xmax[i]):
            y[i]=xmax[i]

        if (random.random() <= C):
            z[i] =  y[i]
        else:
            z[i] =  x_best[i]
    return z


# Helper functions
def get_pairs(nIndividuals:int,nParents:int,parent_indx_seed=[]):
    """
        Get a list of all the pairing partners for a particular individual
        Inputs:
            nIndividuals - number of individuals
            nParents - number of parents 
            parent_indx_seed - pre-populate the parent index array
    """    
    rand_indx = random.sample(range(0,nIndividuals),nParents)
    # while(any(x in rand_indx for x in parent_indx_seed)):
    #     rand_indx = random.sample(range(0,nIndividuals),nParents)
    return rand_indx

def set_eval_parameters(eval_parameters:List[Parameter], x:np.ndarray):
    """
        Set the evaluation parameters 
        Inputs:
            eval_parameters - list of parameters as a class. x is mapped to eval_parameter.value
            x - represents an the mutated value/array to be evaluated 
    """
    parameters = copy.deepcopy(eval_parameters)
    for indx in range(len(parameters)):
        parameters[indx].value = x[indx]
    return parameters