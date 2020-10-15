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
from ..helpers import Parameter, convert_to_ndarray
from ..base_classes import Individual


class de_mutation_type(Enum):
    """
        differential evolution mutation type. users can select what kind of mutation type to use 
    """
    de_rand_1_bin = 1
    de_best_1_bin = 2
    de_rand_2_bin = 3
    de_best_2_bin = 4
    simple = 5

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
    mutation_type: de_mutation_type = field(repr=True,default=de_mutation_type.de_rand_1_bin)
    sigma: float = field(repr=True,default=0.2)
    mu: float = field(repr=True,default=0.02)
    F: float = field(repr=True,default=0.6)
    C: float = field(repr=True,default=0.8)
    min_parents:int = field(repr=True,default=2)
    max_parents:int = field(repr=True,default=10)

def de_best_1_bin(individuals:List[Individual],objectives:List[Parameter],eval_parameters:List[Parameter],performance_parameters:List[Parameter],min_parents:int,max_parents:int,F:float, C:float):
    """
        Applies mutation and crossover using de_1_rand_bin to a list of individuals 
        Inputs:
            individuals - list of individuals. Takes the best individual[0] (sorted lowest to highest)
            objectives - list of objectives List[Parameter]
            performance_parameters - list of parameters List[parameter]
            F - Amplification Factor [0,2]
            C - Crossover factor [0,1]
        Citatons:
            https://gist.github.com/martinus/7434625df79d820cd4d9
            Storn, R., & Price, K. (1997). Differential Evolution -- A Simple and Efficient Heuristic for global Optimization over Continuous Spaces. Journal of Global Optimization, 11(4), 341–359. https://doi.org/10.1023/A:1008202821328 
            Ao, Y., & Chi, H. (2009). Multi-parent Mutation in Differential Evolution for Multi-objective Optimization. 2009 Fifth International Conference on Natural Computation, 4, 618–622. https://doi.org/10.1109/ICNC.2009.149
    """ 
    # nParents = random.randint(min_parents,max_parents)        
    newIndividuals=list()        
    nIndividuals = len(individuals)
    xmin = individuals[0].eval_parameter_min
    xmax = individuals[0].eval_parameter_max

    min_parents = max([2,min_parents])
    max_parents = max([2,max_parents])
    nparents = random.randint(min_parents,max_parents)  # Number of parents for crossover

    x1 = individuals[0].eval_parameters         # Use the best individual
    for i in range(len(individuals)):               # Start loop through population          
        p_indicies = get_pairs(nIndividuals,2,[i])  # random pick 3 vectors that are not i
        xp = list()                                 # gets a list of evaluation parameters
        for indx in p_indicies:
            xp.append(individuals[indx].eval_parameters)
        
        k = random.randint(0,len(x1))               # Randomly pick an index to force change
        z = x1*0
        for j in range(len(x1)):
            if (random.random()>C) or (j==k):       # Perform D-1 binomial trials
                z[j] = x1[j] + F*(xp[0][j]-xp[1][j])
            else:
                z[j] = x1[j] 

            if (xmin is not None) and (z[j]<xmin[j]):
                z[j]=xmin[j]
            if (xmax is not None) and (z[j]>xmax[j]):
                z[j]=xmax[j]
        
        newIndividuals.append(Individual(eval_parameters=set_eval_parameters(eval_parameters,z),objectives=objectives,performance_parameters=performance_parameters))
    return newIndividuals 

def de_rand_1_bin(individuals:List[Individual],objectives:List[Parameter],eval_parameters:List[Parameter],performance_parameters:List[Parameter],min_parents:int,max_parents:int,F:float, C:float):
    """
        Applies mutation and crossover using de_rand_1_bin to a list of individuals 
        Inputs:
            individuals - list of individuals. Takes the best individual[0] (sorted lowest to highest)
            objectives - list of objectives List[Parameter]
            performance_parameters - list of parameters List[parameter]
            F - Amplification Factor [0,2]
            C - Crossover factor [0,1]
        Citatons:
            https://gist.github.com/martinus/7434625df79d820cd4d9
            Storn, R., & Price, K. (1997). Differential Evolution -- A Simple and Efficient Heuristic for global Optimization over Continuous Spaces. Journal of Global Optimization, 11(4), 341–359. https://doi.org/10.1023/A:1008202821328 
            Ao, Y., & Chi, H. (2009). Multi-parent Mutation in Differential Evolution for Multi-objective Optimization. 2009 Fifth International Conference on Natural Computation, 4, 618–622. https://doi.org/10.1109/ICNC.2009.149
    """ 
    # nParents = random.randint(min_parents,max_parents)        
    newIndividuals=list()        
    nIndividuals = len(individuals)
    xmin = individuals[0].eval_parameter_min
    xmax = individuals[0].eval_parameter_max

    min_parents = max([3,min_parents])
    max_parents = max([3,max_parents])
    nparents = random.randint(min_parents,max_parents)          # Number of parents for crossover
    w = np.random.dirichlet(np.ones(nparents),size=1)           # Random weights that sum to 1
    w = w[0]
    for i in range(len(individuals)):                           # Start loop through population  
        x1 = individuals[i].eval_parameters                
        
        p_indicies = get_pairs(nIndividuals,nparents,[i])       # random pick 3 vectors that are not i
        xp = list()                                             # gets a list of evaluation parameters
        for indx in p_indicies:
            xp.append(individuals[indx].eval_parameters)
        
        k = random.randint(0,len(x1))                           # Randomly pick an index to force change
        z = x1*0
        for j in range(len(x1)):
            if (random.random()>C) or (j==k):                   # Perform D-1 binomial trials
                temp = 0                                        # Summation for number of parents 
                if nparents > 3:                                # Use mult-parent mutation
                    for k in range(1,math.floor(nparents/2)):   
                        temp += F*(xp[2*k-1][j]-xp[2*k][j])
                else:
                    temp = F*(xp[1][j]-xp[2][j])                # Use default
                z[j] = xp[0][j] + temp
            else:
                z[j] = x1[j] 

            if (xmin is not None) and (z[j]<xmin[j]):
                z[j]=xmin[j]
            if (xmax is not None) and (z[j]>xmax[j]):
                z[j]=xmax[j]
        
        newIndividuals.append(Individual(eval_parameters=set_eval_parameters(eval_parameters,z),objectives=objectives,performance_parameters=performance_parameters))
    return newIndividuals 

def de_rand_2_bin(individuals:List[Individual],objectives:List[Parameter],eval_parameters:List[Parameter],performance_parameters:List[Parameter],min_parents:int,max_parents:int,F:float, C:float):
    """
        Applies mutation and crossover using de_rand_2_bin to a list of individuals 
        Inputs:
            individuals - list of individuals. Takes the best individual[0] (sorted lowest to highest)
            objectives - list of objectives List[Parameter]
            performance_parameters - list of parameters List[parameter]
            F - Amplification Factor [0,2]
            C - Crossover factor [0,1]
        Citatons:
            https://gist.github.com/martinus/7434625df79d820cd4d9
            Storn, R., & Price, K. (1997). Differential Evolution -- A Simple and Efficient Heuristic for global Optimization over Continuous Spaces. Journal of Global Optimization, 11(4), 341–359. https://doi.org/10.1023/A:1008202821328 
            Ao, Y., & Chi, H. (2009). Multi-parent Mutation in Differential Evolution for Multi-objective Optimization. 2009 Fifth International Conference on Natural Computation, 4, 618–622. https://doi.org/10.1109/ICNC.2009.149
    """ 
    # nParents = random.randint(min_parents,max_parents)        
    newIndividuals=list()        
    nIndividuals = len(individuals)
    xmin = individuals[0].eval_parameter_min
    xmax = individuals[0].eval_parameter_max

    min_parents = max([5,min_parents])
    max_parents = max([5,max_parents])
    nparents = random.randint(min_parents,max_parents)  # Number of parents for crossover

    for i in range(len(individuals)):               # Start loop through population  
        x1 = individuals[i].eval_parameters                
        
        p_indicies = get_pairs(nIndividuals,5,[i])  # random pick 3 vectors that are not i
        xp = list()                                 # gets a list of evaluation parameters
        for indx in p_indicies:
            xp.append(individuals[indx].eval_parameters)
        
        k = random.randint(0,len(x1))               # Randomly pick an index to force change
        z = x1*0
        for j in range(len(x1)):
            if (random.random()>C) or (j==k):       # Perform D-1 binomial trials
                z[j] = xp[4][j] + F*(xp[0][j]-xp[1][j]) + F*(xp[2][j]-xp[3][j])
            else:
                z[j] = x1[j] 

            if (xmin is not None) and (z[j]<xmin[j]):
                z[j]=xmin[j]
            if (xmax is not None) and (z[j]>xmax[j]):
                z[j]=xmax[j]
        
        newIndividuals.append(Individual(eval_parameters=set_eval_parameters(eval_parameters,z),objectives=objectives,performance_parameters=performance_parameters))
    return newIndividuals 

def de_best_2_bin(individuals:List[Individual],objectives:List[Parameter],eval_parameters:List[Parameter],performance_parameters:List[Parameter],min_parents:int,max_parents:int,F:float, C:float):
    """
        Applies mutation and crossover using de_rand_2_bin to a list of individuals 
        Inputs:
            individuals - list of individuals. Takes the best individual[0] (sorted lowest to highest)
            objectives - list of objectives List[Parameter]
            performance_parameters - list of parameters List[parameter]
            F - Amplification Factor [0,2]
            C - Crossover factor [0,1]
        Citatons:
            https://gist.github.com/martinus/7434625df79d820cd4d9
            Storn, R., & Price, K. (1997). Differential Evolution -- A Simple and Efficient Heuristic for global Optimization over Continuous Spaces. Journal of Global Optimization, 11(4), 341–359. https://doi.org/10.1023/A:1008202821328 
            Ao, Y., & Chi, H. (2009). Multi-parent Mutation in Differential Evolution for Multi-objective Optimization. 2009 Fifth International Conference on Natural Computation, 4, 618–622. https://doi.org/10.1109/ICNC.2009.149
    """ 
    # nParents = random.randint(min_parents,max_parents)        
    newIndividuals=list()        
    nIndividuals = len(individuals)
    xmin = individuals[0].eval_parameter_min
    xmax = individuals[0].eval_parameter_max

    min_parents = max([4,min_parents])
    max_parents = max([4,max_parents])
    nparents = random.randint(min_parents,max_parents)  # Number of parents for crossover

    x1 = individuals[0].eval_parameters             # Best individual
    for i in range(len(individuals)):               # Start loop through population  
        p_indicies = get_pairs(nIndividuals,4,[i])  # random pick 3 vectors that are not i
        xp = list()                                 # gets a list of evaluation parameters
        for indx in p_indicies:
            xp.append(individuals[indx].eval_parameters)
        
        k = random.randint(0,len(x1))               # Randomly pick an index to force change
        z = x1*0
        for j in range(len(x1)):
            if (random.random()>C) or (j==k):       # Perform D-1 binomial trials
                z[j] = x1[j] + F*(xp[0][j]-xp[1][j]) + F*(xp[2][j]-xp[3][j])
            else:
                z[j] = x1[j] 

            if (xmin is not None) and (z[j]<xmin[j]):
                z[j]=xmin[j]
            if (xmax is not None) and (z[j]>xmax[j]):
                z[j]=xmax[j]
        
        newIndividuals.append(Individual(eval_parameters=set_eval_parameters(eval_parameters,z),objectives=objectives,performance_parameters=performance_parameters))
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

# Helper functions
def get_pairs(nIndividuals:int,nParents:int,parent_indx_seed=[]):
    """
        Get a list of all the pairing partners for a particular individual
        Inputs:
            nIndividuals - number of individuals
            nParents - number of parents 
            parent_indx_seed - pre-populate the parent index array
    """    
    parent_indicies = list()
    for i in range(nParents):
        rand_indx = random.randint(0,nIndividuals-1)
        while(rand_indx in parent_indx_seed):
            rand_indx = random.randint(0,nIndividuals-1)
        parent_indicies.append(rand_indx)
    return parent_indicies

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