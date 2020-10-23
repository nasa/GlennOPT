import numpy as np
from ..base_classes.individual import Individual
from typing import List
import math


def diversity(individuals:List[Individual]):
    '''
        Computes the diversity of a population. Higher diversity value the better.

        Inputs:
            individuals - number of individuals 
        J. Hu, J. Zeng, and Y. Tan, ‘‘A diversity-guided particle swarm optimizer for dynamic environments,’’ in Bio-Inspired Computational Intelligence and Applications. Berlin, Germany: Springer, 2007, pp. 239–247
    '''
    Np = len(individuals)
    D = len(individuals[0].eval_parameters)
    x_avg = 1/len(individuals) * np.sum(np.array([ind.eval_parameters for ind in individuals]),axis=0)
    
    L = np.zeros(Np)
    for i in range(individuals):        
        x = individuals[i]
        for j in range(D):            
            L[i] += (x[j]-x_avg[j])**2
        L[i] = math.sqrt(L[i])
    
    pop_diversity = 1/(Np*np.max(L)) * np.sum(L)
    return pop_diversity

def distance(individuals:List[Individual], newIndividuals:List[Individual]):
    pop, _ , _ = get_eval_param_matrix(individuals)
    newPop, _ , _ = get_eval_param_matrix(newIndividuals)
    dist = np.absolute(pop - newPop)   # Compute distance
    return dist