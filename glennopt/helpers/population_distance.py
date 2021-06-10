import numpy as np
from ..base import Individual, Parameter
from typing import List
import math
from .mutate import get_eval_param_matrix, get_objective_matrix, set_eval_parameters
import random

def diversity(individuals:List[Individual]) -> float:
    """Computes the diversity of a population. Higher diversity value the better.

    Citations: 
        J. Hu, J. Zeng, and Y. Tan, ‘‘A diversity-guided particle swarm optimizer for dynamic environments,’’ in Bio-Inspired Computational Intelligence and Applications. Berlin, Germany: Springer, 2007, pp. 239–247

    Args:
        individuals (List[Individual]): List of individuals of a population 

    Returns:
        float: diversity value 
    """
    Np = len(individuals)
    D = len(individuals[0].eval_parameters)
    x_avg = 1/len(individuals) * np.sum(np.array([ind.eval_parameters for ind in individuals]),axis=0)
    
    L = np.zeros(Np)
    for i in range(Np):        
        x = individuals[i].eval_parameters
        for j in range(D):            
            L[i] += (x[j]-x_avg[j])**2
        L[i] = math.sqrt(L[i])
    
    pop_diversity = 1/(Np*np.max(L)) * np.sum(L)
    return pop_diversity

def distance(individuals:List[Individual], newIndividuals:List[Individual]):
    """Calculates the distance between individuals

    Args:
        individuals (List[Individual]): past list of individuals 
        newIndividuals (List[Individual]): updated list of individuals 

    Returns:
        float: distance
    """
    pop, _ , _ = get_eval_param_matrix(individuals)
    newPop, _ , _ = get_eval_param_matrix(newIndividuals)
    dist = np.sum(np.absolute(pop - newPop),axis=1)   # Compute distance
    return np.average(dist)

