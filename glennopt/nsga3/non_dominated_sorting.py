import copy
import numpy as np
from collections import defaultdict
from numpy import linalg as LA
from typing import TypeVar,List
from ..base import Individual

def non_dominated_sorting(individuals:List[Individual],k:int,first_front_only=False):
    '''
        Loops through the list of individuals and checks which one
        Inputs:
            fitnesses - array of objectives
            k - number of individuals to select 
            first_front_only - gets only the best individuals
    '''
    
    def dominates(x:np.ndarray,y:np.ndarray) -> bool:
        '''
            Returns true if all or any the objectives of x are less than y
        '''
        b = np.all(x <= y) & np.any(x<y)
        return b

    map_fit_ind = defaultdict(list)
    for ind in individuals:
        map_fit_ind[ind].append(ind)
    fits = list(map_fit_ind.keys())

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    # Rank first Pareto front
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i+1:]:
            if dominates(fit_i.objectives,fit_j.objectives):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif dominates(fit_j.objectives,fit_i.objectives):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    # Rank the next front until all individuals are sorted or
    # the given number of individual are sorted.
    if not first_front_only:
        N = min(len(individuals), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts
