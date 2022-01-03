import numpy as np
from collections import defaultdict
from typing import Tuple,List
from ..base import Individual
from itertools import chain

def non_dominated_sorting(individuals:List[Individual],k:int,first_front_only=False):
    """Loops through the list of individuals and sorts through them. 

    Citation: 
        Yuan, Y., Xu, H., & Wang, B. (2014). An Improved NSGA-III Procedure for Evolutionary Many-objective Optimization. Genetic and Evolutionary Computation Conference (GECCO 2014), 661–668. https://doi.org/10.1145/2576768.2598342

    Args:
        individuals (List[Individual]): all individuals of a population 
        k (int): number of individuals to select 
        first_front_only (bool, optional): gets only the best individuals. Defaults to False.

    Returns:
        List[Individual]: List containing fronts so List of lists of individuals. 
    """
    def dominates(x:np.ndarray,y:np.ndarray) -> bool:
        """Returns true if all or any the objectives of x are less than y

        Args:
            x (np.ndarray): a set of individual x's objective values 
            y (np.ndarray): a set of individual y's objective values 

        Returns:
            bool: True if individual x dominates individual y 
        """        
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


def find_extreme_points(fitnesses, best_point, extreme_points=None):
    """Finds the individuals with extreme values for each objective function. These definitions need to be updated. I used to know all this. 

    Warning:
        Don't call this directly unless you know what you are doing

    Args:
        fitnesses ([ndarray]): how close individuals are to best point.1
        best_point ([ndarray]): array containing the best points
        extreme_points ([ndarray], optional): Points on the extreme of either objective 1 or objective 2. Defaults to None.

    Returns:
        ndarray: fitness of individuals. Description needs update

    """
    
    # Keep track of last generation extreme points
    if extreme_points is not None:
        fitnesses = np.concatenate((fitnesses, extreme_points), axis=0)

    # Translate objectives
    ft = fitnesses - best_point

    # Find achievement scalarizing function (asf)
    asf = np.eye(best_point.shape[0])
    asf[asf == 0] = 1e6
    asf = np.max(ft * asf[:, np.newaxis, :], axis=2)

    # Extreme point are the fitnesses with minimal asf
    min_asf_idx = np.argmin(asf, axis=1)
    return fitnesses[min_asf_idx, :]

def find_intercepts(extreme_points:List[np.ndarray], best_point:np.ndarray, current_worst:np.ndarray, front_worst:np.ndarray):
    """Find intercepts between the hyperplane defined by the extreme and best point and each axis with the ideal point as origin.

        Think of the hyperplane as a plane defined with one end being the best (minimum) and the other end being the extreme point (worst)
    
    Warning:
        Don't call this directly unless you know what you are doing

    Args:
        extreme_points (List[np.ndarray]): List of the extreme points [best, worst]. Example for 2 objectives [[16.34, - 4.07], [7.28, 6.69]]
        best_point (np.ndarray): best that minimizes all objectives. Example for 2 objectives [7.28, -4.07]
        current_worst (np.ndarray): worst individual that maximizes all objectives. Example for 2 objectives [16.34, 6.69]
        front_worst (np.ndarray): worst individual on pareto front. 

    Returns:
        np.ndarray: objective values that intercept the hyperplane axis. Example for 2 objectives [16.34, 6.69]
    """
    # Construct hyperplane sum(f_i^n) = 1
    b = np.ones(extreme_points.shape[1])    # np.ones is used to construct the axis    
    A = extreme_points - best_point         # Hyper plane with one end at best and the other at worst point 
    try:
        x = np.linalg.solve(A, b)           # Solves for where the plane intercepts the axis
    except np.linalg.LinAlgError:
        intercepts = current_worst
    else:
        intercepts = 1 / x

        if (not np.allclose(np.dot(A, x), b) or
                np.any(intercepts <= 1e-6) or
                np.any((intercepts + best_point) > current_worst)):
            intercepts = front_worst

    return intercepts


def associate_to_niche(fitnesses:np.ndarray, reference_points:np.ndarray, best_point:np.ndarray, intercepts:np.ndarray):
    """Associates individuals to reference points and calculates niche number. Corresponds to Algorithm 3 of Deb & Jain (2014).

    Args:
        fitnesses (np.ndarray): multi-dimensional array of unnormalized objective values from all individuals.
        reference_points (np.ndarray): multi-dimensional array representing the ideal normalized pareto front/surface. Example for 2 objectives [ [0,1], [0.25, 0.75],[0.5, 0.5], [0.75, 0.25], [1, 0] ]
        best_point (np.ndarray): Array representing the best individual's objective. Example for 2 objectives [7.28, -4.07]
        intercepts (np.ndarray): [description]

    Returns:
        [type]: [description]
    """
    # Normalize by ideal point and intercepts
    fn = (fitnesses - best_point) / (intercepts - best_point)

    # Create distance matrix
    fn = np.repeat(np.expand_dims(fn, axis=1), len(reference_points), axis=1)
    norm = np.linalg.norm(reference_points, axis=1)

    distances = np.sum(fn * reference_points, axis=2) / norm.reshape(1, -1)
    distances = distances[:, :, np.newaxis] * reference_points[np.newaxis, :, :] / norm[np.newaxis, :, np.newaxis]
    distances = np.linalg.norm(distances - fn, axis=2)

    # Retrieve min distance niche index
    niches = np.argmin(distances, axis=1)
    distances = distances[range(niches.shape[0]), niches]
    return niches, distances

def niching(individuals, k, niches, distances, niche_counts):
    """[summary]

    Args:
        individuals ([type]): [description]
        k ([type]): [description]
        niches ([type]): [description]
        distances ([type]): [description]
        niche_counts ([type]): [description]

    Returns:
        [type]: [description]
    """
    selected = []
    available = np.ones(len(individuals), dtype=np.bool)
    while len(selected) < k:
        # Maximum number of individuals (niches) to select in that round
        n = k - len(selected)

        # Find the available niches and the minimum niche count in them
        available_niches = np.zeros(len(niche_counts), dtype=np.bool)
        available_niches[np.unique(niches[available])] = True
        min_count = np.min(niche_counts[available_niches])

        # Select at most n niches with the minimum count
        selected_niches = np.flatnonzero(np.logical_and(available_niches, niche_counts == min_count))
        np.random.shuffle(selected_niches)
        selected_niches = selected_niches[:n]

        for niche in selected_niches:
            # Select from available individuals in niche
            niche_individuals = np.flatnonzero(np.logical_and(niches == niche, available))
            np.random.shuffle(niche_individuals)

            # If no individual in that niche, select the closest to reference
            # Else select randomly
            if niche_counts[niche] == 0:
                sel_index = niche_individuals[np.argmin(distances[niche_individuals])]
            else:
                sel_index = niche_individuals[0]

            # Update availability, counts and selection
            available[sel_index] = False
            niche_counts[niche] += 1
            selected.append(individuals[sel_index])
    return selected


def uniform_reference_points(nobj, p=4, scaling=None):
    """Generate reference points uniformly on the hyperplane intersecting each axis at 1. The scaling factor is used to combine multiple layers of reference points.

    Args:
        nobj (int): number of objectives
        p (int, optional): number of references per objective. Defaults to 4.
        scaling ([type], optional): [description]. Defaults to None.
    
    Returns:
        List[Tuple(float)]: List of normalized objective function values representing the reference minimum values of each objective. Think of it as a reference pareto front/surface/(whatever in 3+ dimensions )
        
    """
    def gen_refs_recursive(ref, nobj, left, total, depth):
        points = []
        if depth == nobj - 1:
            ref[depth] = left / total
            points.append(ref)
        else:
            for i in range(left + 1):
                ref[depth] = i / total
                points.extend(gen_refs_recursive(ref.copy(), nobj, left - i, total, depth + 1))
        return points

    ref_points = np.array(gen_refs_recursive(np.zeros(nobj), nobj, p, p, 0))
    if scaling is not None:
        ref_points *= scaling
        ref_points += (1 - scaling) / nobj

    return ref_points

def sort_and_select_population(individuals:List[Individual], reference_points:np.ndarray,pop_size:int):
        """Takes a list of individuals, finds the fronts and the best designs
            
            Code is a combination from deap and yarpiz
            https://github.com/DEAP
            https://yarpiz.com/456/ypea126-nsga3

        Args:
            individuals (List[Individual]): List of individuals from a DOE or POP or anything really
            reference_points (np.ndarray): reference points along the pareto front.  
            pop_size (int): Population size to consider 

        Raises:
            Exception: something bad has happened

        Returns:
            (tuple): containing

                **chosen** (List[List[int]]): individuals along front in order of best to worst front 
                **best_point** (int): index of best individual 
                **worst_point** (int): index of worst individual 
                **extreme_points** (List[int]):  indexies of the extreme point 
        """
        
        if (pop_size>len(individuals)):
            raise Exception("population size needs to be <= the number of individuals")
        

        pareto_fronts = non_dominated_sorting(individuals,pop_size)
        fitnesses = np.array([ind.objectives for f in pareto_fronts for ind in f])
        fitnesses *= -1

        best_point = np.min(fitnesses,axis=0)
        worst_point = np.max(fitnesses,axis=0)

        extreme_points = find_extreme_points(fitnesses,best_point)
        front_worst = np.max(fitnesses[:sum(len(f) for f in pareto_fronts),:],axis=0)
        intercepts = find_intercepts(extreme_points,best_point,worst_point,front_worst)
        niches, dist = associate_to_niche(fitnesses, reference_points, best_point, intercepts)
        
        # Get counts per niche for individuals in all front but the last
        niche_counts = np.zeros(len(reference_points), dtype=int)
        index, counts = np.unique(niches[:-len(pareto_fronts[-1])], return_counts=True)
        niche_counts[index] = counts

        # Choose individuals from all fronts but the last
        chosen = list(chain(*pareto_fronts[:-1]))

        # Use niching to select the remaining individuals
        sel_count = len(chosen)
        n = pop_size - sel_count
        selected = niching(pareto_fronts[-1], n, niches[sel_count:], dist[sel_count:], niche_counts)
        chosen.extend(selected)

        return chosen, best_point, worst_point, extreme_points
