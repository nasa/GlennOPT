import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from tqdm import trange
from typing import List

from ..base import Individual
from .convert_to_ndarray import convert_to_ndarray
from .nsga_functions import non_dominated_sorting

def get_best(individuals:List[Individual],pop_size:int):
    """Gets the best individual vs Pop. Some populations won't generate a better design but the best design will always be carried to the next population for crossover + mutation     
       
        Important: Call this function with inputs from ``individuals = ns.read_calculation_folder()``

    Args:
        individuals (List[Individual]): Takes in a list of the individuals
        pop_size (int): population size i.e. how many individuals

    Returns:
        (tuple): tuple containing:

            **objectives** (List[Parameter]): numpy array of best objective values for each population. For multi-objective problems use best fronts for better representation of design space

            **pop_folders** (List[int]): this is a list of populations 

            **best_fronts** (List[List[Individual]]): List of individuals contained in best fronts, empty list if single objective
    """

    best_individuals,best_fronts = get_pop_best(individuals)

    objectives = list()
    nobjectives = len(individuals[0][0].objectives)
    keys = list(best_individuals.keys())
    for i in range(len(individuals)):
        key = keys[i]
        temp_objectives = list()
        if i == 0:
            for o in range(nobjectives):
                temp_objectives.append(best_individuals[key][o].objectives[o])
        else:
            for o in range(nobjectives):
                if best_individuals[key][o].objectives[o] < objectives[-1][o]:
                    temp_objectives.append(best_individuals[key][o].objectives[o])
                else:
                    temp_objectives.append(objectives[-1][o])
        objectives.append(temp_objectives)
    
    # Read calculation folder
    best_fronts = list()
    if nobjectives>1:
        rolling_best = list()
        for i in trange(len(individuals), desc='Running Non-dimensional sorting'):
            pop_individuals = individuals[i]
            rolling_best.extend(pop_individuals)
            best_fronts.append(non_dominated_sorting(rolling_best,len(pop_individuals),True))
            if (len(rolling_best)>pop_size):                
                rolling_best = rolling_best[-pop_size:] # Keep only the pop_size
        
    pop_folders = keys
    return convert_to_ndarray(objectives), pop_folders, best_fronts  # First front

def get_pop_best(individuals:List[Individual]):
    '''
        Gets the best individuals from each population (not rolling best)
        typically you would use opt.read_calculation_folder() where opt is an object representing your nsga3 or sode class.
        
        Note: Important: Call this function with inputs from `individuals = ns.read_calculation_folder()`

        Returns:
            (tuple): tuple containing:

            **best_individuals** (List[Dict[str,List[Individual]]]): this is an array of individuals that are best at each objective

                |    [ 
                |        POP001: [best_individual_objective1, best_individual,objective2, best_individual,objective3], best_individual_compromise
                |        POP002: [best_individual_objective1, best_individual,objective2, best_individual,objective3], best_individual_compromise
                |        POP003: [best_individual_objective1, best_individual,objective2, best_individual,objective3], best_individual_compromise
                |    ]

            **comp_individuals** (List[Individual]): this is an array of individuals that is the best compromise between all the objectives

    '''
    # Read calculation folder
    
    # (Compromise Target) At the minimum index of each objective what are the values of the other objectives
    nobjectives = len(individuals[0][0].objectives)
    best_fronts = list()

    best_individuals = dict()

    for pop_individuals in individuals:
        pop = pop_individuals[0].population
        if nobjectives>1:
            best_fronts.append(non_dominated_sorting(pop_individuals,len(pop_individuals),True))
        
        for ind in pop_individuals:
            if pop not in best_individuals.keys():        # Prepopulate
                best_individuals[pop] = list()
                for o in range(nobjectives):
                    best_individuals[pop].append(ind) 
            else:                                       # Compare   
                for o in range(nobjectives): # Checks for the best objective
                    current_best = best_individuals[pop][o].objectives[o]
                    if ind.objectives[o]<current_best:
                        best_individuals[pop][o] = ind
    
    return best_individuals, best_fronts

def plot_pareto(best_fronts,pop,objective1_index,objective2_index):
    '''

    '''
    fig,ax = plt.subplots()

    colors = cm.rainbow(np.linspace(0, 1, len(best_fronts)))        
    indx = 0
    legend_labels = []
    # Scan the pandas file, grab objectives for each population
    for ind_list in best_fronts:
        obj1_data = []
        obj2_data = []
        c=colors[indx]
        for ind in ind_list:
            obj1_data.append(ind.objectives[objective1_index])
            obj2_data.append(ind.objectives[objective2_index])
        # Plot the gathered data
        ax.scatter(obj1_data, obj2_data, color=c, s=10,alpha=0.5)
        legend_labels.append(pop[indx])
        indx+=1

    ax.set_xlabel(obj1_name)
    ax.set_ylabel(obj2_name)
    if xlim is not None:
        ax.set_xlim(xlim[0],xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0],ylim[1])
    ax.legend(legend_labels)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show()
