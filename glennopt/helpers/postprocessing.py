from ..nsga3.non_dominated_sorting import non_dominated_sorting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def get_pop_best(individuals):
    '''
        Gets the best individuals from each population (not rolling best)
        typically you would use opt.read_calculation_folder() where opt is an object representing your nsga3 or sode class.

        Returns:
            best_individuals - this is an array of individuals that are best at each objective
                [ 
                    POP001: [best_individual_objective1, best_individual,objective2, best_individual,objective3], best_individual_compromise
                    POP002: [best_individual_objective1, best_individual,objective2, best_individual,objective3], best_individual_compromise
                    POP003: [best_individual_objective1, best_individual,objective2, best_individual,objective3], best_individual_compromise
                ]
            comp_individuals - this is an array of individuals that is the best compromise between all the objectives
    '''
    # Read calculation folder
    
    # (Compromise Target) At the minimum index of each objective what are the values of the other objectives
    nobjectives = len(individuals[0].objectives)
    best_fronts = list()

    best_individuals = dict()
    dist = list()
    dist_temp = list()
    prev_pop = individuals[0][0].population

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
            
        # Lets find the miniumum distance and best compromise for a given population
        min_indx = dist_temp.index(min(dist_temp))
        dist.append(dist_temp[min_indx])
        dist_temp.clear()
        prev_pop = pop
    return best_individuals, best_fronts

def get_best(individuals):
    '''
        Gets the best individual vs Pop
        Some populations won't generate a better design but the best design will always be carried to the next population for crossover + mutation     
        Returns:
            objectives - list of best objective values for each population. For multi-objective problems use best fronts for better representation of design space
            pop_folders - this is a list of populations 
            best_fronts - List of individuals contained in best fronts, empty list if single objective
    '''
    best_individuals, pop_folders = get_pop_best()
    objectives = list()
    nobjectives = len(individuals[0].objectives)
    for i in range(len(pop_folders)):
        key = pop_folders[i]
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
        for pop_individuals in individuals:
            rolling_best.extend(pop_individuals)
            best_fronts.append(non_dominated_sorting(rolling_best,len(pop_individuals),True))                
    return objectives, pop_folders, best_fronts

def plot_best_pop(individuals,objective_index):
    """
        Creates a plot of the population vs the objective value
        INPUTS:
            objective_index - which objective to compare, index starts at 0 
    """
    best_objectives,pop_folders,best_fronts = get_pop_best(individuals)
    
    indx = 0
    legend_labels = list()
    
    objective_data = list()
    best_indivduals = list()
    for _,ind in best_objectives.items():
        objective_data.append(ind[objective_index].objectives[objective_index])
        best_indivduals.append(ind[objective_index].name)
    
    fig,ax = plt.subplots()
    colors = cm.rainbow(np.linspace(0, 1, len(pop_folders)))
    ax.scatter(list(best_objectives.keys()), objective_data,color='blue',s=5)
    ax.set_yscale('log')
    ax.set_xticks(list(best_objectives.keys()))
    ax.set_xlabel('Population')
    ax.set_ylabel('Objective Value')
    ax.set_title('Objective Index: ' + str(objective_index))
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show()

def plot_best_objective(individuals,objective_index):
    '''
        Creates a plot of the best objective vs population number
        INPUTS:
            objective_index - objective to plot
    '''
    best_objectives, pop_folders = get_best(individuals)
    best_objectives = convert_to_ndarray(best_objectives)

    indx = 0
    
    fig,ax = plt.subplots()
    colors = cm.rainbow(np.linspace(0, 1, len(self.pandas_cache.keys())))
    ax.scatter(pop_folders, best_objectives,color='blue',s=5)
    ax.set_yscale('log')
    ax.set_xticks(pop_folders)
    ax.set_xlabel('Population')
    ax.set_ylabel('Objective Value')
    ax.set_title('Objective Index: ' + str(objective_index))
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show()