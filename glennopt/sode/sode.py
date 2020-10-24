"""
    Single objective differential evolution
"""
from random import shuffle
import operator
import subprocess, copy, math
import sys
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from glennopt.helpers import Parameter, diversity, distance
from glennopt.base import Optimizer, Individual
from glennopt.nsga3.mutate import de_best_1_bin,de_rand_1_bin, mutation_parameters, de_mutation_type, simple,de_rand_1_bin_spawn,de_dmp, get_eval_param_matrix, get_objective_matrix, set_eval_parameters
from random import seed, gauss, random, randint
from tqdm import trange

class SODE(Optimizer):
    def __init__(self,eval_script:str = "evaluation.py", eval_folder:str = "Evaluation",pop_size:int=32, optimization_folder:str=None):
        super().__init__(name="SODE",eval_script=eval_script,eval_folder=eval_folder, opt_folder=optimization_folder)
        '''        
            Inputs:
                eval_script - Evaluation python script that will be called. Either Output.txt is read or an actual output is read. 
                eval_folder - folder to be copied into each individual evaluation directory. If this is null, the population directory isn't created and neither are the individual directories
                num_populations - number of populations to evaluate from the starting population
                pop_size - number of individuals in a given population
                optimization_folder - where optimization should start
        '''
        self.pop_size = pop_size
        self.individuals = None
        self.__mutation_params = mutation_parameters()

    # * Part of initialization    
    def add_eval_parameters(self,eval_params:List[Parameter]):
        self.eval_parameters = eval_params # Sets base class variable

    def add_objectives(self,objectives:List[Parameter]):
        self.objectives = objectives # Sets base class variable

    def add_performance_parameters(self,performance_params:List[Parameter] = None):
        self.performance_parameters = performance_params # Sets base class variable
    # *       

    # * Mutation Properties
    @property
    def mutation_params(self):
        return self.__mutation_params
    @mutation_params.setter
    def mutation_params(self,v):
        self.__mutation_params = v
    # * 

    def __set_eval_parameters__(self,y:np.ndarray):
        """
            only call this function within the class, do not expose to outside. once we have the parameters set, we might need to set the values based on an array.
        """
        parameters = copy.deepcopy(self.eval_parameters)
        for indx in range(len(parameters)):
            parameters[indx].value = y[indx]
        return parameters

    def start_doe(self,doe_size:int=128):
        """
            Starts a design of experiments. If the DOE has already started and there is an output file for an individual then the individual won't be evaluated 
        """        
        doe_individuals = []
        for i in trange(doe_size):
            parameters = copy.deepcopy(self.eval_parameters)
            for eval_param in parameters:
                eval_param.value = np.random.uniform(eval_param.min_value,eval_param.max_value,1)[0]
            doe_individuals.append(Individual(eval_parameters=parameters,objectives=self.objectives, performance_parameters = self.performance_parameters))
        
        # * Begin the evaluation
        self.evaluate_population(individuals=doe_individuals,population_number=-1)
        # * Read the DOE
        individuals = self.read_population(population_number=-1)
        self.append_restart_file(individuals) # Create the restart file

    def optimize_from_population(self,pop_start:int,n_generations:int):
        """
            Reads the values of a population, this can be a DOE or a previous evaluation
            Starts the optimization 

            Inputs:
                pop_start (-1 for DOE), reads the population folder and starts at pop_start+1
                n_generations - number of generations to run for
        """
        # * Read in all the results of the DOE, this should be done by a single thread
        # Check restart file, if not read the population
        
        individuals = self.read_restart_file()

        if (len(individuals)==0):
            individuals = self.read_population(population_number=pop_start)            
        
        # Crossover and Mutate the doe individuals to generate the next individuals used in the population
        # Sort the population into [fill in here]
        # self.__optimize__(individuals=individuals,n_generations=n_generations,pop_start=pop_start+1,params=params,F=F)
        individuals = sorted(individuals, key=operator.attrgetter('objectives'))
        individuals = individuals[:self.pop_size]
        if pop_start == -1: 
            pop_end = n_generations+pop_start
        else:
            pop_end = n_generations +pop_start+1

        for pop in trange(pop_start+1,pop_end):
            newIndividuals = self.__crossover_mutate__(individuals)
            self.evaluate_population(newIndividuals,pop) 
            newIndividuals = self.read_population(pop)
            # Calculate population distance
            pop_diversity = diversity(newIndividuals)
            pop_dist = distance(newIndividuals)
            # Calculate diversity 
            individuals = self.select_individuals(individuals,newIndividuals)
            sorted_inds = sorted(individuals, key=operator.attrgetter('objectives'))
            self.append_restart_file(sorted_inds)
            self.__append_history_file(pop,sorted_inds[0],pop_diversity,pop_dist)
            

    def __crossover_mutate__(self,individuals:List[Individual]):
        '''
            Applies Crossover and Mutate
        '''        
        nIndividuals = len(individuals)
        num_params = len(individuals[0].eval_parameters)        
        if self.mutation_params.mutation_type == de_mutation_type.de_best_1_bin:
            newIndividuals = de_best_1_bin(individuals=individuals,objectives=self.objectives,
                eval_parameters=self.eval_parameters,performance_parameters=self.performance_parameters,
                F=self.mutation_params.F,C=self.mutation_params.C)

        elif self.mutation_params.mutation_type == de_mutation_type.de_rand_1_bin:
            shuffle(individuals)

            newIndividuals = de_rand_1_bin(individuals=individuals,objectives=self.objectives,
                eval_parameters=self.eval_parameters,performance_parameters=self.performance_parameters,
                F=self.mutation_params.F,C=self.mutation_params.C)

        elif self.mutation_params.mutation_type == de_mutation_type.simple:
            shuffle(individuals)
            nCrossover = int(self.pop_size/2)

            nMutation = self.pop_size-nCrossover
            newIndividuals = simple(individuals=individuals,nCrossover=nCrossover,nMutation=nMutation,objectives=self.objectives,eval_parameters=self.eval_parameters,performance_parameters=self.performance_parameters,mu=self.mutation_params.mu,sigma=self.mutation_params.sigma)

        elif self.mutation_params.mutation_type == de_mutation_type.de_rand_1_bin_spawn:
            shuffle(individuals)
            parents = individuals[0:self.mutation_params.nParents]
            
            newIndividuals = de_rand_1_bin_spawn(individuals=parents,objectives=self.objectives,
                eval_parameters=self.eval_parameters,performance_parameters=self.performance_parameters,
                F=self.mutation_params.F,C=self.mutation_params.C,num_children=len(individuals))

        else:# self.mutation_params.mutation_type==mutation_parameters.de_dmp:
            newIndividuals = de_dmp(individuals=individuals,
                objectives=self.objectives, eval_parameters=self.eval_parameters, performance_parameters=self.performance_parameters)

        return newIndividuals
    
    def select_individuals(self,prevIndividuals:List[Individual],newIndividuals:List[Individual]):
        '''
            Select individuals using diversity and distance. Use this only for single objective type problems. This is not suitable for multi-objective. 

            Inputs:
                previndividuals - previous population
                newIndividuals - new population
            Citations:
                (described in) Ghosh, A., Das, S., Mallipeddi, R., Das, A. K., & Dash, S. S. (2017). A Modified Differential Evolution With Distance-based Selection for Continuous Optimization in Presence of Noise. IEEE Access, 5, 26944–26964. https://doi.org/10.1109/ACCESS.2017.2773825

                (Modified version of) S. Das, A. Konar, and U. K. Chakraborty, ‘‘Improved differential evolution algorithms for handling noisy optimization problems,’’ in Proc. IEEE Congr. Evol. Comput., vol. 2. Sep. 2005, pp. 1691–1698.
        '''
        F = get_objective_matrix(prevIndividuals)
        F_new = get_objective_matrix(newIndividuals)
        
        U = get_eval_param_matrix(newIndividuals)
        X = get_eval_param_matrix(prevIndividuals)
        
        deltaF = np.absolute(F-F_new)   # Compute the delta objective
        dist = np.absolute(U-X)         # Calculate distance

        individuals = list()
        for i in range(len(newIndividuals)):
            if F_new[i]/F[i] < 1: 
                individuals.append(newIndividuals[i])                                               # X_new[i,:] = U[i,:]
            elif (F_new[i]/F[i] > 1) and (random.random() <= math.exp(-deltaF[i]/dist[i])):
                individuals.append(newIndividuals[i])                                               # X_new[i,:] = U[i,:]
            else:
                individuals.append(prevIndividuals[i])                                              # X_new[i,:] = X[i,:]
        return individuals