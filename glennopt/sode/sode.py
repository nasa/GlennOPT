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
from ..helpers import Parameter
from ..base_classes import Optimizer, Individual
from ..nsga3.mutate import de_best_1_bin,de_best_2_bin,de_rand_1_bin,de_rand_2_bin, mutation_parameters, de_mutation_type
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
        
        for pop in trange(pop_start+1,pop_start+n_generations):
            newIndividuals = self.__crossover_mutate__(individuals)
            self.evaluate_population(newIndividuals,pop) 
            newIndividuals = self.read_population(pop)

            individuals.extend(newIndividuals) # add the previous population to the pool 
            # Sort and select
            sorted_inds =  sorted(individuals, key=operator.attrgetter('objectives'))            
            self.append_restart_file(sorted_inds)
            individuals = sorted_inds[:self.pop_size]
            

    def __crossover_mutate__(self,individuals:List[Individual]):
        '''
            Applies Crossover and Mutate
        '''
        
        nIndividuals = len(individuals)
        num_params = len(individuals[0].eval_parameters)        
        if self.mutation_params.mutation_type == de_mutation_type.de_best_1_bin:
            newIndividuals = de_best_1_bin(individuals=individuals,objectives=self.objectives,
                min_parents=self.mutation_params.min_parents,max_parents=self.mutation_params.max_parents,
                eval_parameters=self.eval_parameters,performance_parameters=self.performance_parameters,
                F=self.mutation_params.F,C=self.mutation_params.C)
        elif self.mutation_params.mutation_type == de_mutation_type.de_best_2_bin:
            newIndividuals = de_best_2_bin(individuals=individuals,objectives=self.objectives,
                min_parents=self.mutation_params.min_parents,max_parents=self.mutation_params.max_parents,
                eval_parameters=self.eval_parameters,performance_parameters=self.performance_parameters,
                F=self.mutation_params.F,C=self.mutation_params.C)
        elif self.mutation_params.mutation_type == de_mutation_type.de_rand_1_bin:
            newIndividuals = de_rand_1_bin(individuals=individuals,objectives=self.objectives,
                min_parents=self.mutation_params.min_parents,max_parents=self.mutation_params.max_parents,
                eval_parameters=self.eval_parameters,performance_parameters=self.performance_parameters,
                F=self.mutation_params.F,C=self.mutation_params.C)
        elif self.mutation_params.mutation_type == de_mutation_type.de_rand_2_bin:
            newIndividuals = de_rand_2_bin(individuals=individuals,objectives=self.objectives,
                min_parents=self.mutation_params.min_parents,max_parents=self.mutation_params.max_parents,
                eval_parameters=self.eval_parameters,performance_parameters=self.performance_parameters,
                F=self.mutation_params.F,C=self.mutation_params.C)
        elif self.mutation_params.mutation_type == de_mutation_type.simple:
            newIndividuals = mutation_simple(individuals=individuals,nCrossover=nParents,nMutation=nParents,objectives=self.objectives,eval_parameters=self.eval_parameters,performance_parameters=self.performance_parameters,mu=self.mutation_params.mu,sigma=self.mutation_params.sigma)

        return newIndividuals