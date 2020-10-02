"""
    Single objective differential evolution
"""
from random import shuffle
import operator
import subprocess, copy, math
import sys
sys.path.insert(0,'../')
# import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from glennopt.base_classes import Optimizer, Parameter, Individual
from glennopt.doe import generate_reference_points
from glennopt.helpers import get_pairs
from glennopt.nsga3.mutate import mutate_crossover_de
from random import seed, gauss, random
from typing import List

individual_list = List[Individual]
param_list = List[Parameter]

class SingleObjectiveDE(Optimizer):
    def __init__(self,eval_script:str = "evaluation.py", eval_folder:str = "Evaluation",num_populations:int=10,pop_size:int=32, optimization_folder:str=None, beta:List[float]=[0.2,0.8],cr:float=0.2):
        super().__init__(name="SingleObjectiveDE",eval_script=eval_script,eval_folder=eval_folder, opt_folder=optimization_folder)
        '''        
            Inputs:
                eval_script - Evaluation python script that will be called. Either Output.txt is read or an actual output is read. 

                eval_folder - folder to be copied into each individual evaluation directory. If this is null, the population directory isn't created and neither are the individual directories

                num_populations - number of populations to evaluate from the starting population

                pop_size - number of individuals in a given population

                optimization_folder - where optimization should start
        '''
        self.num_populations = num_populations
        self.pop_size = pop_size
        self.individuals = None        
        self.beta = beta
        self.cr = cr
    
    # * Part of initialization    
    def add_eval_parameters(self,eval_params:param_list):
        self.eval_parameters = eval_params # Sets base class variable

    def add_objectives(self,objectives:param_list):
        self.objectives = objectives # Sets base class variable

    def add_performance_parameters(self,performance_params:param_list = None):
        self.performance_parameters = performance_params # Sets base class variable
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
        for i in range(doe_size):
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
        n_individuals = len(individuals)

        if (len(individuals)==0):
            individuals = self.read_population(population_number=pop_start)            
        
        # Crossover and Mutate the doe individuals to generate the next individuals used in the population
        # Sort the population into [fill in here]
        # self.__optimize__(individuals=individuals,n_generations=n_generations,pop_start=pop_start+1,params=params,F=F)
        all_inds = copy.deepcopy(individuals)
        # best_sc = np.zeros(n_generations)
        for pop in range(pop_start+1,n_generations):
            newIndividuals = self.__crossover_mutate__(individuals)
            self.evaluate_population(newIndividuals,pop) 
            newIndividuals = self.read_population(pop)

            all_inds.extend(newIndividuals) # add the previous population to the pool 
            # Sort and select
            sorted_inds =  sorted(all_inds, key=operator.attrgetter('objectives'))
            individuals = sorted_inds[:n_individuals]
            shuffle(individuals)


    # */ Differential Evolution Functions /*
    def __optimize__(self,individuals:individual_list,n_generations:int,pop_start:int,params:dict,F:list):
        ''' 
            Differential Evolution Main loop

            Inputs:
                individuals - list of individuals to evaluate
                n_generations - number of generations to loop through
                pop_start - starting population number
        '''
        nIndividuals = len(individuals)



    def __crossover_mutate__(self,individuals:individual_list):
        '''
            
        '''
        
        nIndividuals = len(individuals)
        num_params = len(individuals[0].eval_parameters)
        
        newIndividuals = list()
        for i in range(nIndividuals): # For each individual
            x = np.copy(individuals[i].eval_parameters)
            indicies = get_pairs(nIndividuals,nParents=3)
            xparents = [individuals[j].eval_parameters for j in indicies]
            z = mutate_crossover_de(x,xparents)
            newIndividuals.append(Individual(eval_parameters=self.__set_eval_parameters__(z),objectives=self.objectives,performance_parameters=self.performance_parameters))

        return newIndividuals