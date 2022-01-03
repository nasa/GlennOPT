"""
    NSGA3 - Non-dominated sorting genetic algorithm
"""
import os, shutil
import subprocess, copy, math
from random import seed, gauss,random,uniform,randint
from typing import List
from dataclasses import dataclass, field
import numpy as np 
import glob
from tqdm import trange
from ..helpers import diversity, distance
from ..helpers import non_dominated_sorting, find_extreme_points, find_intercepts, associate_to_niche, niching, uniform_reference_points, sort_and_select_population
from ..base import Parameter, Individual, Optimizer
from ..helpers import de_best_1_bin,de_rand_1_bin, mutation_parameters, de_mutation_type, simple,de_rand_1_bin_spawn,de_dmp, get_eval_param_matrix, get_objective_matrix, set_eval_parameters

individual_list = List[Individual]

class NSGA3(Optimizer):
    def __init__(self,eval_command:str = "python evaluation.py", eval_folder:str = "Evaluation",pop_size:int=128, optimization_folder:str=None,single_folder_eval=False, overwrite_input_file=False, pareto_resolution:int=4):
        """
            NSGA-3 multi-dimensional optimizer. This version has been tweaked to include restart capabilities. It can also keep track of additional parameters that can be considered part of the constraints.

            Each evaluation can occur in a separate folder (simulations) or without folders (analytical)
            https://www.egr.msu.edu/~kdeb/papers/k2012009.pdf

        Args:
            eval_command (str, optional): Command that will be executed in the evaluation folder. Defaults to "python evaluation.py".
            eval_folder (str, optional): folder to be copied into each individual evaluation directory. If this is null, the population directory isn't created and neither are the individual directories. Defaults to "Evaluation".
            pop_size (int, optional): number of populations to evaluate from the starting population. Defaults to 128.
            optimization_folder (str, optional): Folder where the optimization and doe work should be stored in. Defaults to None.
            single_folder_eval (bool, optional): where optimization should start. Defaults to False.
            overwrite_input_file(bool, optional): Specifies whether or not to overwrite the input file when restarting a simulation. Defaults to False.
            pareto_resolution (int, optional): This specifies the number of reference points on the pareto front, 4 is good for 2 objectives. This number should increase if more objectives are needed. 2^n objectives something like that would work. 

        """
        super().__init__(name="nsga3",eval_command=eval_command,eval_folder=eval_folder, opt_folder=optimization_folder,single_folder_eval=single_folder_eval,overwrite_input_file=overwrite_input_file)
        
        self.pop_size = pop_size
        self.individuals = None 
        self.pareto_resolution = pareto_resolution       
        self.__mutation_params = mutation_parameters()
        
        
    def add_eval_parameters(self,eval_params:List[Parameter]):
        """Add evaluation parameters. This is part of the initialization    

        Args:
            eval_params (List[Parameter]): Add in a list of evaluation parameters 

        """
        self.eval_parameters = eval_params # Sets base class variable

    def add_objectives(self,objectives:List[Parameter]):
        """Add the objectives 

        Args:
            objectives (List[Parameter]): [description]
        """
        self.objectives = objectives # Sets base class variable

    def add_performance_parameters(self,performance_params:List[Parameter] = None):
        """Add performance parameters 

        Args:
            performance_params (List[Parameter], optional): [description]. Defaults to None.
        """
        self.performance_parameters = performance_params # Sets base class variable
      
    
    # * Mutation Properties
    @property
    def mutation_params(self):
        """Get Mutation parameters

        Returns:
            mutation_parameters: parameter class describes the mutation
        """
        return self.__mutation_params

    @mutation_params.setter
    def mutation_params(self,v:mutation_parameters):
        """Setter for mutation parameters 

        Args:
            v (mutation_parameters): class describing the mutation parameter s
        """
        self.__mutation_params = v
    # * 

    def start_doe(self,doe_individuals:List[Individual]=None,doe_size:int=128):
        """Starts a design of experiments. This generates the parameters for the individuals to be evaluated and executes each case. If the DOE has already started and there is an output file for an individual then the individual won't be evaluated    

        Args:
            doe_individuals (List[Individual], optional): List of individuals. Defaults to None.
            doe_size (int, optional): [description]. Defaults to 128.
        """
        if doe_individuals is None:
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
        self.append_restart_file(individuals)
        
        if self.single_folder_eval: 
            # Delete the population folder
            population_folder = os.path.join(self.optimization_folder,self.__check_population_folder__(-1))
            if os.path.isdir(population_folder):
                shutil.rmtree(population_folder)

       

    def optimize_from_population(self,pop_start:int,n_generations:int):
        """Reads the values of a population, this can be a DOE or a previous evaluation
            Starts the optimization 

            Inputs:
                pop_start (-1 for DOE), reads the population folder and starts at pop_start+1
                n_generations - number of generations to run for
        """
        # * Read in all the results of the DOE, this should be done by a single thread
        # Check restart file, if not read the population
        self.load_history_file()
        individuals = self.read_restart_file()

        if (len(individuals)==0):                               
            individuals = self.read_calculation_folder()

        if (len(individuals)<self.pop_size):
            raise Exception("Number of individuals in the restart file is less than the population size."
                + " lower the population size or increase the DOE count(if restarting from a DOE)")

        # Crossover and Mutate the doe individuals to generate the next individuals used in the population
        # Sort the population into [fill in here]
        ref_points = uniform_reference_points(len(self.objectives), p=self.pareto_resolution, scaling=None)
        individuals,best_point, worst_point, extreme_points = sort_and_select_population(individuals=individuals,reference_points=ref_points, pop_size=self.pop_size)
        self.__optimize__(individuals=individuals,n_generations=n_generations,pop_start=pop_start+1, reference_points=ref_points)


    def __optimize__(self,individuals:individual_list,n_generations:int,pop_start:int, reference_points:np.ndarray):
        """ NSGA-III main loop
            Note: This function will read given starting population's results in, perform necessary crossover and mutation to generate enough individuals for the next iteration (self.pop_size)

            Inputs:
                individuals - list of individuals to evaluate
                n_generations - number of generations to loop through
                pop_start - starting population number
        """
        
        nIndividuals = len(individuals)
         
        # * Loop through all individuals
        for pop in range(pop_start,pop_start+n_generations):
            newIndividuals = self.__crossover_mutate__(individuals)

            # Evaluate
            self.evaluate_population(newIndividuals,pop_start)            
            newIndividuals = self.read_population(pop_start)
            # Sort and select
            pop_diversity = diversity(newIndividuals)       # Calculate diversity 
            pop_dist = distance(individuals,newIndividuals) # Calculate population distance between past and future
            newIndividuals.extend(individuals) # add the previous population to the pool                                    
            individuals,best_point, worst_point, extreme_points = sort_and_select_population(newIndividuals,reference_points, self.pop_size)            
            self.append_restart_file(individuals)        # Keep the last designs
            
            self.append_history_file(pop,individuals[0],pop_diversity,pop_dist)

            if self.single_folder_eval:
                # Delete the population folder
                population_folder = os.path.join(self.optimization_folder,self.__check_population_folder__(pop_start))
                if os.path.isdir(population_folder):
                    shutil.rmtree(population_folder)
            pop_start+=1 # increment the population
        # * End Loop through all individuals
    
    
    
    

    def __crossover_mutate__(self,individuals:List[Individual]):
        """[summary]

        Args:
            individuals (List[Individual]): [description]

        Returns:
            [type]: [description]
        """

        nIndividuals = len(individuals)
        num_params = len(individuals[0].eval_parameters)        
        if self.mutation_params.mutation_type == de_mutation_type.de_best_1_bin:
            newIndividuals = de_best_1_bin(individuals=individuals,objectives=self.objectives,
                eval_parameters=self.eval_parameters,performance_parameters=self.performance_parameters,
                F=self.mutation_params.F,C=self.mutation_params.C)
        elif self.mutation_params.mutation_type == de_mutation_type.de_rand_1_bin:
            newIndividuals = de_rand_1_bin(individuals=individuals,objectives=self.objectives,
                eval_parameters=self.eval_parameters,performance_parameters=self.performance_parameters,
                F=self.mutation_params.F,C=self.mutation_params.C)
        elif self.mutation_params.mutation_type == de_mutation_type.de_dmp:
            newIndividuals = de_dmp(individuals=individuals,
                objectives=self.objectives, eval_parameters=self.eval_parameters, performance_parameters=self.performance_parameters)
        else:               # self.mutation_params.mutation_type == de_mutation_type.simple
            nCrossover =  int(self.pop_size/2)
            nMutation = self.pop_size-nCrossover
            newIndividuals = simple(individuals=individuals,nCrossover=nCrossover,nMutation=nMutation,objectives=self.objectives,eval_parameters=self.eval_parameters,performance_parameters=self.performance_parameters,mu=self.mutation_params.mu,sigma=self.mutation_params.sigma)

        return newIndividuals
    
    
    
    

