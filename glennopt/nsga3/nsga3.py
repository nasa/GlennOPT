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
from itertools import chain
from tqdm import trange

from ..helpers import diversity, distance
from ..helpers import non_dominated_sorting
from ..base import Parameter, Individual, Optimizer
from ..helpers import de_best_1_bin,de_rand_1_bin, mutation_parameters, de_mutation_type, simple,de_rand_1_bin_spawn,de_dmp, get_eval_param_matrix, get_objective_matrix, set_eval_parameters

individual_list = List[Individual]

class NSGA3(Optimizer):
    def __init__(self,eval_script:str = "evaluation.py", eval_folder:str = "Evaluation",pop_size:int=128, optimization_folder:str=None,single_folder_eval=False):
        super().__init__(name="nsga3",eval_script=eval_script,eval_folder=eval_folder, opt_folder=optimization_folder,single_folder_eval=single_folder_eval)
        """
            NSGA-3 multi-dimensional optimizer. This version has been tweaked to include restart capabilities. It can also keep track of additional parameters that can be considered part of the constraints.

            Each evaluation can occur in a separate folder (simulations) or without folders (analytical)
            https://www.egr.msu.edu/~kdeb/papers/k2012009.pdf

            Inputs:
                eval_script - Evaluation python script that will be called. Either Output.txt is read or an actual output is read. 
                eval_folder - folder to be copied into each individual evaluation directory. If this is null, the population directory isn't created and neither are the individual directories
                num_populations - number of populations to evaluate from the starting population
                pop_size - number of individuals in a given population
                optimization_folder - where optimization should start
        """
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
    
    def start_doe(self,doe_individuals:List[Individual]=None,doe_size:int=128):
        """
            Starts a design of experiments. If the DOE has already started and there is an output file for an individual then the individual won't be evaluated             
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
        """
            Reads the values of a population, this can be a DOE or a previous evaluation
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
        ref_points = self.uniform_reference_points(len(self.objectives), p=4, scaling=None)
        individuals,best_point, worst_point, extreme_points = self.sort_and_select_population(individuals=individuals,reference_points=ref_points)
        self.__optimize__(individuals=individuals,n_generations=n_generations,pop_start=pop_start+1, reference_points=ref_points)


    def __optimize__(self,individuals:individual_list,n_generations:int,pop_start:int, reference_points:np.ndarray):
        """ 
            NSGA-III main loop
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
            individuals,best_point, worst_point, extreme_points = self.sort_and_select_population(newIndividuals,reference_points)            
            self.append_restart_file(individuals)        # Keep the last designs

            
            self.append_history_file(pop,individuals[0],pop_diversity,pop_dist)

            if self.single_folder_eval:
                # Delete the population folder
                population_folder = os.path.join(self.optimization_folder,self.__check_population_folder__(pop_start))
                if os.path.isdir(population_folder):
                    shutil.rmtree(population_folder)
            pop_start+=1 # increment the population
        # * End Loop through all individuals
    
    def sort_and_select_population(self,individuals:individual_list, reference_points:np.ndarray):
        '''
            Takes a list of individuals, finds the fronts and the best designs
            Inputs: 
                individuals:         
            
            Code is a combination from deap and yarpiz
            https://github.com/DEAP
            https://yarpiz.com/456/ypea126-nsga3
        '''
        if (self.pop_size>len(individuals)):
            raise Exception("population size needs to be <= the number of individuals")
        

        pareto_fronts = non_dominated_sorting(individuals,self.pop_size)
        fitnesses = np.array([ind.objectives for f in pareto_fronts for ind in f])
        fitnesses *= -1

        best_point = np.min(fitnesses,axis=0)
        worst_point = np.max(fitnesses,axis=0)

        extreme_points = self.__find_extreme_points__(fitnesses,best_point)
        front_worst = np.max(fitnesses[:sum(len(f) for f in pareto_fronts),:],axis=0)
        intercepts = self.__find_intercepts__(extreme_points,best_point,worst_point,front_worst)
        niches, dist = self.__associate_to_niche__(fitnesses, reference_points, best_point, intercepts)
        
        # Get counts per niche for individuals in all front but the last
        niche_counts = np.zeros(len(reference_points), dtype=int)
        index, counts = np.unique(niches[:-len(pareto_fronts[-1])], return_counts=True)
        niche_counts[index] = counts

        # Choose individuals from all fronts but the last
        chosen = list(chain(*pareto_fronts[:-1]))

        # Use niching to select the remaining individuals
        sel_count = len(chosen)
        n = self.pop_size - sel_count
        selected = self.__niching__(pareto_fronts[-1], n, niches[sel_count:], dist[sel_count:], niche_counts)
        chosen.extend(selected)

        return chosen, best_point, worst_point, extreme_points

    
    def __find_extreme_points__(self,fitnesses, best_point, extreme_points=None):
        'Finds the individuals with extreme values for each objective function.'
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

    def __find_intercepts__(self,extreme_points, best_point, current_worst, front_worst):
        """Find intercepts between the hyperplane and each axis with
        the ideal point as origin."""
        # Construct hyperplane sum(f_i^n) = 1
        b = np.ones(extreme_points.shape[1])
        A = extreme_points - best_point
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            intercepts = current_worst
        else:
            intercepts = 1 / x

            if (not np.allclose(np.dot(A, x), b) or
                    np.any(intercepts <= 1e-6) or
                    np.any((intercepts + best_point) > current_worst)):
                intercepts = front_worst

        return intercepts

    def __associate_to_niche__(self,fitnesses, reference_points, best_point, intercepts):
        """Associates individuals to reference points and calculates niche number.
        Corresponds to Algorithm 3 of Deb & Jain (2014)."""
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

    def __niching__(self,individuals, k, niches, distances, niche_counts):
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

    def uniform_reference_points(self,nobj, p=4, scaling=None):
        """
            Generate reference points uniformly on the hyperplane intersecting
            each axis at 1. The scaling factor is used to combine multiple layers of
            reference points.
            
            Inputs:
                nobj - number of objectives
                p = number of references per objective
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
    
    
    
    

