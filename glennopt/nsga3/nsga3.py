"""
    NSGA3 - Non-dominated sorting genetic algorithm
"""
import os, shutil
import subprocess, copy, math
from random import seed, gauss, random,uniform
from typing import List
from dataclasses import dataclass, field
import numpy as np 
import glob

from ..base_classes import Optimizer
from ..helpers import Parameter
from . import NSGA_Individual
from . import non_dominated_sorting
from . import associate_to_reference_point
from . import generate_reference_points
from . import mutation_simple, mutation_de_1_rand_bin, mutation_de_best_2_bin, de_mutation_type, mutation_parameters

individual_list = List[NSGA_Individual]

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
    
    def start_doe(self,doe_size:int=128):
        """
            Starts a design of experiments. If the DOE has already started and there is an output file for an individual then the individual won't be evaluated             
        """
        doe_individuals = []
        for i in range(doe_size):
            parameters = copy.deepcopy(self.eval_parameters)
            for eval_param in parameters:
                eval_param.value = np.random.uniform(eval_param.min_value,eval_param.max_value,1)[0]
            
            doe_individuals.append(NSGA_Individual(eval_parameters=parameters,objectives=self.objectives, performance_parameters = self.performance_parameters))
        
        # * Begin the evaluation
        self.evaluate_population(individuals=doe_individuals,population_number=-1)
        # * Read the DOE
        individuals = self.read_population(population_number=-1)
        params = {}
        params['nPop'] = self.pop_size
        params['Zr'] = generate_reference_points(len(self.objectives),4)
        _, params['nZr'] = params['Zr'].shape
        params['zmin'] = []
        params['zmax'] = []
        params['smin'] = []
        [individuals,_,params] = self.sort_and_select_population(individuals=individuals,params=params)
        self.append_restart_file(individuals) # Create the restart file
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
        
        individuals = self.read_restart_file()
        
        if (len(individuals)==0):
            individuals = self.read_population(population_number=pop_start)            

        # Normalize the Population
        params = {}
        params['nPop'] = self.pop_size
        params['Zr'] = generate_reference_points(len(self.objectives),4)
        _, params['nZr'] = params['Zr'].shape
        params['zmin'] = []
        params['zmax'] = []
        params['smin'] = []

        # Crossover and Mutate the doe individuals to generate the next individuals used in the population
        # Sort the population into [fill in here]
        [individuals,F,params] = self.sort_and_select_population(individuals=individuals,params=params)
        self.__optimize__(individuals=individuals,n_generations=n_generations,pop_start=pop_start+1,params=params,F=F)


    def __optimize__(self,individuals:individual_list,n_generations:int,pop_start:int,params:dict,F:list):
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
            new_pop = []
            if self.mutation_params.mutation_type == de_mutation_type.de_1_rand_bin:
                max_parents = self.pop_size*self.mutation_params.max_parents
                min_parents = self.pop_size*self.mutation_params.min_parents
                min_parents = 4 if min_parents<4 else min_parents
                nParents = math.floor(np.random.rand(1)*max_parents+min_parents) 
                nParents = 3
                newIndividuals = mutation_de_1_rand_bin(individuals,nParents)
            elif self.mutation_params.mutation_type == de_mutation_type.de_best_2_bin:
                if len(F[len(F)-1]) == 0: # Take the last front. Sometimes the last array of F is []
                    temp = F[len(F)-2]
                else:
                    temp = F[len(F)-1]                    
                best_indx = temp[np.random.randint(0,len(temp))] # best index is a random individual taken from best front
                newIndividuals = mutation_de_best_2_bin(best_indx,individuals)
            else:
                # use simple mutation
                nCrossover = int(nIndividuals*0.5)
                nMutation = nIndividuals-nCrossover
                newIndividuals = mutation_simple(individuals,nCrossover,nMutation)
            # concatenante lists
            new_pop.extend(newIndividuals)

            self.evaluate_population(new_pop,pop_start)
            
            newIndividuals = self.read_population(pop_start)
            # Sort and select
            newIndividuals.extend(individuals) # add the previous population to the pool                        
            [individuals,F,params] = self.sort_and_select_population(newIndividuals,params)
            self.append_restart_file(newIndividuals)
            if self.single_folder_eval:
                # Delete the population folder
                population_folder = os.path.join(self.optimization_folder,self.__check_population_folder__(pop_start))
                if os.path.isdir(population_folder):
                    shutil.rmtree(population_folder)
            pop_start+=1 # increment the population
        # * End Loop through all individuals
    
    def sort_and_select_population(self,individuals:individual_list,params:dict):
        """
            Takes a list of individuals, finds the fronts and the best designs                        
        """
        individuals,params = self.__normalize_population__(individuals,params)
        
        [individuals, F] = non_dominated_sorting(individuals)
                
        [pop, d, rho] = associate_to_reference_point(individuals, params['Zr'])
        LastFront = []
        newpop = []
        for l in range(len(F)):
            if len(newpop) + len(F[l]) >= self.pop_size:
                LastFront = F[l]
                break
            individuals_slice =[individuals[i] for i in F[l]]           
            newpop.extend(individuals_slice)
            # newpop = newpop.extend(individuals[F[l]])
            # newpop = [newpop; pop(F{l})];
        
        while True:
            # --- Begin While Loop --- 
            j = np.argmin(rho)
            
            AssocitedFromLastFront = []
            for i in LastFront:
                if individuals[i].association_ref == j:
                    AssocitedFromLastFront.append(i)
                
            if (len(AssocitedFromLastFront)==0):
                rho[j] = float("inf") # * Paht: this is the count/density of individuals closest to the pareto front
                continue # Returns to the start of the while loop
            
            if rho[j] == 0:
                ddj = d[AssocitedFromLastFront, j]
                new_member_ind = np.argmin(ddj)
            else:
                if (len(AssocitedFromLastFront)==1):
                    new_member_ind = 0
                else:
                    new_member_ind = np.random.randint(0,len(AssocitedFromLastFront)-1)
                        
            MemberToAdd = AssocitedFromLastFront[new_member_ind]
            
            del LastFront[LastFront == MemberToAdd]
            
            newpop.append(individuals[MemberToAdd])
            
            rho[j] += 1
            
            if (len(newpop) >= self.pop_size):
                break
            # --- End While Loop --- 
        [individuals, F] = non_dominated_sorting(newpop)
        return individuals,F,params

    # * The follow codes are a part of sort and select
    def __normalize_population__(self, individuals:List[Parameter],params:dict):
        zmin = self.__update_ideal_point__(individuals,params)
        objectives_mat = np.zeros((len(self.objectives), len(individuals)))

        for i in range(len(self.objectives)):
            for j in range(len(individuals)):
                objectives_mat[i,j] = individuals[j].objectives[i]
        
        fp = objectives_mat - (np.zeros((len(self.objectives), len(individuals))) + zmin.reshape((len(self.objectives),1)))
        params = self.__perform_scalarizing__(fp,params,len(individuals))
        a = self.__find_hyper_plane_intercepts__(params['zmax']) # Finds the maximum objective value

        for i in range(len(individuals)):
            individuals[i].normalized_cost = np.zeros(len(self.objectives))
            for j in range(len(self.objectives)):
                individuals[i].normalized_cost[j] = fp[j,i]/a[j] # Normalize everything by the maximum objective value
        
        return individuals,params

    def __perform_scalarizing__(self,z:np.ndarray,params:dict,n_individuals:int):
        """
            Scales the objectives 
            Inputs:
                z - objectives x number of individuals
                params - dictionary with z (objectives) s is the factor to make them all normalized 1/objective
                n_individuals - number of individuals
        """
        def get_scalarized_vector(nObj,j):
            w = np.zeros(nObj)
            w[j] = 1
            return w

        nObj = len(self.objectives)
        
        if (len(params['smin'])>0):
            zmax = params['zmax']
            smin = params['smin']
        else:
            zmax = np.zeros((nObj,nObj))
            smin = np.zeros(nObj) + float("inf")
        
        for i in range(nObj):
            w = get_scalarized_vector(nObj,i)

            s = np.zeros(n_individuals)
            for j in range(n_individuals):
                s[j] = max(np.nan_to_num(np.divide(z[:,j],w),'inf'))  # s represents a matrix or vector of the max value of the objectives for reach individual. Objective 1 and Objective 2 -> Take the max so it can be either objective. objectives x individuals
            
            sminj = min(s)
            indx = np.argmin(s)

            if (sminj < smin[i]): 
                zmax[:,i] = z[:,indx]
                smin[i] = sminj
        
        params['zmax'] = zmax
        params['smin'] = smin

        return params

    def __update_ideal_point__(self, individuals:List[Parameter],params:dict):
        """
            Finds the minimum value of each objective
        """
        if (len(params["zmin"])==0):
            prev_zmin = np.zeros(len(self.objectives)) + float("inf")            
        zmin = prev_zmin
        for i in range(len(individuals)):
            zmin = np.minimum(zmin, individuals[i].objectives) # 
        return zmin

    def __find_hyper_plane_intercepts__(self,zmax:np.ndarray):
        """
            Find Hyper plane intercepts 
        """
        nrows_zmax,ncols_zmax = zmax.shape
        if (nrows_zmax*ncols_zmax==1): # Single objective
            a = np.ones((1,ncols_zmax))
        else:                
            try:
                w = np.divide(np.ones((1,ncols_zmax)),np.linalg.inv(zmax)) # try this is /z doesnt work
            except:
                pass
            finally: # the above statement can fail if zmax is singular (no inverse)
                w = np.ones((1,ncols_zmax))
            a = np.transpose(1.0/w) # 1 divided by each element
            # * --- sort and select ---
        return a
    
    
    
    

