"""
    adjoint - gradient based optimization 
"""
import os, shutil
import subprocess, copy, math
from random import seed, gauss,random,uniform,randint
from typing import List
from dataclasses import dataclass, field
import numpy as np 

import torch
import torch.nn as nn
from torch.optim import LBFGS
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from ..helpers import MultiLayerLinear 
from ..helpers import diversity, distance
from ..helpers import non_dominated_sorting
from ..base import Parameter, Individual, Optimizer
from ..helpers import jacobian

individual_list = List[Individual]

class Adjoint(Optimizer):
    def __init__(self,eval_command:str = "python evaluation.py", eval_folder:str = "Evaluation",pop_size:int=128, optimization_folder:str=None,single_folder_eval:bool=False, overwrite_input_file:bool=False, linear_network:List[int]=[64,128,128,64],epochs:int=10, train_test_split:float=0.7):
        """The objective of adjoint is to find the minimum of the jacobian of the evaluation parameters.

        Args:
            eval_command (str, optional): [description]. Defaults to "python evaluation.py".
            eval_folder (str, optional): [description]. Defaults to "Evaluation".
            pop_size (int, optional): [description]. Defaults to 128.
            optimization_folder (str, optional): Folder where the optimization and doe work should be stored in. Defaults to None.
            single_folder_eval (bool, optional): Evaluate within a single folder and not make a bunch of folders . Defaults to False.
            overwrite_input_file (bool, optional): whether or not to overwrite the input file with new data when restarting a simulation. Defaults to False.
            linear_network (list, optional): Size of MultiLinear network. Defaults to [128,256,256].
            epochs (int, optional): Number of epochs to train neural network for 
        """
        super().__init__(name="adjoint",eval_command=eval_command,eval_folder=eval_folder, opt_folder=optimization_folder,single_folder_eval=single_folder_eval,overwrite_input_file=overwrite_input_file)
        
        self.pop_size = pop_size
        self.individuals = None
        self.linear_network = linear_network
        self.epochs = epochs
        self.train_test_split = train_test_split

    def train(self):
        """Trains the neural network to predict the output given an input 

        Uses:
            Optimizer (LBFGS): https://johaupt.github.io/python/pytorch/neural%20network/optimization/pytorch_lbfgs.html 
        """
        # * Handling the Data
        # individuals = self.read_calculation_folder()        
        individuals = self.read_restart_file()
        pareto_fronts = non_dominated_sorting(individuals,len(individuals))

        non_dominated_sorting
        y = torch.as_tensor(np.array([ind.objectives for ind in individuals]),dtype=torch.float32)
        x = torch.as_tensor(np.array([ind.eval_parameters for ind in individuals]),dtype=torch.float32)
        x.requires_grad=True

        data = list(zip(x,y))
        test_size = int(len(data)*(1-self.train_test_split))
        train_size = int(len(data) - test_size)
        train_dataset, test_dataset = train_test_split(data,test_size=test_size,train_size=train_size,shuffle=True)

        # * Defining the Model        
        n_inputs = x.shape[1]
        n_outputs = y.shape[1]
        model = MultiLayerLinear(n_inputs,n_outputs,h_sizes=self.linear_network)

        optimizer = LBFGS(model.parameters(), history_size=10, max_iter=5)
 
        criterion = nn.MSELoss()
           
        for epoch in range(self.epochs):
            train_running_loss = 0 
            model.train()
            for i, (x, y) in enumerate(train_dataset):
                def closure():
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                    if loss.requires_grad:
                        loss.backward() # Zero gradients, backward pass, and update weights
                    return loss
                optimizer.step(closure)
                # calculate the loss again for monitoring
                output = model(x)
                loss = closure()
                train_running_loss += loss.item()

            model.eval()
            test_loss = 0
            for j, (x,y) in enumerate(test_dataset):
                y_pred = model(x)
                test_loss += criterion(y_pred, y).item()
            
            print(f"Epoch: {epoch + 1:02}/{self.epochs} Train Loss: {train_running_loss:.5e} Test Loss: {test_loss:.5e}")
        self.model = model

        # Evaluate Jacobian for all individuals along the pareto-front
        model.eval()
        jacobians = list()
        for ind in pareto_fronts[0]: # Take best individuals
            x = torch.ones((1,n_inputs),dtype=torch.float32)    # This represents a gradient with itself https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/009cea8b0f40dfcb55e3280f73b06cc2/autograd_tutorial.ipynb#scrollTo=tjaTadQPgI-W
            jacobians.append(torch.autograd.functional.jacobian(func=model,inputs=x, create_graph=True))

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
        ref_points = self.uniform_reference_points(len(self.objectives), p=4, scaling=None)
        individuals,best_point, worst_point, extreme_points = self.sort_and_select_population(individuals=individuals,reference_points=ref_points)
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