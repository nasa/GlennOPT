"""
    adjoint - gradient based optimization 
"""
import os, shutil
import copy
from typing import List
from scipy.optimize import minimize
import numpy as np
from sklearn import preprocessing
from tqdm import trange

import torch
import torch.nn as nn
from torch.optim import LBFGS
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..helpers import MultiLayerLinear 
from ..helpers import diversity, distance
from ..helpers import non_dominated_sorting, find_extreme_points, find_intercepts, associate_to_niche, niching, uniform_reference_points 
from ..base import Parameter, Individual, Optimizer
from .nsga3 import find_intercepts, find_extreme_points
individual_list = List[Individual]

class Adjoint(Optimizer):
    def __init__(self,eval_command:str = "python evaluation.py", eval_folder:str = "Evaluation",pop_size:int=128, optimization_folder:str=None,single_folder_eval:bool=False, overwrite_input_file:bool=False, linear_network:List[int]=[64,128,128,64],epochs:int=20, train_test_split:float=0.7):
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

    def hessian(x:np.ndarray):
        """
            Calculate the hessian matrix with finite differences. This assumes the delta h is constant
            Example taken from https://stackoverflow.com/a/31207520/1599606 

        Args:
            x (np.ndarray): Numpy array representing the function values (f) evaluated at constant h 

        Returns:
            np.ndarray: Returns an array of shape (x.dim, x.ndim) + x.shape where the array[i, j, ...] corresponds to the second derivative x_ij
        """
        x_grad = np.gradient(x) 
        hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
        for k, grad_k in enumerate(x_grad):
            # iterate over dimensions
            # apply gradient again to every component of the first derivative.
            tmp_grad = np.gradient(grad_k) 
            for l, grad_kl in enumerate(tmp_grad):
                hessian[k, l, :, :] = grad_kl
        return hessian

    def train(self,individuals:List[Individual]):
        """Trains the neural network to predict the output given an input 
    
        Optimizer (LBFGS): 
            https://johaupt.github.io/python/pytorch/neural%20network/optimization/pytorch_lbfgs.html 

        Args:
            individuals (List[Individual]): [description]

        Returns:
            [type]: [description]
        """
        # * Handling the Data
        pareto_fronts = non_dominated_sorting(individuals,len(individuals))

        # non_dominated_sorting
        labels = torch.as_tensor(np.array([ind.objectives for ind in individuals]),dtype=torch.float32)
        features = torch.as_tensor(np.array([ind.eval_parameters for ind in individuals]),dtype=torch.float32)

        # Normalization 
        scaler = preprocessing.StandardScaler()
        scaler.fit(labels)
        scaler.fit(features)

        data = list(zip(features,labels))
        test_size = int(len(data)*(1-self.train_test_split))
        train_size = int(len(data) - test_size)
        train_dataset, test_dataset = train_test_split(data,test_size=test_size,train_size=train_size,shuffle=True)

        # * Defining the Model        
        n_inputs = features.shape[1]
        n_outputs = labels.shape[1]
        
        model = MultiLayerLinear(n_inputs,n_outputs,h_sizes=self.linear_network)
        optimizer = LBFGS(model.parameters(), lr=0.2,history_size=10, max_iter=5)
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
                loss = closure()
                train_running_loss += loss.item()

            model.eval()
            test_loss = 0
            for j, (x,y) in enumerate(test_dataset):
                y_pred = model(x)
                test_loss += criterion(y_pred, y).item()
            
            print(f"Epoch: {epoch + 1:02}/{self.epochs} Train Loss: {train_running_loss:.5e} Test Loss: {test_loss:.5e}")
        return model
    
    
    # def __objective_function__(self,model:nn.Module,reference_point:float,output_index:int):

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

        model = self.train(individuals)
        
        # ! This will have to change Loop through all individuals 
        # for pop in range(pop_start,pop_start+n_generations):


        #     newIndividuals = self.__crossover_mutate__(individuals)

        #     # Evaluate
        #     self.evaluate_population(newIndividuals,pop_start)            
        #     newIndividuals = self.read_population(pop_start)
        #     # Sort and select
        #     pop_diversity = diversity(newIndividuals)       # Calculate diversity 
        #     pop_dist = distance(individuals,newIndividuals) # Calculate population distance between past and future
        #     newIndividuals.extend(individuals) # add the previous population to the pool                       



    def __adoint_objective_fun__(self,x0:np.ndarray,model:nn.Module,reference_points:List[np.ndarray],intercepts:np.ndarray,dist_index:int):
        """Objective function of adjoint using neural networks

        Args:
            x0 (np.ndarray): [description]
            model (nn.Module): [description]
            reference_points (List[np.ndarray]): [description]
            intercepts (np.ndarray)
        """
        fitnesses = -1*x0

        best_point = np.min(fitnesses,axis=0)
        worst_point = np.max(fitnesses,axis=0)
        _, dist = associate_to_niche(fitnesses, reference_points, best_point, intercepts)

        return dist[dist_index]


        

        

        # individuals,best_point, worst_point, extreme_points = self.sort_and_select_population(newIndividuals,reference_points)            
        # self.append_restart_file(individuals)        # Keep the last designs

        
        # self.append_history_file(pop,individuals[0],pop_diversity,pop_dist)

        # if self.single_folder_eval:
        #     # Delete the population folder
        #     population_folder = os.path.join(self.optimization_folder,self.__check_population_folder__(pop_start))
        #     if os.path.isdir(population_folder):
        #         shutil.rmtree(population_folder)
        # pop_start+=1 # increment the population
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