"""
    adjoint - gradient based optimization 
"""
import os, shutil
import copy, random
from typing import Dict, List, Tuple
from scipy.optimize import minimize
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import LBFGS, AdamW
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from ..helpers import MultiLayerLinear 
from ..helpers import diversity, distance, mutation_parameters, de_mutation_type, simple,de_dmp, de_best_1_bin,de_rand_1_bin
from ..helpers import uniform_reference_points, sort_and_select_population
from ..helpers import evaluation_func, transform_data, compute_weights, objective_weighted_loss, compute_loss
from ..base import Parameter, Individual, Optimizer
from .nsga3 import find_intercepts, find_extreme_points


class NSGA3_ML(Optimizer):
    def __init__(self,eval_command:str = "python evaluation.py", eval_folder:str = "Evaluation", optimization_folder:str=None,single_folder_eval:bool=False, overwrite_input_file:bool=False, linear_network:List[int]=[64,64,64,64],epochs:int=100, train_test_split:float=0.8,pop_size:int=32,ml_evals:int=5):
        """The objective of adjoint is to find the minimum of the jacobian of the evaluation parameters.

        Args:
            eval_command (str, optional): [description]. Defaults to "python evaluation.py".
            eval_folder (str, optional): [description]. Defaults to "Evaluation".
            optimization_folder (str, optional): Folder where the optimization and doe work should be stored in. Defaults to None.
            single_folder_eval (bool, optional): Evaluate within a single folder and not make a bunch of folders . Defaults to False.
            overwrite_input_file (bool, optional): whether or not to overwrite the input file with new data when restarting a simulation. Defaults to False.
            linear_network (list, optional): Size of MultiLinear network. Defaults to [128,256,256].
            epochs (int, optional): Number of epochs to train neural network for 
            train_test_split (float, optional): Number of datapoints to be assigned to train and test
            pop_size (int, optional): Population size 
            ml_evals(int, optional): Number of internal machine learning evaluations to perform before evaluating the next population

        """
        super().__init__(name="NSGA3_ML",eval_command=eval_command,eval_folder=eval_folder, opt_folder=optimization_folder,single_folder_eval=single_folder_eval,overwrite_input_file=overwrite_input_file)
        
        self.individuals = None
        self.linear_network = linear_network
        self.epochs = epochs
        self.train_test_split = train_test_split
        self.pop_size = pop_size
        self.__mutation_params = mutation_parameters()
        
        self.models = list()
        self.optimizers = list()
        self.label_scalers = None
        self.feature_scalers = None
        self.labels_str = None
        self.features_str = None
        self.ml_evals = ml_evals
    
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

    def train(self,individuals:List[Individual],weights:List[float],retrain:bool=False) -> Tuple[float, float]:
        """Trains the neural network to predict the output given an input 
    
        Optimizer (LBFGS): 
            https://johaupt.github.io/python/pytorch/neural%20network/optimization/pytorch_lbfgs.html 

        Args:
            individuals (List[Individual]): All the individuals from best to worst
            weights (List[float]): List of weights for each individual
            retrain (bool, Optional): (True) retrains the existing model on new data. (False) create a new model

        Returns:
            Tuple[float float]: Train Loss and test loss 
        """
        weights = torch.tensor(weights,dtype=torch.float32)
        # * Normalizing the Data
        if retrain: # If we simply train it with more data then there's no reason create new normalization scalers
            normalized_individuals, label_scalers, feature_scalers, labels_str, features_str = transform_data(individuals,self.label_scalers,self.feature_scalers)
        else:
            normalized_individuals, label_scalers, feature_scalers, labels_str, features_str = transform_data(individuals)
        self.label_scalers = label_scalers
        self.feature_scalers = feature_scalers
        self.labels_str = labels_str
        self.features_str = features_str

        labels = torch.as_tensor(np.array([ind.objectives for ind in normalized_individuals]),dtype=torch.float32)
        features = torch.as_tensor(np.array([ind.eval_parameters for ind in normalized_individuals]),dtype=torch.float32)
        
        # Transform
        data = list(zip(features,labels))
        test_size = int(len(data)*(1-self.train_test_split))
        train_size = int(len(data) - test_size)

        train_indices, test_indices = train_test_split(list(range(len(data))),test_size=test_size,train_size=train_size,shuffle=True)
        train_dataset = [(data[i], weights[i]) for i in train_indices] # Tuple containing the weight
        test_dataset = [(data[i],weights[i]) for i in test_indices]
        
        train_dl = DataLoader(train_dataset,batch_size=128,shuffle=False)
        test_dl = DataLoader(test_dataset,batch_size=128,shuffle=False)
        # * Defining the Model        
        n_inputs = features.shape[1]
        n_outputs = labels.shape[1]
        criterions = list()
        
        if len(self.models) == 0 or retrain == False:
            self.models.clear(); self.optimizers.clear()
            for l in self.label_scalers: # For each label we make a new model 
                self.models.append(MultiLayerLinear(n_inputs,1,h_sizes=self.linear_network))
                # self.optimizer = LBFGS(self.model.parameters(), lr=0.0001,history_size=100, max_eval=int(20*1.25), max_iter=20)
                self.optimizers.append(AdamW(self.models[-1].parameters()))
                criterions.append(objective_weighted_loss())
        n = len(self.models)
        for i in range(n): # Train a model for each objective
            print(f"training model {i+1} out of {n}")
            for _ in range(self.epochs):
                train_loss = 0 
                n_train = 0
                self.models[i].train()
                for i, d in enumerate(train_dl):
                    x,y = d[0]
                    w = d[1]
                    batch_size = x.shape[0]
                    self.optimizers[i].zero_grad()
                    y_pred = self.models[i](x)
                    loss = criterions[i](y_pred, y[:,i], w)
                    loss.backward() # Zero gradients, backward pass, and update weights
                    self.optimizers[i].step()
                    # calculate the loss again for monitoring
                    train_loss += loss.item()
                    n_train += batch_size
                
                test_loss = 0
                n_test = 0
                self.models[i].eval()
                for j, d in enumerate(test_dl):
                    x,y = d[0]
                    w = d[1]
                    batch_size = x.shape[0]
                    y_pred = self.models[i](x)
                    test_loss += criterions[i](y_pred, y[:,i], w).item()
                    n_test += batch_size
                train_loss /= n_train
                test_loss /= n_test
        return train_loss, test_loss
            # print(f"Epoch: {epoch + 1:02}/{self.epochs} Train Loss: {train_loss:.5e} Test Loss: {test_loss:.5e}")
        
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

        newIndividuals = self.read_calculation_folder() # Use all individuals, there's no restart file
        newIndividuals = [item for sublist in newIndividuals for item in sublist]  # Flattens the list of lists
        # bounds = [(ep.min_value, ep.max_value) for ep in self.eval_parameters]

        if (len(newIndividuals)<self.pop_size):
            raise Exception("Number of individuals in the restart file is less than the population size."
                + " lower the population size or increase the DOE count(if restarting from a DOE)")

        # Do this before going into the train loop. This part of the code should happen after a new population is evaluated 
        ref_points = uniform_reference_points(len(self.objectives), p=4, scaling=None)
        individuals,_, _, _, pareto_groups = sort_and_select_population(individuals=newIndividuals,reference_points=ref_points, pop_size=self.pop_size)
        weights = compute_weights(pareto_groups)
        all_individuals = list() 
        all_individuals.extend(copy.deepcopy(individuals))
        loss_fn = objective_weighted_loss()
        for pop in range(pop_start+1,pop_start+n_generations):  # Population Loop 
            if self.ml_evals == 0:
                newIndividuals = self.__crossover_mutate__(individuals) # This becomes normal nsga3            
            ''' 
                Train a Neural network on Individuals. Initially this is all the individuals
            '''
            train_loss, test_loss = self.train(copy.deepcopy(all_individuals),weights, False)
            
            '''
                Calculate new evaluation points using neural networks
            '''
            pop_dist_ml = list()
            pop_diversity_ml = list() 
            for _ in range(self.ml_evals): # Perform nsga evaluations on the ML Model 
                newIndividuals = self.__crossover_mutate__(individuals)
                newIndividuals = evaluation_func(newIndividuals,self.models,self.label_scalers,self.feature_scalers)                
                pop_diversity_ml.append(diversity(newIndividuals))       # Calculate diversity 
                pop_dist_ml.append(distance(individuals,newIndividuals)) # Calculate population distance between past and future

                newIndividuals.extend(individuals) # add the previous population to the pool                                    

                individuals,_, _, _,_ = sort_and_select_population(newIndividuals,ref_points, self.pop_size)

            # Evaluate
            self.evaluate_population(individuals,pop)
            newIndividuals = self.read_population(pop)
            _,_, _, _, pareto_groups = sort_and_select_population(newIndividuals,ref_points, self.pop_size)    # reduces the size of newIndividuals to the population size    
            weights = compute_weights(pareto_groups)
            loss = compute_loss(individuals,newIndividuals, weights)    # Compare how ML is predicting the latest pareto front 
            print(f"Pareto Weighted Loss {loss:03e}")                   # This should improve as pareto front gets more well defined

            # Sort and select after extending the list of individuals with the old list
            pop_diversity = diversity(newIndividuals)       # Calculate diversity 
            pop_dist = distance(individuals,newIndividuals) # Calculate population distance between past and future
            all_individuals.extend(copy.deepcopy(newIndividuals))
            individuals,_, _, _, pareto_groups = sort_and_select_population(all_individuals,ref_points, len(all_individuals))    # reduces the size of newIndividuals to the population size                
            weights = compute_weights(pareto_groups)

            self.append_restart_file(individuals)        # Keep the last best designs
            self.append_history_file(pop,individuals[0],pop_diversity,pop_dist,train_loss,test_loss,loss) # Save the best one, for multi-objective this could be meaningless

            if self.single_folder_eval:
                # Delete the population folder
                population_folder = os.path.join(self.optimization_folder,self.__check_population_folder__(pop))
                if os.path.isdir(population_folder):
                    shutil.rmtree(population_folder)
  

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
    