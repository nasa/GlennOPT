"""
    adjoint - gradient based optimization 
"""
import os, shutil
import copy, random
from typing import List, Tuple
from scipy.optimize import minimize, Bounds
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..helpers import MultiLayerLinear 
from ..helpers import diversity, distance, set_eval_parameters
from ..helpers import non_dominated_sorting, find_intercepts, uniform_reference_points, sort_and_select_population
from ..helpers import compute_mse, transform_data
from ..base import Parameter, Individual, Optimizer
from .nsga3 import find_intercepts
individual_list = List[Individual]

@torch.no_grad()
def surrogate_objective_func(x0:np.ndarray,model:nn.Module,reference_points:List[np.ndarray],dist_index:int, labels_scaler:List[MinMaxScaler], features_scaler:List[MinMaxScaler]):
    """Objective function of adjoint using neural networks. The goal is to use this function to find values of x0 that minimize the distance to the reference point

    Args:
        x0 (np.ndarray): initial guess
        model (nn.Module): neural network model used for prediction
        reference_points (List[np.ndarray]): array of reference points along the pareto front
        intercepts (np.ndarray)

    Returns:
        (float): sum of all the fitnesses. This is the value to be minimized
    """
    x0_2 = copy.deepcopy(x0)
    for l in range(len(features_scaler)):
        x0_2[l] = features_scaler[l].transform(x0_2[l].reshape(-1,1))

    x0_2 = torch.as_tensor(x0_2,dtype=torch.float32)
    fitnesses = model(x0_2)
    fitnesses = fitnesses.detach().numpy()    
    for f in range(len(labels_scaler)):        
        fitnesses[f] = labels_scaler[f].inverse_transform(fitnesses[f].reshape(-1,1)*reference_points[dist_index][f])

    return np.sum(fitnesses[f])


class NSOPT(Optimizer):
    def __init__(self,eval_command:str = "python evaluation.py", eval_folder:str = "Evaluation", optimization_folder:str=None,single_folder_eval:bool=False, overwrite_input_file:bool=False, linear_network:List[int]=[64,64,64,64],epochs:int=200, train_test_split:float=0.8,pareto_resolution:int=32, min_method:str='Powell'):
        """NSOPT - Non-dominated Sorting Machine Learning Optimization Strategy for multi-objective applications.
            This strategy uses machine learning to build a surrogate model and scipy.optimizer.minimize to search for the values of $\vec{x}$ minimizes all objectives. The results will be a pareto front or surface or whatever that is in higher dimensions. 

        Args:
            eval_command (str, optional): This is the evaluation command that will be called inside the Individual's folder. Defaults to "python evaluation.py".
            eval_folder (str, optional): Folder that contains the executable files that are copied into each separate evaluation. Defaults to "Evaluation".
            optimization_folder (str, optional): Folder where the optimization and doe work should be stored in. Defaults to None.
            single_folder_eval (bool, optional): Evaluate within a single folder and not make a bunch of folders . Defaults to False.
            overwrite_input_file (bool, optional): whether or not to overwrite the input file with new data when restarting a simulation. Defaults to False.
            linear_network (list, optional): Size of MultiLinear network. Defaults to [64,64,64,64].
            epochs (int, optional): Number of epochs to train neural network for. Defaults to 200
            train_test_split (float, optional): Number of datapoints to be assigned to train and test. Defaults to 0.8.
            pareto_resolution (int, optional): indicates how many points to use to search for the pareto. This is also the population size. 
            min_method (str, optional): minimization method from https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html. Defaults to 'Powell'

        """
        super().__init__(name="NSOPT",eval_command=eval_command,eval_folder=eval_folder, opt_folder=optimization_folder,single_folder_eval=single_folder_eval,overwrite_input_file=overwrite_input_file)
        
        self.individuals = None
        self.linear_network = linear_network
        self.epochs = epochs
        self.train_test_split = train_test_split
        self.pareto_resolution = pareto_resolution
        self.model = None
        self.optimizer = None
        self.min_method = min_method

        self.label_scalers = None
        self.feature_scalers = None
        self.labels_str = None
        self.features_str = None
        self.n_inputs = None
        self.n_outputs = None


    def train(self,individuals:List[Individual],retrain:bool=False):
        """Trains the neural network to predict the output given an input 

        Args:
            individuals (List[Individual]): List of individuals to train the neural network
            retrain (bool, Optional): (True) retrains the existing model on new data. (False) create a new model for every population. Setting it to false means training takes longer but the model is more accurate because it renormalizes the inputs and outputs. 

        Returns:
            Tuple[float float]: Train Loss and test loss 
        """
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
        train_dataset, test_dataset = train_test_split(data,test_size=test_size,train_size=train_size,shuffle=True)
        train_dl = DataLoader(train_dataset,batch_size=128,shuffle=True)
        test_dl = DataLoader(test_dataset,batch_size=128,shuffle=False)
        # * Defining the Model        
        n_inputs = features.shape[1]
        n_outputs = labels.shape[1]

        if self.model is None or retrain == False:
            self.model = MultiLayerLinear(n_inputs,n_outputs,h_sizes=self.linear_network)
            # self.optimizer = LBFGS(self.model.parameters(), lr=0.0001,history_size=100, max_eval=int(20*1.25), max_iter=20)
            self.optimizer = AdamW(self.model.parameters())
            self.n_inputs = n_inputs
            self.n_outputs = n_outputs 

        criterion = nn.MSELoss()
        
        for epoch in range(self.epochs):
            train_running_loss = 0 
            n_train = 0
            self.model.train()
            for i, (x, y) in enumerate(train_dl):
                batch_size = x.shape[0]
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = criterion(y_pred, y)
                loss.backward() # Zero gradients, backward pass, and update weights
                self.optimizer.step()
                # calculate the loss again for monitoring
                train_running_loss += loss.item()
                n_train += batch_size
            
            test_loss = 0
            n_test = 0
            self.model.eval()
            for j, (x,y) in enumerate(test_dl):
                batch_size = x.shape[0]
                y_pred = self.model(x)
                test_loss += criterion(y_pred, y).item()
                n_test += batch_size
            train_running_loss /= n_train
            test_loss /= n_test
        return train_running_loss, test_loss

    def optimize_from_population(self,pop_start:int,n_generations:int):
        """Starts the optimization by reading the values of a population. This can be a DOE or a previous evaluation 

        Args:
            pop_start (int): pop_start=-1 for DOE. Reads the population folder and starts at pop_start+1
            n_generations (int): Number of generations to iterate for 

        Raises:
            Exception: [description]
        """
        # * Read in all the results of the DOE, this should be done by a single thread
        # Check restart file, if not read the population
        self.load_history_file()
        ref_points = uniform_reference_points(len(self.objectives), p=self.pareto_resolution, scaling=None) # the -1 is there so that the number of reference points matches the number of individuals

        all_individuals = list() 

        newIndividuals = self.read_calculation_folder() # Use all individuals, there's no restart file
        newIndividuals = [item for sublist in newIndividuals for item in sublist]  # Flattens the list of lists
        lb = [ep.min_value for ep in self.eval_parameters]
        ub = [ep.max_value for ep in self.eval_parameters]
        bounds = Bounds(lb,ub)
        if (len(newIndividuals)<self.pareto_resolution):
            raise Exception("Number of individuals in the restart file is less than the population size."
                + " lower the population size or increase the DOE count(if restarting from a DOE)")

        # Do this before going into the train loop. This part of the code should happen after a new population is evaluated 
        individuals,best_point, worst_point, extreme_points = sort_and_select_population(individuals=newIndividuals,reference_points=ref_points, pop_size=self.pareto_resolution)
        
        all_individuals.extend(copy.deepcopy(newIndividuals))
        newIndividuals=individuals

        for pop in range(pop_start+1,pop_start+n_generations):  # Population Loop 
            ''' 
                Train a Neural network on Individuals. Initially this is all the individuals
            '''
            train_loss, test_loss = self.train(copy.deepcopy(all_individuals), False)   
            torch.save({'pop':pop,'state_dict':self.model.state_dict(),
                        'optimizer':self.optimizer.state_dict(),
                        'n_inputs':self.n_inputs,
                        'n_outputs':self.n_outputs},'ml_model.pt')   # This saved file can be used later when you want to plot for any input or output
            label_scalers = [self.label_scalers[l] for l in self.labels_str]
            feature_scalers = [self.feature_scalers[l] for l in self.features_str]
            '''
                Run parts of NSGA3 code to find intercepts 
            '''
            pareto_fronts = non_dominated_sorting(individuals,self.pareto_resolution)
            fitnesses = np.array([ind.objectives for f in pareto_fronts for ind in f])
            fitnesses *= -1            

            front_worst = np.max(fitnesses[:sum(len(f) for f in pareto_fronts),:],axis=0)
            intercepts = find_intercepts(extreme_points,best_point,worst_point,front_worst)

            features = np.array([ind.eval_parameters for ind in individuals])       # Normalized Features
            newIndividuals.clear()
            # surrogate_objective_func(res.x,self.model,ref_points,intercepts,o, best_point,label_scalers,feature_scalers)
            for o in range(self.pareto_resolution):
                x0 = features[random.randrange(0,len(features))]
                # least_squares(surrogate_objective_func,x0,jac=)
                res = minimize(surrogate_objective_func,x0,bounds=bounds,method=self.min_method,args=(self.model,ref_points,o,label_scalers,feature_scalers))
                surrogate_objective_func(res.x,self.model,ref_points,o,label_scalers,feature_scalers)                
                newIndividuals.append(
                    Individual(eval_parameters=set_eval_parameters(self.eval_parameters,copy.deepcopy(res.x)),
                        objectives=copy.deepcopy(self.objectives),performance_parameters=copy.deepcopy(self.performance_parameters))
                        )
            # newIndividuals = inverse_transform_data(self.label_scalers,self.feature_scalers,newIndividuals)
            # Evaluate
            self.evaluate_population(newIndividuals,pop)
            newIndividuals = self.read_population(pop)
            mse = compute_mse(individuals,newIndividuals)
            print(f"POP {pop} mse {mse:03e}")

            # Sort and select
            pop_diversity = diversity(newIndividuals)       # Calculate diversity 
            pop_dist = distance(individuals,newIndividuals) # Calculate population distance between past and future
            all_individuals.extend(copy.deepcopy(newIndividuals))
            newIndividuals.extend(individuals)              
            
            individuals,best_point, worst_point, extreme_points = sort_and_select_population(individuals=newIndividuals,reference_points=ref_points, pop_size=self.pareto_resolution)

            self.append_restart_file(newIndividuals)        # Keep the last designs
            self.append_history_file(pop,individuals[0],pop_diversity,pop_dist,train_loss,test_loss,mse)

            # if pop %4 ==0:
            #     all_individuals,_, _, _ = sort_and_select_population(individuals=all_individuals,reference_points=ref_points, pop_size=self.pareto_resolution*4)

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
            objectives (List[Parameter]): List of objectives 
        """
        self.objectives = objectives # Sets base class variable

    def add_performance_parameters(self,performance_params:List[Parameter] = None):
        """Add performance parameters 

        Args:
            performance_params (List[Parameter], optional): List of performance parameters. Defaults to None.
        """
        self.performance_parameters = performance_params # Sets base class variable 
    
    def start_doe(self,doe_individuals:List[Individual]=None,doe_size:int=128):
        """Starts a design of experiments. This generates the parameters for the individuals to be evaluated and executes each case. If the DOE has already started and there is an output file for an individual then the individual won't be evaluated    

        Args:
            doe_individuals (List[Individual], optional): List of individuals to evaluate. Defaults to None.
            doe_size (int, optional): Number of individuals to evaluate in the design of experiments. This is only used if doe_individuals is None. Defaults to 128.
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


