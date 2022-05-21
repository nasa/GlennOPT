from cProfile import label
from typing import Dict, List, Tuple
import torch 
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ..base import Individual
import numpy as np 
from sklearn.metrics import mean_squared_error

@torch.no_grad()
def evaluation_func(individuals:List[Individual], models:List[nn.Module], label_scalers:Dict[str,MinMaxScaler], feature_scalers:Dict[str,MinMaxScaler]):
    """Evaluates the individuals with the neural network function 

    Args:
        x0 (np.ndarray): [description]
        models (List[nn.Module]): [description]
        reference_points (List[np.ndarray]): [description]
        intercepts (np.ndarray)
    """
    individuals, _, _, labels_str, features_str = transform_data(individuals, label_scalers, feature_scalers)    

    features = np.array([ind.eval_parameters for ind in individuals])
    features = torch.as_tensor(features,dtype=torch.float32)
    fitnesses = list()
    for i in range(len(models)):
        fitnesses.append(models[i](features))
    fitnesses = torch.concat(fitnesses,dim=1)

    for i in range(len(individuals)):
        for j,l in enumerate(labels_str):
            value = label_scalers[l].inverse_transform(fitnesses[i,j].detach().numpy().reshape(1,-1))
            individuals[i].set_objective(l, value[0,0])

        for k,l in enumerate(features_str):
            value = feature_scalers[l].inverse_transform(features[i,k].detach().numpy().reshape(1,-1))
            individuals[i].set_eval_parameter(l, value[0,0])

    return individuals

   
def transform_data(individuals:List[Individual],label_scalers:Dict[str,MinMaxScaler]=None, feature_scalers:Dict[str,MinMaxScaler]=None) -> List[Individual]:
    """Normalizes all the individuals and returns the label scaler for objectives and feature scalers for the evaluation parameters 

    Args:
        individuals (List[Individual]): Unnormalized list of individuals 
        label_scalers (Dict[str,MinMaxScaler]): scalers that can be used to untransform the objectives. Optional, Default = None
        feature_scalers (Dict[str,MinMaxScaler]): scalers that can be used to untransform the evaluation parameters. Optional, Default = None

    Returns:
        (tuple): Tuple containing

            **Individuals** (List[Individual]): Normalized list of individuals
            **label_scalers** (Dict[str,MinMaxScaler]): scalers that can be used to untransform the objectives
            **feature_scalers** (Dict[str,MinMaxScaler]): scalers that can be used to untransform the evaluation parameters

    """
    individuals = [ind for ind in individuals if ind.IsFailed==False] # remove failed simulations

    labels = np.array([ind.objectives for ind in individuals])
    labels_str = [o.name for o in individuals[0].get_objectives_list()]

    features = np.array([ind.eval_parameters for ind in individuals])
    eval_parameters = individuals[0].get_eval_parameter_list()
    feature_bounds = [(ep.min_value, ep.max_value) for ep in eval_parameters]
    features_str = [f.name for f in individuals[0].get_eval_parameter_list()]

    def get_scalers(data:np.ndarray,data_keys:List[str], bounds:List[Tuple[float,float]]=None):
        """This returns the scaled data

        Args:
            data (np.ndarray): data as a numpy array
            data_keys (List[str]): [description]
            bounds (List[Tuple[float,float]], optional): If you specify a list of min and max [(0.1, 0.5)], we use these ranges to scale your features. Defaults to None.

        Returns:
            (tuple): containing

                **scalers** (MinMaxScaler): minmax scaler object
                **new_data** (np.ndarray): scaled data
        """
        new_data = np.zeros(data.shape)
        scalers = dict() # Scale each label
        for i in range(len(data_keys)):
            label_name = data_keys[i]
            scalers[label_name] = MinMaxScaler(feature_range=(0,1))
            # scalers[label_name] = StandardScaler()
            if bounds:
                scalers[label_name].fit(np.array(bounds[i]).reshape(-1,1))
            else:
                scalers[label_name].fit(data[:,i].reshape(-1,1))
            new_data[:,i] = scalers[label_name].transform(data[:,i].reshape(-1,1)).flatten()
        return scalers, new_data
    
    if label_scalers == None:
        label_scalers, labels = get_scalers(labels, labels_str)
    else:
        _, labels = get_scalers(labels, labels_str)
    
    if feature_scalers == None:
        feature_scalers, features = get_scalers(features, features_str,feature_bounds)
        # feature_scalers, features = get_scalers(features, features_str)
    else:
        _, features = get_scalers(features, features_str,feature_bounds)
        # _, features = get_scalers(features, features_str)

    for i in range(labels.shape[0]):
        for j in range(len(labels_str)):
            individuals[i].set_objective(labels_str[j],labels[i,j])

    for i in range(features.shape[0]):
        for j in range(len(features_str)):
            individuals[i].set_eval_parameter(features_str[j],features[i,j])

    return individuals, label_scalers, feature_scalers, labels_str, features_str

def inverse_transform_data(label_scalers:Dict[str,MinMaxScaler], feature_scalers:Dict[str,MinMaxScaler],individuals:List[Individual]) -> List[Individual]:
    """Inverse scales the data. Data is scaled for machine learning, this scales it back to original values 

    Args:
        label_scalers (Dict[str,MinMaxScaler]): [description]
        feature_scalers (Dict[str,MinMaxScaler]): [description]
        individuals (List[Individual]): [description]

    Returns:
        List[Individual]: [description]
    """
    individuals = [ind for ind in individuals if ind.IsFailed==False] # remove failed simulations

    labels = np.array([ind.objectives for ind in individuals])
    labels_str = [o.name for o in individuals[0].get_objectives_list()]

    features = np.array([ind.eval_parameters for ind in individuals])
    features_str = [f.name for f in individuals[0].get_eval_parameter_list()]
    
    for i in range(len(labels_str)):
        label_str = labels_str[i]
        labels[:,i] = label_scalers[label_str].inverse_transform(labels[:,i].reshape(-1,1)).flatten()
    
    for i in range(len(features_str)):
        feature_str = features_str[i]
        features[:,i] = feature_scalers[feature_str].inverse_transform(features[:,i].reshape(-1,1)).flatten()

    for i in range(labels.shape[0]):
        for j in range(len(labels_str)):
            individuals[i].set_objective(labels_str[j],labels[i,j])

    for i in range(features.shape[0]):
        for j in range(len(features_str)):
            individuals[i].set_eval_parameter(features_str[j],features[i,j])
    return individuals

class objective_weighted_loss():
    def __call__(self, labels:torch.Tensor,labels_actual:torch.Tensor,weights:torch.Tensor=None):
        """Weighted Loss based on minimum objective value 

        Weighted error formula https://datascience.stackexchange.com/questions/66326/need-of-weighted-mean-squared-error 

        Args:
            labels (torch.Tensor): predicted value
            labels_actual (torch.Tensor): Actual value
            weights (List[float]): weights for each individual, if this is none then standard mse is computed
        """
        if weights == None:
            weights = torch.ones(labels.shape[0])
        return 1/len(weights) * torch.sum(weights*(labels - labels_actual)**2) / torch.sum(weights)
        

def compute_weights(pareto_group:List[List[Individual]],decay_rate:float=0.5) -> List[float]:
    """Weights are based on how close the individual is to the pareto front. 

    Example:
    >>> chosen,best_point, worst_point, extreme_points = sort_and_select_population(individuals=newIndividuals,reference_points=ref_points, pop_size=self.pop_size)
    >>> weights = compute_weights()

    Args:
        pareto_group (List[List[Individual]]): Pareto groups 
        decay_rate (float, optional): _description_. Defaults to 0.2.

    Returns:
        List[float]: list of weights assigned to each individual
    """
    start_weight = 1 # starting weight
    n_pareto = len(pareto_group) # This is number of pareto fronts 
    group_weights = [start_weight * decay_rate**i  for i in range(n_pareto)] # Weight for each pareto group
    all_weights = list()
    for i in range(len(pareto_group)):
        weights = list()
        for _ in pareto_group[i]:
            weights.append(group_weights[i])
        all_weights.append(weights)
    return flatten_list_of_list(all_weights)

def flatten_list_of_list(regular_list:List[List[object]]) -> List[object]:
    """Flattens a list of list 
    https://stackabuse.com/python-how-to-flatten-list-of-lists/

    Args:
        regular_list (List[List[object]]): _description_

    Returns:
        List[object] : Returns a flattened list
    """
    return [item for sublist in regular_list for item in sublist]


def compute_loss(ml_individuals:List[Individual],eval_individuals:List[Individual], weights:torch.Tensor):
    """_summary_

    Args:
        ml_individuals (List[Individual]): _description_
        eval_individuals (List[Individual]): _description_
        weights (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """

    loss_fn = objective_weighted_loss()
    passed_simulations = [i for i in range(len(eval_individuals)) if eval_individuals[i].IsFailed==False] # remove failed simulations

    ml_individuals = [ml_individuals[i] for i in passed_simulations]
    eval_individuals = [eval_individuals[i] for i in passed_simulations]

    objectives1 = torch.tensor([ind.objectives for ind in ml_individuals])
    objectives2 = torch.tensor([ind.objectives for ind in eval_individuals])
    losses = 0  
    for i in range(objectives1.shape[1]):
        losses += loss_fn(objectives1[:,i], objectives2[:,i],torch.tensor(weights))
    losses /= objectives1.shape[1] 
    return losses
    # mse = ((objectives1 - objectives2)**2).mean(axis=0)
    # return mse
