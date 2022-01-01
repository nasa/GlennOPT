import sys
from typing import TypeVar,List
from .parameter import Parameter
import numpy as np

class Individual:
    """This class represents each individual or "evaluation". In each evaluation there are a set of parameters for objectives as well as additional parameters that are kept track. 
    """
    
    def __init__(self,eval_parameters:List[Parameter],objectives:List[Parameter],performance_parameters:List[Parameter]):
        """Initializes an individual with objectives and a list of parameters

        Args:
            eval_parameters (List[Parameter]): List of evaluation parameters. Evaluation parameters have a name and a min and max bound. This gives the optimizer a search space
            objectives (List[Parameter]): List of objectives. Each objective has a name and can be constrained 
            performance_parameters (List[Parameter]): List of performance parameters, stuff that you also want to track and may want to constrain, if you want.
        """
        self.__name = ""
        self.__population = -1
        self.__objectives = objectives
        self.__eval_parameters = eval_parameters
        self.__performance_parameters = performance_parameters
        
    
    def __hash__(self):        
        return hash(tuple(self.objectives))

    def __str__(self):        
        return str(tuple(self.objectives))
    
    def __repr__(self):
        """Returns a readable format when you are debugging using an IDE that is not jupyter notebook 

        Returns:
            str: readable format for debugging 
        """
        return "%s.(%s)(%s)" % (self.__class__.__name__,str(tuple(self.objectives)), str(tuple(self.eval_parameters)))

    @property 
    def name(self) -> str:
        """Name of the individual evaluation. For example IND000 

        Returns:
            str: name of the individual
        """
        return self.__name

    @name.setter
    def name(self,v:str):
        """Settor for name property

        Args:
            v (str): new name
        """
        self.__name = v

    @property 
    def population(self) -> int:
        """returns the population 

        Returns:
            int: [description]
        """
        return self.__population

    @population.setter
    def population(self,v:int):
        self.__population = v

    @property
    def objectives(self, include_constraint=True, C=0.5,a=2) -> np.ndarray:
        """Returns a numpy array of the objective values 

        Args:
            include_constraint (bool, optional): Whether or not to include constraints. Value of the objects are skewed if constraints are violated. Defaults to True.
            C (float, optional): Penalty coefficient. Defaults to 0.5.
            a (int, optional): Penalty exponential coefficient. Defaults to 2.

        Returns:
            np.ndarray: Array of values. For example [1,2]
        """
        y = np.zeros(len(self.__objectives))
        
        if (include_constraint): # apply constraint if constraint_value is negative
            y = self.__apply_dynamic_penalty(C=C,a=a)
        else:
            for i in range(len(self.__objectives)):
                y[i] = self.__objectives[i].value
            
        return y
    
    def constraints(self) -> np.ndarray:
        """Get the constraint matrix 

        Returns:
            np.ndarray: Return the constraint matrix 
        """
        perf_constraint = 0        
        if (self.performance_parameters is not None):
            for j in range(len(self.performance_parameters)):
                if (self.__performance_parameters[j].constraint_less_than is not None):
                    perf_constraint += (self.__performance_parameters[j].value - self.__performance_parameters[j].constraint_less_than)
                if (self.__performance_parameters[j].constraint_greater_than is not None):
                    perf_constraint += (self.__performance_parameters[j].constraint_greater_than - self.__performance_parameters[j].value)
        obj_constraint = 0
        for i in range(len(self.__objectives)):
            if (self.__objectives[i].constraint_less_than is not None):
                obj_constraint += (self.__objectives[i].constraint_less_than - self.__objectives[i].value)
            if (self.__objectives[i].constraint_greater_than is not None):
                obj_constraint += (self.__objectives[i].value - self.__objectives[i].constraint_greater_than)
        constraint = max([0,perf_constraint,obj_constraint])
        return constraint

    def __apply_dynamic_penalty(self,C:float,a:float) -> np.ndarray:
        """Applies dynamic penalty to the objectives 

        Args:
            C (float): penalty coefficient, this reduces the impact of bad designs at higher populations
            a (float): penalty exponential coefficient

        Returns:
            np.ndarray: objectives after the constraint has been applied 
        """
        y = np.zeros(len(self.__objectives))
        constraint = self.constraints()

        population = max([1,self.population]) # this way we avoid nan values
        for i in range(len(self.__objectives)):
            y[i] = self.__objectives[i].value + 1.0/(2.0*np.power(C*population,a)) * np.power(constraint,2) # Dynamic Penalty with Equation(2)
        return y
    
    @objectives.setter
    def objectives(self,v:List[Parameter]):
        """setter for objectives

        Args:
            v (List[Parameter]): array of parameters to set the objectives 
        """
        self.__objectives = v

    def get_objective(self,name:str) -> Parameter:
        """Searches for an objective based on a name provided and returns the objective as a parameter object

        Args:
            name (str): name of the objective. Example: "objective 1"

        Returns:
            Parameter: objective as parameter object 
        """
        temp = next((item.value for item in self.__objectives if item.name == name), None)
        return temp

    def get_objectives_list(self) -> List[Parameter]:
        """Returns a list of objectives as parameters 

        Returns:
            List[Parameter]: List of objectives 
        """        
        return self.__objectives
    
    def set_objective(self,name:str,val:float):
        """Assigns a value to a parameter for an individual
            typically called by the optimizer objectives[2] = 450
            
        Args:
            name (str): Name of the objective. Example: "Objective 1"
            val (float): value to set the objective to
        """
        for i in range(len(self.__objectives)):
            if (self.__objectives[i].name.lower() == name.lower()):
                self.__objectives[i].value = val     
                break
    
    @property
    def eval_parameters(self) -> np.ndarray:
        """Get eval parameters 

        Returns:
            np.ndarray: Returns a numpy array with the evaluation parameters 
        """
        y = np.zeros(len(self.__eval_parameters))
        for i in range(len(self.__eval_parameters)):
            y[i] = self.__eval_parameters[i].value
        return y
    
    @property
    def eval_parameter_min(self) -> np.ndarray:
        """Gets the minimum value of all the evaluation parameters

        Returns:
            np.ndarray: Array containing the minimum values of the evaluation paramters. 
        """
        y = np.zeros(len(self.__eval_parameters))
        for i in range(len(self.__eval_parameters)):
            y[i] = self.__eval_parameters[i].min_value
        return y
    
    @property
    def eval_parameter_max(self) -> np.ndarray:
        """returns the maximum value of each evaluation parameter as a numpy array 

        Returns:
            np.ndarray: Array containing maximum values of evaluation parameter
        """
        y = np.zeros(len(self.__eval_parameters))
        for i in range(len(self.__eval_parameters)):
            y[i] = self.__eval_parameters[i].max_value
        return y

    @eval_parameters.setter
    def eval_parameters(self,v):
        self.__eval_parameters = v
    
    def get_eval_parameter(self,name:str) -> Parameter:
        """Gets the evaluation parameter containing name

        Args:
            name (str): Name of the evaluation parameter to get

        Returns:
            Parameter: Parameter object containing name or None
        """        
        temp = next((item.value for item in self.__eval_parameters if item.name == name), None)
        return temp
    
    def set_eval_parameter_at_indx(self,indx:int,val:float):
        """Set the evaluation parameter value at a particular index

        Args:
            indx (int): Index in parameter array
            val (float): Value to replace with
        """
        self.__eval_parameters[indx].value = val

    def get_eval_parameter_list(self) -> List[Parameter]:
        """Gets all the evaluation parameters 

        Returns:
            List[Parameter]: List of evaluation parameters 
        """
        return self.__eval_parameters

    def set_eval_parameter(self,name:str,val:float):
        """set the evaluation parameter value given a name

        Args:
            name (str): Name of the parameter
            val (float): value to set to
        """
        for i in range(len(self.__eval_parameters)):
            if (self.__eval_parameters[i].name.lower() == name.lower()):
                self.__eval_parameters[i].value = val
    
    @property
    def performance_parameters(self) -> np.ndarray:
        """Gets the performance parameters as a numpy array

        Returns:
            np.ndarray: numpy array of performance parameters
        """
        if (self.__performance_parameters is not None):
            y = np.zeros(len(self.__performance_parameters))
            for i in range(len(self.__performance_parameters)):
                y[i] = self.__performance_parameters[i].value
            return y
        return None

    @performance_parameters.setter
    def performance_parameters(self,v:List[Parameter]):
        """Sets the performance parameters 

        Args:
            v (List[Parameter]): List of performance parameters defined as Parameter 
        """
        self.__performance_parameters = v
    
    def get_performance_parameter(self,name:str) -> Parameter:
        """Gets the performance parameter given a name

        Args:
            name (str): name of performance parameter to get 

        Returns:
            Parameter: Performance parameter or None
        """
        temp = next((item.value for item in self.__performance_parameters if item.name == name), None)
        return temp

    def set_performance_parameter_at_indx(self,indx:int,val:float):
        """Sets the performance parameter at an index 

        Args:
            indx (int): index of performance parameter 
            val (float): [description]
        """
        self.__eval_parameters[indx].value = val

    def set_performance_parameter(self,name:str,val:float):
        """Sets the performance parameter given the name

        Args:
            name (str): name of the parameter 
            val (float): value to set it to 
        """
        for i in range(len(self.__performance_parameters)):
            if (self.__performance_parameters[i].name.lower() == name.lower()):
                self.__performance_parameters[i].value = val

    def get_performance_parameters_list(self) -> List[Parameter]:
        """Gets a list of performance parameters 

        Returns:
            List[Parameter]: list of performance parameters 
        """
        return self.__performance_parameters

    @property
    def IsFailed(self) -> bool:
        """Checks the objectives to make sure the individual hasn't failed

        Returns:
            bool: True = failed, False = passed
        """
        objectives = [True for o in self.get_objectives_list() if o.value_if_failed == o.value]
        return any(objectives)  # Returns true if any of the values are True
