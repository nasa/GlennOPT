import sys
sys.path.insert(0,'../')
from collections import OrderedDict
from glennopt.helpers.parameter import Parameter
import numpy as np
from typing import TypeVar,List


class Individual:
    '''
        This class represents each individual or "evaluation". In each evaluation there are a set of parameters for objectives as well as additional parameters that are kept track. 
    '''
    def __init__(self,eval_parameters:List[Parameter],objectives:List[Parameter],performance_parameters:List[Parameter]):
        '''
        Initializes an individual with objectives and a list of parameters
        '''
        self.__name = ""
        self.__population = -1
        self.__objectives = objectives
        self.__eval_parameters = eval_parameters
        self.__performance_parameters = performance_parameters
        #self.__perf_constraint_penalty = np.zeros(len(performance_parameters))
        #self.__obj_constraint_penalty = np.zeros(len(objectives))
    
    @property 
    def name(self):
        return self.__name

    @name.setter
    def name(self,v):
        self.__name = v

    @property 
    def population(self) -> int:
        return self.__population

    @population.setter
    def population(self,v:int):
        self.__population = v

    @property
    def objectives(self, include_constraint=True, C=0.5,a=2):
        y = np.zeros(len(self.__objectives))
        
        if (include_constraint): # apply constraint if constraint_value is negative
            y = self.__apply_dynamic_penalty(C=C,a=a)
        else:
            for i in range(len(self.__objectives)):
                y[i] = self.__objectives[i].value
            
        return y

    def __apply_dynamic_penalty(self,C,a):
        '''
            C = penalty coefficient, this reduces the impact of bad designs at higher populations
            a = penalty exponential coefficient, 
        '''
        perf_constraint = 0
        y = np.zeros(len(self.__objectives))
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
        population = max([1,self.population]) # this way we avoid nan values
        for i in range(len(self.__objectives)):
            y[i] = self.__objectives[i].value + 1.0/(2.0*np.power(C*population,a)) * np.power(constraint,2) # Dynamic Penalty with Equation(2)
        return y
    
    @objectives.setter
    def objectives(self,v):
        self.__objectives = v

    def get_objective(self,name):
        '''
            Returns a dictionary containing the parameters for an individual
        '''
        temp = next((item.value for item in self.__objectives if item.name == name), None)
        return temp

    def get_objectives_list(self):
        '''
            Returns a dictionary containing the parameters for an individual
        '''        
        return self.__objectives
    
    def set_objective(self,name,val):
        '''
            Assigns a value to a parameter for an individual
        '''
        for i in range(len(self.__objectives)):
            if (self.__objectives[i].name.lower() == name.lower()):
                self.__objectives[i].value = val        

    def set_objective_at_indx(self,indx,val):
        self.__eval_parameters[indx].value = val
    
    @property
    def eval_parameters(self):
        y = np.zeros(len(self.__eval_parameters))
        for i in range(len(self.__eval_parameters)):
            y[i] = self.__eval_parameters[i].value
        return y

    @eval_parameters.setter
    def eval_parameters(self,v):
        self.__eval_parameters = v
    
    def get_eval_parameter(self,name):
        '''
            Returns a dictionary containing the parameters for an individual
        '''
        temp = next((item.value for item in self.__eval_parameters if item.name == name), None)
        return temp
    
    def set_eval_parameter_at_indx(self,indx,val):
        self.__eval_parameters[indx].value = val

    def get_eval_parameter_list(self):
        '''
            Returns a dictionary containing the parameters for an individual
        '''        
        return self.__eval_parameters


    def set_eval_parameter(self,name,val):
        for i in range(len(self.__eval_parameters)):
            if (self.__eval_parameters[i].name.lower() == name.lower()):
                self.__eval_parameters[i].value = val
    
    ####

    @property
    def performance_parameters(self):
        y = np.zeros(len(self.__performance_parameters))
        for i in range(len(self.__performance_parameters)):
            y[i] = self.__performance_parameters[i].value
        return y

    @performance_parameters.setter
    def performance_parameters(self,v):
        self.__performance_parameters = v
    
    def get_performance_parameter(self,name):
        '''
            Returns a dictionary containing the parameters for an individual
        '''
        temp = next((item.value for item in self.__performance_parameters if item.name == name), None)
        return temp

    def set_performance_parameter_at_indx(self,indx,val):
        self.__eval_parameters[indx].value = val

    def set_performance_parameter(self,name,val):
        for i in range(len(self.__performance_parameters)):
            if (self.__performance_parameters[i].name.lower() == name.lower()):
                self.__performance_parameters[i].value = val

    def get_performance_parameters_list(self):
        '''
            Returns a dictionary containing the parameters for an individual
        '''        
        return self.__performance_parameters