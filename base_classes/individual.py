from collections import OrderedDict
import glennopt.helpers.parameter as Parameter
import numpy as np
from typing import TypeVar,List
T = TypeVar('T', bound='Parameter')
parameter_list = List[T]


class Individual:
    '''
        This class represents each individual or "evaluation". In each evaluation there are a set of parameters for objectives as well as additional parameters that are kept track. 
    '''
    def __init__(self,eval_parameters:parameter_list,objectives:parameter_list,performance_parameters:parameter_list=[]):
        '''
        Initializes an individual with objectives and a list of parameters
        '''
        self.__name = ""
        self.__population = -1
        self.__objectives = objectives
        self.__eval_parameters = eval_parameters
        self.__performance_parameters = performance_parameters
    
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
    def objectives(self):
        y = np.zeros(len(self.__objectives))
        for i in range(len(self.__objectives)):
            y[i] = self.__objectives[i].value
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

    # def apply_constraints(self):
    #     for indx in range(len(self.objectives)): # I could do a 
    #         if self.objectives[indx].value >