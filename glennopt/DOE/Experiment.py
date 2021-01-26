import pandas as pd
import itertools
import copy
from ..base import Individual, Parameter
import numpy as np
from tqdm import trange
from doepy import build
i

class Base:
    def __init__(self):
        self.objectives = list()
        self.perf_parameters = list()
        self.eval_parameters = list()

        self.num_parameters = 0
        self.num_objectives = 0
        self.num_perf_parameter = 0

    def add_parameter(self,name:str = None, min_value:float = None ,max_value:float = None,value_if_failed:float = 100000, constr_less_than:float = None, constr_greater_than:float = None)->None:
        self.eval_parameters.append(Parameter(name, min_value,max_value,value_if_failed, constr_less_than, constr_greater_than))
        self.num_parameters = len(self.eval_parameters)

    def add_objectives(self,name:str = None, min_value:float = None ,max_value:float = None,value_if_failed:float = 100000, constr_less_than:float = None, constr_greater_than:float = None)->None:
        self.objectives.append(Parameter(name, min_value,max_value,value_if_failed, constr_less_than, constr_greater_than))
        self.num_objectives = len(self.objectives)

    def add_perf_parameter(self,name:str = None, min_value:float = None ,max_value:float = None,value_if_failed:float = 100000, constr_less_than:float = None, constr_greater_than:float = None)->None:
        self.perf_parameters.append(Parameter(name, min_value,max_value,value_if_failed, constr_less_than, constr_greater_than))
        self.num_perf_parameter = len(self.perf_parameters)

    
    def get_eval_value(self):
        design =self.create_design()
        values= design[[self.eval_parameters[i].name for i in range(len(self.eval_parameters))]].values
        return [tuple(x) for x in values]

    def generate_doe(self):
        individuals=list()
        eval_values=self.get_eval_value()
        for i in trange(len(eval_values)):
            parameter = copy.deepcopy(self.eval_parameters)
            for indx in range(len(parameter)):
                parameter[indx].value=eval_values[i][indx]
            individuals.append(Individual(eval_parameters=parameter,objectives=self.objectives,performance_parameters=self.perf_parameters))
        return individuals

    def read_json(filename):
        pass

    def write_json(filename):
            

class Default(Base):
    def __init__(self,number_of_evals):
        '''
            This defaults to creating a randomized set of design of experiments 
        '''
        super(Default, self).__init__()
        self.num_evals = number_of_evals
        self.eval_parameters=[]
        self.objectives=[]
        self.perf_parameters=[]

    def create_design(self):
        df = pd.DataFrame(data=[{self.eval_parameters[j].name : np.random.uniform(self.eval_parameters[j].min_value,self.eval_parameters[j].max_value,1)[0] for j in range(len(self.eval_parameters))} for i in range(self.num_evals) ])
        return df


class LatinHyperCube(Base):   
    def __init__ (self,samples=10,levels=4):
        '''
            This method is samples the design space in smaller cells.
            
            Inputs: 
                samples - number of evaluations 
                levels - number of divisions of the evaluation parameters (this breaks it into cubes)
            Citations:
                https://pythonhosted.org/pyDOE/randomized.html
                https://doepy.readthedocs.io/en/latest/
        '''
        super(LatinHyperCube, self).__init__()
        self.samples = samples 
        self.levels = levels

    def create_design(self):
        param_dict = dict()
        for p in self.eval_parameters:
            r = np.linspace(p.min_value ,p.max_value ,self.levels)
            param_dict[p.name] = r.tolist()
        df = build.space_filling_lhs(param_dict, self.samples)
        return df

class CCD(Base):
    def __init__ (self,center_points:tuple=(4,4),alpha:str="o",face:str="ccc"):
        '''
            Central Composite Design

            Inputs
                center points (4,4) 4 - factorial block (4 = square) These are the edges. the other 4 is the + sign. You can change this.
                alpha = "o" orthogonal or "r" for rotatable
                face = circumscribed "ccc" default, faced"ccf", inscribed "cci"                
        '''
        super(CCD, self).__init__()
        self.center= center_points
        self.alpha = alpha
        self.face = face 

    def create_design(self):
        param_dict = dict((p.name, [p.min_value, p.max_value]) for p in self.eval_parameters)    
        df = build.central_composite(param_dict,face=self.face)
        return df
    
class FullFactorial(Base):
    def __init__(self,levels=4):
        '''
            Factorial based design of experiments. number of evaluations scale with 2^(level-1)
            
            Inputs
                levels = 4
        '''
        super(FullFactorial, self).__init__()
        self.levels=levels
        
    def create_design(self):        
        param_dict = dict() 
        for p in self.eval_parameters:
            r = np.linspace(p.min_value ,p.max_value ,self.levels)
            param_dict[p.name] = r.tolist()
            
        df = build.full_fact(param_dict)
        return df

            
            

    

  





    







