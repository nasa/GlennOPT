import pyDOE2 as doe
import pandas as pd
import copy
from ..base import Individual, Parameter
import numpy as np
from tqdm import trange



class Base:
    def __init__(self):
       pass

    def add_parameter(self,name:str = None, min_value:float = None ,max_value:float = None,value_if_failed:float = 100000, constr_less_than:float = None, constr_greater_than:float = None)->None:
        self.eval_parameters.append(Parameter(name, min_value,max_value,value_if_failed, constr_less_than, constr_greater_than))

    def add_objectives(self,name:str = None, min_value:float = None ,max_value:float = None,value_if_failed:float = 100000, constr_less_than:float = None, constr_greater_than:float = None)->None:
        self.objectives.append(Parameter(name, min_value,max_value,value_if_failed, constr_less_than, constr_greater_than))

    def add_perf_parameter(self,name:str = None, min_value:float = None ,max_value:float = None,value_if_failed:float = 100000, constr_less_than:float = None, constr_greater_than:float = None)->None:
        self.perf_parameters.append(Parameter(name, min_value,max_value,value_if_failed, constr_less_than, constr_greater_than))

    
        
    def get_eval_value(self):
        design =self.create_design()
        values= design[[self.eval_parameters[i].name for i in range(len(self.eval_parameters))]].values
        return [tuple(x) for x in values]

    def generate_doe(self):
        individual=[]
        eval_values=self.get_eval_value()
        for i in trange(len(eval_values)):
            parameter = copy.deepcopy(self.eval_parameters)
            for indx in range(len(parameter)):
                parameter[indx].value=eval_values[i][indx]
            individual.append(Individual(eval_parameters=parameter,objectives=self.objectives,performance_parameters=self.perf_parameters))
        return individual
        
"""Create a experiment of your choice
"""         


class Default(Base):
    def __init__(self,number_of_evals):
        self.num_evals = number_of_evals
        self.eval_parameters=[]
        self.objectives=[]
        self.perf_parameters=[]

    def create_design(self):
        df = pd.DataFrame(data=[{self.eval_parameters[j].name : np.random.uniform(self.eval_parameters[j].min_value,self.eval_parameters[j].max_value,1)[0] for j in range(len(self.eval_parameters))} for i in range(self.num_evals) ])
        return df
    

class CCD(Base):
    def __init__ (self,number_of_parameters:int=2,center_points:tuple=(4,4),alpha:str="o",face:str="ccc"):
        
        self.num = number_of_parameters
        self.center= center_points
        self.alpha = alpha
        self.face = face 
        self.eval_parameters=[]
        self.objectives=[]
        self.perf_parameters=[]
        


    def coded_calculation(self,min_val,max_val,coded_variable):
        cal = (max_val-min_val)/2
        return (cal*coded_variable)+(min_val)

    def create_design(self):
        df = pd.DataFrame(data=doe.ccdesign(n=self.num,center=self.center, alpha=self.alpha, face=self.face), columns=[f'{self.eval_parameters[i].name}_coded'for i in range(self.num)])
        for i in range(self.num):
            df[self.eval_parameters[i].name]= df[f'{self.eval_parameters[i].name}_coded'].apply(lambda x:self.coded_calculation(self.eval_parameters[i].min_value,self.eval_parameters[i].max_value,x))
        return df
    


 
            
            

    

  





    







