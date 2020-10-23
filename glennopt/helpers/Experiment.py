import pyDOE2 as doe
import pandas as pd
from glennopt.helpers import Parameter
import numpy as np 

"""
Create a experiment of your choice
"""
    


class CCD:
    def __init__ (self,number_of_parameters:int=2,center_points:tuple=(4,4),alpha:str="o",face:str="ccc"):
        self.num = number_of_parameters
        self.center= center_points
        self.alpha = alpha
        self.face = face 
        self.eval_parameters=[]
    
    def add_parameter(self,name:str = None, min_value:float = None ,max_value:float = None,value_if_failed:float = 100000, constr_less_than:float = None, constr_greater_than:float = None)->None:
        self.eval_parameters.append(Parameter(name, min_value,max_value,value_if_failed, constr_less_than, constr_greater_than))



    def coded_calculation(self,min_val,max_val,coded_variable):
        cal = (max_val-min_val)/2
        return (cal*coded_variable)+(min_val)

    def create_design(self):
        df = pd.DataFrame(data=doe.ccdesign(n=self.num,center=self.center, alpha=self.alpha, face=self.face), columns=[f'{self.eval_parameters[i].name}_coded'for i in range(self.num)])
        for i in range(self.num):
            df[self.eval_parameters[i].name]= df[f'{self.eval_parameters[i].name}_coded'].apply(lambda x:self.coded_calculation(self.eval_parameters[i].min_value,self.eval_parameters[i].max_value,x))
        return df
    
    def get_eval_value(self):
        design =self.create_design()
        values= design[[self.eval_parameters[i].name for i in range(self.num)]].values
        return [tuple(x) for x in values]


           
            
            

    

  





    







