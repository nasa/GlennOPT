import pyDOE2 as doe
import pandas as pd





class CCD:
    def __init__ (self,parameter:object=None,number_of_parameters:int=2):
        self.name = parameter.name
        self.min = parameter.min_value
        self.max= parameter.max_value
        self.num = number_of_parameters

    def coded_calculation(self,min_val,max_val,coded_variable):
        cal = (max_val-min_val)/2
        return cal*(coded_variable)+(min_val)

    def create_design(self):
        df = pd.DataFrame(data=doe.ccdesign(n=self.num,center=(4,4)),columns=[f'x{i+1}_coded'for i in range(self.num)])
        for i in range(self.num):
            df[f'x{i+1}']= df[f"x{i+1}_coded"].apply(lambda x:self.coded_calculation(self.min,self.max,x))
        return df
    
    def _get_eval_value_(self,random_index):

        # should be at random 
        design =self.create_design()
    
        return design.loc[random_index,self.name]




