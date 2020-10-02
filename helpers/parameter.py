
class Parameter:
    def __init__(self,name:str = None, min_value:float = None ,max_value:float = None,value_if_failed:float = 100000, constr_less_than:float = None, constr_greater_than:float = None)->None:
        '''
            This represents a value that a single evaluation keeps track of. For example the objective or if this is an computational simulation, we keep track of the stress, strain, volume, etc., anything that isn't a constraint. 

            Inputs:
                name: (string) name of the parameter 
                lower_value: (float) this represents the lower bound of the variable
                upper_value: (float) this represents the upper bound of the variable
                value_if_failed: (float) this is set to 100000, probably a good value for a minimization problem
                constr_less_than: (float) default is None for no constraint
                constr_greater_than: (float) default is None for no constraint

        '''
        self.parameter = {}
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.value_if_failed = value_if_failed
        self.constraint_less_than = constr_less_than
        self.constraint_greater_than = constr_greater_than
        self.value = value_if_failed # * Setting to default Value
    