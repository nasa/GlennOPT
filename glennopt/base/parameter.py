from dataclasses import dataclass, field, fields
from typing import List
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Parameter:
    """This represents a value that a single evaluation keeps track of. For example the objective or if this is an computational simulation, we keep track of the stress, strain, volume, etc., anything that isn't a constraint. 

        Inputs:
            name: (string) name of the parameter 
            lower_value: (float) this represents the lower bound of the variable
            upper_value: (float) this represents the upper bound of the variable
            value_if_failed: (float) this is set to 100000, probably a good value for a minimization problem
            constr_less_than: (float) default is None for no constraint
            constr_greater_than: (float) default is None for no constraint
    """
    parameter = dict()
    name: str
    min_value:float = None
    max_value:float = None
    value_if_failed:float = 10000
    constraint_greater_than:float = None
    constraint_less_than:float = None
    value:float = 10000