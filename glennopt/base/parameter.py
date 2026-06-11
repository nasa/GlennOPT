from dataclasses import dataclass, field, fields
from enum import Enum
from typing import List
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Parameter:
    """This represents a value that a single evaluation keeps track of. For example the objective or if this is an computational simulation, we keep track of the stress, strain, volume, etc., anything that isn't a constraint.

        Inputs:
            name: (string) name of the parameter
            min_value: (float) lower bound of the variable
            max_value: (float) upper bound of the variable
            value_if_failed: (float) assigned when an evaluation fails. Defaults to 10000 (suits minimization).
            constraint_greater_than: (float) optional lower constraint. Defaults to None.
            constraint_less_than: (float) optional upper constraint. Defaults to None.
            test_value: (float) optional baseline / nominal value. Useful for seeding a smoke-test individual or for downstream tooling to know the "default" point in the design space. Not used by the optimizer itself.
    """
    parameter = dict()
    name: str
    min_value:float = None
    max_value:float = None
    value_if_failed:float = 10000
    constraint_greater_than:float = None
    constraint_less_than:float = None
    value:float = 10000
    test_value:float = None

class ParameterList:
    parameters:List[Parameter]
