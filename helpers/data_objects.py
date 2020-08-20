from dataclasses import dataclass, field
from enum import Enum


class de_mutation_type(Enum):
    """
        differential evolution mutation type. users can select what kind of mutation type to use 
    """
    de_1_rand_bin = 1
    de_best_2_bin = 2
    simple = 3

@dataclass
class mutation_parameters:
    """
        Data class for storing the mutation parameters used for NSGA and differential evolution problems 

        Properties:
            mutation_type:
            sigma: 
            mu: 
            F:
            C:
            min_parents = 0.1
            max_parents = 0.9
    """
    mutation_type: de_mutation_type = field(repr=True,default=de_mutation_type.de_1_rand_bin)
    sigma: float = field(repr=True,default=0.2)
    mu: float = field(repr=True,default=0.02)
    F: float = field(repr=True,default=0.6)
    C: float = field(repr=True,default=0.8)
    min_parents:float = field(repr=True,default=0.4)
    max_parents:float = field(repr=True,default=0.9)

@dataclass
class parallel_settings:
    """
        These settings control how the optimizer will execute. Example
        
        config = parallel_settings()
        ns = NSGA3()
        ns.parallel_settings = config

        Parameters:
            concurrent_executions - number of simultaneous executions the optimizer will launch
            
            cores_per_execution - number of cores for each execution. Specifies how to divide up the hostnames in the machine file

            execution_timeout - amount of minutes until the process is stopped
            
            machine_filename - this contains a list of hostnames 1 for each core
    """
    concurrent_executions: int = field(repr=True,default=1)
    cores_per_execution: int = field(repr=True,default=1)
    execution_timeout: int = field(repr=True,default=10)
    machine_filename = 'machinefile.txt'
    database_filename = 'database.csv'
    