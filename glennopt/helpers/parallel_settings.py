from dataclasses import dataclass, field
from enum import Enum


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
    