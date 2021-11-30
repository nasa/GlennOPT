from dataclasses import dataclass, field
from enum import Enum
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class parallel_settings:
    """These settings control how the optimizer will execute. 
    Example: 
        .. code-block: python
            config = parallel_settings()
            ns = NSGA3()
            ns.parallel_settings = config
    
    Args:
        concurrent_executions (int): Number of concurrent executions per machine 
        cores_per_execution (int): number of cores per exectuion. Defaults to 1. Set it to 0 to ignore using cores per execution and just go with concurrent executions 
        execution_timeout (int): execution timeout in minutes. Defaults to 10 minutes 
        machine_filename (str): path to the machine file. Defaults to 'machinefile.txt'
        database_filename (str): path to the database. Defaults to 'database.csv'
        ignore_cores (bool): indicate whether to ignore the number of cores per execution and just execute the code

    """
    concurrent_executions: int = field(repr=True,default=1)
    cores_per_execution: int = field(repr=True,default=1)
    execution_timeout: int = field(repr=True,default=10)
    machine_filename = 'machinefile.txt'
    database_filename = 'database.csv'
