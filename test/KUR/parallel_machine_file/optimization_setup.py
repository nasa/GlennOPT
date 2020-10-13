"""
    Simple, non parallel optimization set up example. 
"""
import sys,os
sys.path.insert(0,'../../../')
from glennopt.helpers import Parameter, mutation_parameters, de_mutation_type, parallel_settings
from glennopt.nsga3 import NSGA3
from glennopt.doe import generate_reference_points


# Generate the DOE
current_dir = os.getcwd()
ns = NSGA3(eval_script = "Evaluation/evaluation.py", eval_folder="Evaluation",num_populations=10,pop_size=40,optimization_folder=current_dir)

eval_parameters = []
eval_parameters.append(Parameter(name="x1",min_value=-5,max_value=5))
eval_parameters.append(Parameter(name="x2",min_value=-5,max_value=5))
eval_parameters.append(Parameter(name="x3",min_value=-5,max_value=5))
ns.add_eval_parameters(eval_params = eval_parameters)

objectives = []
objectives.append(Parameter(name='objective1'))
objectives.append(Parameter(name='objective2'))
ns.add_objectives(objectives=objectives)

# No performance Parameters
performance_parameters = []
performance_parameters.append(Parameter(name='p1'))
performance_parameters.append(Parameter(name='p2'))
performance_parameters.append(Parameter(name='p3'))
ns.add_performance_parameters(performance_params = performance_parameters)

# Mutation settings
params = mutation_parameters
ns.mutation_params.mutation_type = de_mutation_type.de_best_2_bin

# Parallel settings
parallelSettings = parallel_settings()
parallelSettings.concurrent_executions = 4
parallelSettings.cores_per_execution = 4
parallelSettings.execution_timeout = 1 # minutes
# * These are not needed 
parallelSettings.machine_filename = 'machinefile.txt' 
# parallelSettings.database_filename = 'database.csv'
ns.parallel_settings = parallelSettings

ns.start_doe(doe_size=128)
ns.optimize_from_population(pop_start=-1,n_generations=10)



