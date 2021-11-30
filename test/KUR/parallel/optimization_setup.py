"""
    Simple, non parallel optimization set up example. 
"""
import sys,os
sys.path.insert(0,'../../../')
from glennopt.base import Parameter
from glennopt.helpers import parallel_settings, mutation_parameters, de_mutation_type
from glennopt.optimizers import NSGA3
from glennopt.DOE import FullFactorial, CCD, LatinHyperCube, Default
# Generate the DOE
pop_size=32
current_dir = os.getcwd()
ns = NSGA3(eval_command = "python evaluation.py", eval_folder="Evaluation",pop_size=pop_size,optimization_folder=current_dir)


# doe = Default(15) # Default
# doe = CCD()
doe = FullFactorial(levels=8)
# doe = LatinHyperCube(128)

doe.add_parameter(name="x1",min_value=-5,max_value=5)
doe.add_parameter(name="x2",min_value=-5,max_value=5)
doe.add_parameter(name="x3",min_value=-5,max_value=5)
ns.add_eval_parameters(eval_params=doe.eval_parameters)

doe.add_objectives(name='objective1')
doe.add_objectives(name='objective2')
ns.add_objectives(objectives=doe.objectives)


objectives = []
objectives.append(Parameter(name='objective1'))
objectives.append(Parameter(name='objective2'))
ns.add_objectives(objectives=objectives)

doe.add_perf_parameter(name='p1')
doe.add_perf_parameter(name='p2')
doe.add_perf_parameter(name='p2')
ns.add_performance_parameters(performance_params= doe.perf_parameters)

# Mutation settings
ns.mutation_params.mutation_type = de_mutation_type.simple
ns.mutation_params.F = 0.6
ns.mutation_params.C = 0.7
# Parallel settings

ns.parallel_settings.concurrent_executions = 64
ns.parallel_settings.cores_per_execution = 1
ns.parallel_settings.execution_timeout = 1 # minutes
# * These are not needed 
# parallelSettings.machine_filename = 'machinefile.txt' 
# parallelSettings.database_filename = 'database.csv'

ns.start_doe(doe.generate_doe())
ns.optimize_from_population(pop_start=-1,n_generations=50)
