"""
    Simple, non parallel optimization set up example. 
"""
import sys,os
sys.path.insert(0,'../../../')
from glennopt.helpers import mutation_parameters, de_mutation_type, parallel_settings
from glennopt.base import Parameter
from glennopt.optimizers import NSGA3
from glennopt.DOE import FullFactorial, CCD, LatinHyperCube, Default


# Generate the DOE
current_dir = os.getcwd()
pop_size = 32
ns = NSGA3(eval_command = "python evaluation.py", eval_folder="Evaluation",pop_size=pop_size,optimization_folder=current_dir)

doe = LatinHyperCube(128)


doe.add_parameter(name="x1",min_value=-5,max_value=5)
doe.add_parameter(name="x2",min_value=-5,max_value=5)
doe.add_parameter(name="x3",min_value=-5,max_value=5)
ns.add_eval_parameters(eval_params=doe.eval_parameters)     # Add the evaluation parameters from doe to NSGA3

doe.add_objectives(name='objective1')
doe.add_objectives(name='objective2')
ns.add_objectives(objectives=doe.objectives)

# No performance Parameters
doe.add_perf_parameter(name='p1')
doe.add_perf_parameter(name='p2')
doe.add_perf_parameter(name='p3')
ns.add_performance_parameters(performance_params = doe.perf_parameters)

# Mutation settings
ns.mutation_params.mutation_type = de_mutation_type.de_rand_1_bin
ns.mutation_params.F = 0.4
ns.mutation_params.C = 0.7

# Parallel settings
p = parallel_settings()
p.concurrent_executions = 10
p.cores_per_execution = 2
p.execution_timeout = 1 # minutes
p.machine_filename = 'machinefile.txt'  
ns.parallel_settings = p

ns.start_doe(doe.generate_doe())
ns.optimize_from_population(pop_start=-1,n_generations=50)



