"""
    Simple, non parallel optimization set up example. 
"""
import sys,os
sys.path.insert(0,'../../../')
from glennopt.base import Parameter 
from glennopt.helpers import mutation_parameters, de_mutation_type
from glennopt.optimizers import NSGA3_ML
from glennopt.DOE import Default,CCD,FullFactorial,LatinHyperCube

# Generate the DOE
pop_size=32
current_dir = os.getcwd()
ns = NSGA3_ML(eval_command = "python evaluation.py", eval_folder="Evaluation",optimization_folder=current_dir,epochs=100, pop_size=pop_size,ml_evals=5)

# doe = Default(15) # Default
# doe = CCD()
# doe = FullFactorial(levels=8)
doe = LatinHyperCube(128)

doe.add_parameter(name="x1",min_value=-5,max_value=5)
doe.add_parameter(name="x2",min_value=-5,max_value=5)
doe.add_parameter(name="x3",min_value=-5,max_value=5)
ns.add_eval_parameters(eval_params=doe.eval_parameters)

doe.add_objectives(name='objective1')
doe.add_objectives(name='objective2')
ns.add_objectives(objectives=doe.objectives)

# No performance Parameters
doe.add_perf_parameter(name='p1')
doe.add_perf_parameter(name='p2')
doe.add_perf_parameter(name='p2')
ns.add_performance_parameters(performance_params= doe.perf_parameters)

# Parallel Settings (You don't need to run this block if you only want serial execution)
ns.parallel_settings.concurrent_executions = 8    # Change to 1 for serial
ns.parallel_settings.cores_per_execution = 0   
ns.parallel_settings.execution_timeout = 0.2      # minutes

ns.mutation_params.mutation_type = de_mutation_type.de_rand_1_bin   # Choice of de_best_1_bin (single objective) or de_rand_1_bin (multi-objective)
ns.mutation_params.F = 0.8
ns.mutation_params.C = 0.7

ns.start_doe(doe.generate_doe())
ns.optimize_from_population(pop_start=-1,n_generations=40)

