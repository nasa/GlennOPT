"""
    Simple, non parallel optimization set up example. 
"""
import sys,os
sys.path.insert(0,'../../../')
from glennopt.base import Parameter
from glennopt.helpers import mutation_parameters, de_mutation_type
from glennopt.nsga3 import NSGA3
from glennopt.doe import Default,CCD

# Generate the DOE
pop_size=32
current_dir = os.getcwd()
ns = NSGA3(eval_script = "Evaluation/evaluation.py", eval_folder="Evaluation",pop_size=pop_size,optimization_folder=current_dir)

doe = CCD() 

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

ns.mutation_params.mutation_type = de_mutation_type.de_rand_1_bin
ns.mutation_params.F = 0.8
ns.mutation_params.C = 0.7
ns.start_doe(doe.generate_doe())
ns.optimize_from_population(pop_start=-1,n_generations=50)