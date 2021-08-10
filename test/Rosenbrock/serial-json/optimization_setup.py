"""
    Simple, non parallel optimization set up example. 
"""
import sys,os
sys.path.insert(0,'../../../')
from glennopt.base import Parameter
from glennopt.optimizers import SODE
from glennopt.helpers import de_mutation_type, mutation_parameters
from glennopt.DOE import Default,CCD,FullFactorial,LatinHyperCube

# Generate the DOE
current_dir = os.getcwd()
pop_size = 20
sode = SODE(eval_command = "python evaluation.py", eval_folder="Evaluation",pop_size=pop_size,optimization_folder=current_dir)

# doe = Default(15) # Default
# doe = CCD()
doe = FullFactorial(levels=8)
# doe = LatinHyperCube(128)

doe.add_parameter(name="x1",min_value=-3,max_value=3)
doe.add_parameter(name="x2",min_value=-3,max_value=3)
sode.add_eval_parameters(eval_params=doe.eval_parameters)

doe.add_objectives(name='objective1')
sode.add_objectives(objectives=doe.objectives)

# No performance Parameters
doe.add_perf_parameter(name='p1')
doe.add_perf_parameter(name='p2')
sode.add_performance_parameters(performance_params=doe.perf_parameters)

# params = mutation_parameters
sode.mutation_params.mutation_type = de_mutation_type.de_rand_1_bin
sode.mutation_params.F = 0.8
sode.mutation_params.C = 0.7
sode.start_doe(doe.generate_doe())
sode.optimize_from_population(pop_start=-1,n_generations=40)