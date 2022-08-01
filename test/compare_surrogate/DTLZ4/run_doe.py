"""
    Simple, non parallel optimization set up example. 
"""
import sys,os
sys.path.insert(0,'../../../')
from glennopt.base import Parameter 
from glennopt.helpers import mutation_parameters, de_mutation_type
from glennopt.optimizers import Adjoint
from glennopt.DOE import Default,CCD,FullFactorial,LatinHyperCube

# Generate the DOE
pop_size=16
current_dir = os.getcwd()
adjoint = Adjoint(eval_command = "python evaluation.py", eval_folder="Evaluation",optimization_folder=current_dir,pareto_resolution=64)

# doe = Default(15) # Default
# doe = CCD()
# doe = FullFactorial(levels=8)
doe = LatinHyperCube(128)

doe.add_parameter(name="x1",min_value=-5,max_value=5)
doe.add_parameter(name="x2",min_value=-5,max_value=5)
doe.add_parameter(name="x3",min_value=-5,max_value=5)
adjoint.add_eval_parameters(eval_params=doe.eval_parameters)

doe.add_objectives(name='objective1')
doe.add_objectives(name='objective2')
adjoint.add_objectives(objectives=doe.objectives)

# No performance Parameters
doe.add_perf_parameter(name='p1')
doe.add_perf_parameter(name='p2')
doe.add_perf_parameter(name='p2')
adjoint.add_performance_parameters(performance_params= doe.perf_parameters)

adjoint.start_doe(doe.generate_doe())
# ns.optimize_from_population(pop_start=-1,n_generations=50)

