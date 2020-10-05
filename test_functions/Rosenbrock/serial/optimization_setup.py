"""
    Simple, non parallel optimization set up example. 
"""
import sys,os
sys.path.insert(0,'../../../../')
from glennopt.helpers import Parameter, mutation_parameters, de_mutation_type
from glennopt.SODE.SingleObjectiveDE import SingleObjectiveDE 
from glennopt.doe import generate_reference_points

# Generate the DOE
current_dir = os.getcwd()
sode = SingleObjectiveDE(eval_script = "Evaluation/evaluation.py", eval_folder="Evaluation",num_populations=10,pop_size=32,optimization_folder=current_dir)

eval_parameters = []
eval_parameters.append(Parameter(name="x1",min_value=-5,max_value=5))
eval_parameters.append(Parameter(name="x2",min_value=-5,max_value=5))

sode.add_eval_parameters(eval_params = eval_parameters)

objectives = []
objectives.append(Parameter(name='objective1'))
sode.add_objectives(objectives=objectives)

# No performance Parameters
performance_parameters = []
performance_parameters.append(Parameter(name='p1'))
performance_parameters.append(Parameter(name='p2'))
sode.add_performance_parameters(performance_params = performance_parameters)

# params = mutation_parameters
# sode.mutation_params.mutation_type = de_mutation_type.de_1_rand_bin

sode.start_doe(doe_size=128)
sode.optimize_from_population(pop_start=-1,n_generations=20)



