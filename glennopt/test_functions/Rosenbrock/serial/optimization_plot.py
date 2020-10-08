#TODO: Change this to plotting for SODE

import sys,os
sys.path.insert(0,'../../../../')
from glennopt.helpers import Parameter
from glennopt.nsga3 import NSGA3,mutation_parameters, de_mutation_type
from glennopt.sode import SODE

# Generate the DOE
current_dir = os.getcwd()
pop_size = 16
sode = SODE(eval_script = "Evaluation/evaluation.py", eval_folder="Evaluation",pop_size=pop_size,optimization_folder=current_dir)

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

# Get the best individual in each population for objective 1
best_individuals, comp_individual = sode.get_best()
sode.plot_best_objective(objective_index=0)