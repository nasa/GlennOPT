"""
    Simple, non parallel optimization set up example. 
"""
import sys,os
sys.path.insert(0,'../../../')
from glennopt.helpers import Parameter
from glennopt.sode import SODE
from glennopt.nsga3 import de_mutation_type, mutation_parameters

# Generate the DOE
current_dir = os.getcwd()
pop_size = 20
sode = SODE(eval_script = "Evaluation/evaluation.py", eval_folder="Evaluation",pop_size=pop_size,optimization_folder=current_dir)

eval_parameters = []
eval_parameters.append(Parameter(name="x1",min_value=-3,max_value=3))
eval_parameters.append(Parameter(name="x2",min_value=-3,max_value=3))
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
sode.mutation_params.mutation_type = de_mutation_type.de_rand_1_bin
sode.mutation_params.min_parents = 2
sode.mutation_params.max_parents = 5
sode.mutation_params.F = 0.8
sode.mutation_params.C = 0.7
sode.start_doe(doe_size=16) #64
sode.optimize_from_population(pop_start=-1,n_generations=40)



