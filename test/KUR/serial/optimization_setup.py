"""
    Simple, non parallel optimization set up example. 
"""
import sys,os
sys.path.insert(0,'../../../')
from glennopt.helpers import Parameter
from glennopt.nsga3 import NSGA3,mutation_parameters, de_mutation_type

# Generate the DOE
pop_size=32
current_dir = os.getcwd()
ns = NSGA3(eval_script = "Evaluation/evaluation.py", eval_folder="Evaluation",pop_size=pop_size,optimization_folder=current_dir)

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

ns.mutation_params.mutation_type = de_mutation_type.de_rand_1_bin
ns.mutation_params.F = 0.8
ns.mutation_params.C = 0.7
ns.start_doe(doe_size=64)
ns.optimize_from_population(pop_start=-1,n_generations=40)