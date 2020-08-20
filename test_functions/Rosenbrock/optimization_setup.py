import sys,os
sys.path.insert(0,'../../../')
from glennopt.helpers import Parameter
from glennopt.nsga3 import NSGA3
from glennopt.doe import generate_reference_points



# Generate the DOE
current_dir = os.getcwd()
ns = NSGA3(crossover_percent=0.5,mutational_percent=0.5,mutation_rate=0.02, eval_script = "Evaluation/evaluation.py", eval_folder="Evaluation",num_populations=10,pop_size=20,optimization_folder=current_dir)

eval_parameters = []
eval_parameters.append(Parameter(name="x1",min_value=-10,max_value=10))
eval_parameters.append(Parameter(name="x2",min_value=-10,max_value=10))
ns.add_eval_parameters(eval_params = eval_parameters)

objectives = []
objectives.append(Parameter(name='objective1'))

ns.add_objectives(objectives=objectives)

# No performance Parameters

# ns.start_doe(doe_size=20)
ns.optimize_from_population(pop_start=-1,n_generations=10)



