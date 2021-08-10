import sys,os
sys.path.insert(0,'../../../')
from glennopt.helpers import Parameter
from glennopt.optimizers import NSGA3

# Generate the DOE
current_dir = os.getcwd()
ns = NSGA3(eval_command = "python evaluation.py", eval_folder="Evaluation",pop_size=32,optimization_folder=current_dir)

eval_parameters = []
eval_parameters.append(Parameter(name="x1",min_value=-10,max_value=10))
eval_parameters.append(Parameter(name="x2",min_value=-10,max_value=10))
eval_parameters.append(Parameter(name="x3",min_value=-10,max_value=10))
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
# ns.start_doe(doe_size=40)
# ns.optimize_from_population(pop_start=-1,n_generations=10)
ns.read_calculation_folder()
ns.plot_2D('objective1','objective2')

