import sys,os
sys.path.insert(0,'../../../')
from glennopt.base import Parameter
from glennopt.helpers import mutation_parameters, de_mutation_type
from glennopt.helpers import get_best,get_pop_best,plot_pop_best,plot_best
from glennopt.nsga3 import NSGA3

# Generate the DOE
current_dir = os.getcwd()
ns = NSGA3(eval_script = "Evaluation/evaluation.py", eval_folder="Evaluation",pop_size=20,optimization_folder=current_dir)

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
individuals = ns.read_calculation_folder()
# ns.plot_2D('objective1','objective2')

objectives, pop, best_fronts = get_best(individuals,pop_size=20)
plot_best(objectives,objective_index=0)

best_individuals, best_fronts = get_pop_best(individuals)
print('done')