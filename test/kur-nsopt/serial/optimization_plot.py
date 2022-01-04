#%% Plot Best objective vs population
import sys,os
sys.path.insert(0,'../../../')
from glennopt.base import Parameter
from glennopt.helpers import mutation_parameters, de_mutation_type
from glennopt.helpers import get_best,get_pop_best
from glennopt.optimizers import NSOPT

# Generate the DOE
current_dir = os.getcwd()
ns = NSOPT(eval_command = "python evaluation.py", eval_folder="Evaluation",optimization_folder=current_dir)

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
ns.add_performance_parameters(performance_params=performance_parameters)
# ns.start_doe(doe_size=40)
# ns.optimize_from_population(pop_start=-1,n_generations=10)
individuals = ns.read_calculation_folder()
ns.to_tecplot()
ns.plot_2D('objective1','objective2',[-20,0],[-15,20])
# ns.plot_2D('objective1','objective2')
#%% Plot Best objective vs population
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import numpy as np
# objectives, pop, best_fronts = get_best(individuals,pop_size=20)
# objective_index = 0 
# _, ax = plt.subplots()    
# ax.scatter(pop, objectives[:,objective_index],color='blue',s=10)
# ax.set_xlabel('Population')
# ax.set_ylabel('Objective {0} Value'.format(objective_index))
# plt.show()

#%% Plot Best individual at each population
# best_individuals, best_fronts = get_pop_best(individuals)

# nobjectives = len(best_individuals[0][0].objectives)
# objective_data = list()
# for pop,best_individual in best_individuals.items():
#     objective_data.append(best_individual[objective_index].objectives[objective_index])

# _,ax = plt.subplots()
# colors = cm.rainbow(np.linspace(0, 1, len(best_individuals.keys())))
# ax.scatter(list(best_individuals.keys()), objective_data, color='blue',s=10)
# ax.set_xlabel('Population')
# ax.set_ylabel('Objective {0} Value'.format(objective_index))
# ax.set_title('Best individual at each population')
# plt.show()


# #%% Plot the pareto front
# best_individuals, best_fronts = get_pop_best(individuals)
# objectives, pop, best_fronts = get_best(individuals,pop_size=30)

# fig,ax = plt.subplots(figsize=(10,8))

# colors = cm.rainbow(np.linspace(0, 1, len(best_fronts)))        
# indx = 0
# legend_labels = []
# # Scan the pandas file, grab objectives for each population
# for ind_list in best_fronts:
#     obj1_data = []
#     obj2_data = []
#     c=colors[indx]
#     for ind in ind_list[0]:
#         obj1_data.append(ind.objectives[0])
#         obj2_data.append(ind.objectives[1])
#     # Plot the gathered data
#     ax.scatter(obj1_data, obj2_data, color=c, s=20,alpha=0.5)
#     legend_labels.append(pop[indx])
#     indx+=1

# ax.set_xlabel('Objective 1')
# ax.set_ylabel('Objective 2')
# ax.set_title('Non-dimensional sorting: Best Front for each population')
# ax.legend(legend_labels)
# fig.canvas.draw()
# fig.canvas.flush_events()
# plt.show()




# # %%
