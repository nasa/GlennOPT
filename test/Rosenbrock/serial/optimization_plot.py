import sys,os
sys.path.insert(0,'../../../')
from glennopt.base import Parameter
from glennopt.helpers import mutation_parameters, de_mutation_type
from glennopt.optimizers import SODE
from glennopt.helpers import get_best,get_pop_best
from glennopt.DOE import Default,CCD,FullFactorial,LatinHyperCube

# Generate the DOE
current_dir = os.getcwd()
pop_size = 16
sode = SODE(eval_command = "python evaluation.py", eval_folder="Evaluation",pop_size=pop_size,optimization_folder=current_dir)

doe = FullFactorial(levels=8)

doe.add_parameter(name="x1",min_value=-3,max_value=3)
doe.add_parameter(name="x2",min_value=-3,max_value=3)
sode.add_eval_parameters(eval_params=doe.eval_parameters)

doe.add_objectives(name='objective1')
sode.add_objectives(objectives=doe.objectives)

# No performance Parameters
doe.add_perf_parameter(name='p1')
doe.add_perf_parameter(name='p2')
sode.add_performance_parameters(performance_params=doe.perf_parameters)


# Plotting code 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
individuals = sode.read_calculation_folder()
objectives, pop, _ = get_best(individuals,pop_size=20)
objective_index = 0 
_, ax = plt.subplots()    
ax.scatter(pop, objectives[:,objective_index],color='blue',s=10)
ax.set_xlabel('Population')
ax.set_ylabel('Objective {0} Value'.format(objective_index))
plt.show()

#%% Plot Best individual at each population
best_individuals, best_fronts = get_pop_best(individuals)

nobjectives = len(best_individuals[0][0].objectives)
objective_data = list()
for pop,best_individual in best_individuals.items():
    objective_data.append(best_individual[objective_index].objectives[objective_index])

_,ax = plt.subplots()
colors = cm.rainbow(np.linspace(0, 1, len(best_individuals.keys())))
ax.scatter(list(best_individuals.keys()), objective_data, color='blue',s=10)
ax.set_xlabel('Population')
ax.set_ylabel('Objective {0} Value'.format(objective_index))
ax.set_title('Best individual at each population')
plt.show()


#%% Plot the pareto front
best_individuals, best_fronts = get_pop_best(individuals)
objectives, pop, best_fronts = get_best(individuals,pop_size=30)

fig,ax = plt.subplots(figsize=(10,8))

colors = cm.rainbow(np.linspace(0, 1, len(best_fronts)))        
indx = 0
legend_labels = []
# Scan the pandas file, grab objectives for each population
for ind_list in best_fronts:
    obj1_data = []
    obj2_data = []
    c=colors[indx]
    for ind in ind_list[0]:
        obj1_data.append(ind.objectives[0])
        obj2_data.append(ind.objectives[1])
    # Plot the gathered data
    ax.scatter(obj1_data, obj2_data, color=c, s=20,alpha=0.5)
    legend_labels.append(pop[indx])
    indx+=1

ax.set_xlabel('Objective 1')
ax.set_ylabel('Objective 2')
ax.set_title('Non-dimensional sorting: Best Front for each population')
ax.legend(legend_labels)
fig.canvas.draw()
fig.canvas.flush_events()
plt.show()