#TODO: Change this to plotting for SODE

import sys,os
sys.path.insert(0,'../../../')
from glennopt.helpers import Parameter, mutation_parameters, de_mutation_type
from glennopt.optimizers import SODE
import numpy as np

# Generate the DOE
current_dir = os.getcwd()
pop_size = 16
sode = SODE(eval_command = "python evaluation.py", eval_folder="Evaluation",pop_size=pop_size,optimization_folder=current_dir)

eval_parameters = list()
# Define evaluation parameters 
dim = 9 # Dimension of X
probeSpacing = 360/dim
tLo     = np.zeros(dim)
tHi     = np.zeros(dim)
minSpacing = 3
for i in range(dim):
    tLo[i] = probeSpacing*i
    if i != dim-1:
        tHi[i] = probeSpacing*(i+1) + minSpacing
    else:
        tHi[-1] = probeSpacing*(i+1) - minSpacing
    eval_parameters.append(Parameter(name="x"+str(i+1),min_value=tLo[i],max_value=tHi[i]))
sode.add_eval_parameters(eval_params = eval_parameters)


objectives = []
objectives.append(Parameter(name='objective1'))
sode.add_objectives(objectives=objectives)

sode.create_restart()
# Get the best individual in each population
best_individuals,best_compromise = sode.get_pop_best()
# Track the best objective as population advances 
best_objective = sode.get_best()
sode.plot_best_objective(objective_index=0)
sode.plot_best_pop(objective_index=0)
print('done')