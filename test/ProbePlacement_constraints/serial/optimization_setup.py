"""
    Simple, non parallel optimization set up example. 
"""
import sys,os
sys.path.insert(0,'../../../')
from glennopt.helpers import Parameter, parallel_settings
from glennopt.sode import SODE
from glennopt.nsga3 import de_mutation_type, mutation_parameters
import numpy as np

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

# Generate the DOE
current_dir = os.getcwd()
pop_size = 16
sode = SODE(eval_script = "Evaluation/evaluation.py", eval_folder="Evaluation",pop_size=pop_size,optimization_folder=current_dir)

sode.add_eval_parameters(eval_params = eval_parameters)

objectives = list()
objectives.append(Parameter(name='objective1'))
sode.add_objectives(objectives=objectives)

# Serial Execution but with a shorter execution timeout. This is needed for numpy.linalg.lstsq timing out or failing
# parallelSettings = parallel_settings()
# parallelSettings.concurrent_executions = 16
# parallelSettings.cores_per_execution: 1
# parallelSettings.execution_timeout = 0.2 # minutes
# sode.parallel_settings = parallelSettings

# params = mutation_parameters
sode.mutation_params.mutation_type = de_mutation_type.de_1_rand_bin
sode.mutation_params.min_parents = int(0.1*pop_size)
sode.mutation_params.max_parents = int(0.9*pop_size)
# sode.start_doe(doe_size=128)
sode.optimize_from_population(pop_start=-1,n_generations=20)



