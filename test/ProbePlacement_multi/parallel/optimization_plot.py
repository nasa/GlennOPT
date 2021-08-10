#TODO: Change this to plotting for ns

import sys,os
sys.path.insert(0,'../../../')
from glennopt.base import Parameter
from glennopt.helpers import mutation_parameters, de_mutation_type
from glennopt.sode import SODE, selection_type
from glennopt.nsga3 import NSGA3
from glennopt.DOE import Default,CCD,FullFactorial,LatinHyperCube
import numpy as np
import os


# Initialize the DOE 
doe = LatinHyperCube(samples=128,levels=4) # 128 random samples of the design space
# These are also available for use
# doe = FullFactorial(levels=2) 
# doe = Default(15) # Default
# doe = CCD()


# Define the Evaluation Parameters 
# Define evaluation parameters 
nProbes = 10
minSpacing = 3
probeSpacing = 360/nProbes
tLo     = np.zeros(nProbes)
tHi     = np.zeros(nProbes)
for i in range(nProbes):
    tLo[i] = probeSpacing*i
    if i != nProbes-1:
        tHi[i] = probeSpacing*(i+1) - minSpacing
    else:
        tHi[-1] = probeSpacing*(i+1)    
    doe.add_parameter(name="x"+str(i+1),min_value=tLo[i],max_value=tHi[i])
constraints = (tLo,tHi)

doe = LatinHyperCube(samples=128,levels=5)
doe.add_objectives(name='objective1')
doe.add_objectives(name='objective2')

# Define any performance parameters you want to keep track of (tracking only)
doe.add_perf_parameter(name='PearsonR')
doe.add_perf_parameter(name='RMS_Error')

# Generate the DOE
pop_size=48
current_dir = os.getcwd()
ns = NSGA3(eval_command = "python evaluation.py", eval_folder="Evaluation",pop_size=pop_size,optimization_folder=current_dir)

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

# Get the best individual in each population
best_individuals,best_compromise = ns.get_pop_best()
# Track the best objective as population advances 
best_objective = ns.get_best()
ns.plot_2D('objective1','objective2',[2,20],[2,20])

print('done')