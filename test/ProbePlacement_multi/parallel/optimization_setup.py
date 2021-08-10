"""
    Simple, non parallel optimization set up example. 
"""
import sys,os
sys.path.insert(0,'../../../')
from glennopt.base import Parameter
from glennopt.helpers import mutation_parameters, de_mutation_type
from glennopt.optimizers import NSGA3
from glennopt.DOE import Default,CCD,FullFactorial,LatinHyperCube
import numpy as np
import os

# Initialize the DOE 
doe = LatinHyperCube(samples=128,levels=4) # 128 random samples of the design space
# These are also available for use
# doe = FullFactorial(levels=2) 
# doe = Default(15) # Default
# doe = CCD()

eval_parameters = list()

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

doe.add_objectives(name='objective1')
doe.add_objectives(name='objective2')

# Define any performance parameters you want to keep track of (tracking only)
doe.add_perf_parameter(name='PearsonR')
doe.add_perf_parameter(name='RMS_Error')

# Set up the optimizer
current_dir = os.getcwd()
pop_size = 48
ns = NSGA3(eval_command = "python evaluation.py", eval_folder="Evaluation",pop_size=pop_size,optimization_folder=current_dir)
ns.add_eval_parameters(eval_params=doe.eval_parameters)
ns.add_objectives(objectives=doe.objectives)
ns.add_performance_parameters(performance_params= doe.perf_parameters)

# Parallel Settings (You don't need to run this block if you only want serial execution)
ns.parallel_settings.concurrent_executions = 8    # Change to 1 for serial
ns.parallel_settings.cores_per_execution= 1    
ns.parallel_settings.execution_timeout = 0.2      # minutes

# Start the optimizer
ns.mutation_params.mutation_type = de_mutation_type.de_rand_1_bin
ns.mutation_params.F = 0.6
ns.mutation_params.C = 0.7
# Start the Design of Experiments
ns.start_doe(doe.generate_doe())