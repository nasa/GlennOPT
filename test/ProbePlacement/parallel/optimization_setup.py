"""
    Simple, non parallel optimization set up example. 
"""
import sys,os
sys.path.insert(0,'../../../')
from glennopt.base import Parameter
from glennopt.helpers import mutation_parameters, de_mutation_type
from glennopt.optimizers import SODE
from glennopt.DOE import Default,CCD,FullFactorial,LatinHyperCube
import numpy as np
import os

# Initialize the DOE - Latin hyper cube
doe = LatinHyperCube(128) # 128 random samples of the design space
# These are also available for use
# doe = FullFactorial(levels=8) 
# doe = Default(15) # Default
# doe = CCD()

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

# Define the number of objectives
doe.add_objectives(name='objective1')

# Define any performance parameters you want to keep track of (tracking only)
doe.add_perf_parameter(name='PearsonR')
doe.add_perf_parameter(name='RMS_Error')

# Set up the optimizer
current_dir = os.getcwd()
pop_size = 96
sode = SODE(eval_command = "python evaluation.py", eval_folder="Evaluation",pop_size=pop_size,optimization_folder=current_dir)

# Include load the objectives, performance parameters, evaluation parameters into sode
sode.add_eval_parameters(doe.eval_parameters)
sode.add_objectives(doe.objectives)
sode.add_performance_parameters(doe.perf_parameters)

# Parallel Settings (You don't need to run this block if you only want serial execution)
sode.parallel_settings.concurrent_executions = 8    # Change to 1 for serial
sode.parallel_settings.cores_per_execution = 1    
sode.parallel_settings.execution_timeout = 0.2      # minutes


# Start the optimizer
# sode.mutation_params.mutation_type = de_mutation_type.de_rand_1_bin
sode.mutation_params.mutation_type = de_mutation_type.de_dmp
sode.mutation_params.F = 0.6
sode.mutation_params.C = 0.7
sode.start_doe(doe.generate_doe())
sode.optimize_from_population(pop_start=-1,n_generations=100)