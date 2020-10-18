# NSGA non-dominated sorting test
# https://github.com/DEAP/deap/issues/266


import sys,os
sys.path.insert(0,'../../../')
from glennopt.helpers import Parameter
from glennopt.nsga3 import mutation_parameters, de_mutation_type, non_dominated_sorting
from glennopt.base_classes import Individual
sys.path.insert(0,'../../../')
from copy import deepcopy

eval_parameters = []
eval_parameters.append(Parameter(name="x1",min_value=-5,max_value=5))
eval_parameters.append(Parameter(name="x2",min_value=-5,max_value=5))
eval_parameters.append(Parameter(name="x3",min_value=-5,max_value=5))

objectives = []
objectives.append(Parameter(name='objective1'))
objectives.append(Parameter(name='objective2'))

obj_vals =  [[0.31,6.10],               #a
            [0.43,6.79],                #b
            [0.22, 7.09],               #c
            [0.59, 7.85],               #d
            [0.66, 3.65],               #e
            [0.83, 4.23],               #f
            [0.21, 5.90],               #g
            [0.79, 3.97],               #h
            [0.51, 6.51],               #i
            [0.27, 6.93],               #j
            [0.58, 4.52],               #k
            [0.24, 8.54]]               #l
names = ['a','b','c','d','e','f','g','h','i','j','k','l']

individuals = list()
for vals,name in zip(obj_vals,names):
    ind = Individual(eval_parameters,objectives,None)
    ind.name = name
    ind.set_objective('objective1',vals[0])
    ind.set_objective('objective2',vals[1])
    individuals.append(deepcopy(ind))
    
    

non_dominated_sorting(individuals,12)
print('done')