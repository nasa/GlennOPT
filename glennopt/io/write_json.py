import dataclasses_json
from ..base import Optimizer
from ..DOE import Experiment

def write_json(self,filename:str, doe_object:Experiment, opt_object:Optimizer):
    '''
        Writes the optimization setup to json
    '''
    assert len(doe_object.objectives) >0
    assert len(doe_object.eval_parameters)>0

    object_to_write = dict()
    objectives = list()
    for o in doe_object.objectives:
        objectives.append(o.perf_parameters.to_dict())
        
    eval_params = list()
    for o in doe_object.eval_parameters:
        eval_params.append(o.perf_parameters.to_dict())