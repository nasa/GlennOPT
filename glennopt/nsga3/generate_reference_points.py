from __future__ import absolute_import
import copy
import numpy as np
from ..helpers.convert_to_ndarray import convert_to_ndarray

# http://doi.org/10.1109/TEVC.2013.2281535

'''
    Generate reference points creates a matrix that sums up to 1. The rows represent different combinations of the summation. Columns represents the number of divisions
'''

class reference_point(list): # inherits list
    def __init__(self,*args):
        list.__init__(self,*args)
        self.associations_count = 0
        self.associations = [] 

def generate_reference_points(num_individuals, num_divisions_per_obj=4):
    def gen_refs_recursive(work_point, num_objs:int, left, total, depth):
        if (depth == num_individuals-1):
            work_point[depth] = left/total
            ref = reference_point(copy.deepcopy(work_point))
            return [ref]
        else:
            res = []
            for i in range(0,left+1):
                work_point[depth] = i/total
                res = res + gen_refs_recursive(work_point, num_individuals, left-i, total, depth+1)
            return res
    ret = convert_to_ndarray(gen_refs_recursive([0]*num_individuals, num_individuals, num_divisions_per_obj, num_divisions_per_obj, 0))
    return ret 