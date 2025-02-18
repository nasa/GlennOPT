from __future__ import absolute_import
from importlib import import_module     

from .nsga3 import NSGA3
from .sode import SODE
from .nsopt import NSOPT

if import_module('torch') is not None:
    from .nsga3_ml import NSGA3_ML
else:
    print("torch is not installed, skilling NSGA3_ML")
