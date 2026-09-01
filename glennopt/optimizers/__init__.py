from __future__ import absolute_import
from importlib.util import find_spec

from .nsga3 import NSGA3
from .sode import SODE

if find_spec('torch') is not None:
    from .nsopt import NSOPT
    from .nsga3_ml import NSGA3_ML
else:
    print("torch is not installed, skipping NSOPT and NSGA3_ML. Install with: pip install glennopt[ml]")
