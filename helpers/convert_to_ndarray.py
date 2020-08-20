import numpy as np

def convert_to_ndarray(t):
    """
        converts a scalar or list to numpy array 
    """

    if type(t) is not np.ndarray and type(t) is not list: # Scalar
        t = np.array([t],dtype=float)
    elif (type(t) is list):
        t = np.array(t,dtype=float)
    return t