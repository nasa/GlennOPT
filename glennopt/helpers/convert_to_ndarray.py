import numpy as np

def convert_to_ndarray(t) -> np.ndarray:
    """Converts a scalar or list to a numpy array

    Args:
        t (float,list): [description]

    Returns:
        np.ndarray: variable as an array
    """
    if type(t) is not np.ndarray and type(t) is not list: # Scalar
        t = np.array([t],dtype=float)
    elif (type(t) is list):
        t = np.array(t,dtype=float)
    return t