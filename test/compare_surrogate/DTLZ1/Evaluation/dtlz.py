import math
import numpy as np 

def dtlz1_g(x:np.ndarray, m:int):
    return 100*(np.abs(x[m-1]) + np.sum((x -0.5)**2 - np.cos(20*np.pi*(x-0.5))))

def dtlz1(x:np.ndarray,m:int):
    """
        The Pareto-optimal solution corresponds to xi=0.5 

    Args:
        x (np.ndarray): 1 x nx array of values. Each 0<=x<=1
        m (int): number of objectives 

    Returns:
        _type_: _description_
    """
    f = list()
    n = len(x)-1
    
    for i in range(0, m):
        _f = 0.5 * (1 + dtlz1_g(x[m-1]))
        _f *= np.prod(x[:m-1-i]) # m-1 is same as "m"
        if i > 0:
            _f *= 1 - x[m-1-i]
        f.append(_f)
    return np.array(f)

def dtlz2_g(x:np.ndarray, m:int):
    return np.sum((x[:m-1]-0.5)**2)


def dtlz2(x:np.ndarray,g:function,m:int):
    """
        The Pareto-optimal solutions corresponds to xi=0.5 

    Args:
        x (np.ndarray): 1 x nx array of values. Each 0<=x<=1
        m (int): number of objectives 

    Returns:
        _type_: _description_
    """
    f = list()
    
    for i in range(0, m):
        _f = (1 + dtlz2_g(x[m-1]))
        _f *= np.prod(x[:m-1-i]) # m-1 is same as "m"
        if i > 0:
            _f *= np.sin(x[m-1]*np.pi/2.0)
        f.append(_f)
        
    return f