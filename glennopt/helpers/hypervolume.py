import numpy as np 
from .nsga_functions import dominates

def extract_pareto(objs:np.ndarray) -> np.ndarray:
    """Extracts a pareto front

    Args:
        objs (np.ndarray): shape = n_individuals, n_objectives

    Returns:
        np.ndarray: _description_
    """
    pf = list()
    for i in range(objs.shape[1]):
        dominated = False
        for j in range(objs.shape[1]):
            if dominates(objs[:,j],objs[:,i]):
                dominated = True
                break

        if not dominated:
            pf.append(objs[:,i]) # Append the individual
    pf = np.array(pf).transpose()
    return pf

def worst_design(objs:np.array) -> np.array:
    worst_designs = list()
    for i in range(objs.shape[1]):
        for j in range(objs.shape[1]):
            if dominates(objs[:,j],objs[:,i]): # index j dominates i 
                worst_designs.append(objs[:,i]) # if something constantly gets dominated then it's not a good design 
                break
    if len(worst_designs)>1:
        worst_design(np.array(worst_designs))
    return worst_designs[-1] # Pick the one that gets dominated the most? should check this 
                

def hypervolume(objectives:np.ndarray,ref:np.ndarray):
    """Code converted into python from https://github.com/Alaya-in-Matrix/Hypervolume-HSO 
        All credits for this function should go to the author

    Args:
        objectives (np.ndarray): (objectives, points) 
    """
    num_obj = objectives.shape[0]
    num_pnt = objectives.shape[1]
    if num_obj == 1:
        return ref[0] - objectives[0, 0]
    else:        
        indx = np.argsort(objectives[0,:]) # sorted based on first objective
        sorted_individuals = objectives[:,indx]         
        # I dont think this matters which objective you use if you are simply computing a hypervolume. 
        # Volume takes into account all objectives
        prev_point = ref
        hv = 0 
        for i in range(num_pnt-1, -1, -1):
            print(f'sorted individuals: \n {sorted_individuals}')
            curr_point = sorted_individuals[:,i]
            width = prev_point[0] - curr_point[0]
            tmp_pnts = extract_pareto(sorted_individuals[-(num_obj-1):,:i+1])
            print(f'sorted filtered: \n {sorted_individuals[-(num_obj-1):,:i+1]}')

            sub_hv = hypervolume(tmp_pnts, ref[:num_obj - 1])
            hv += width * sub_hv
            prev_point = curr_point
        return hv

