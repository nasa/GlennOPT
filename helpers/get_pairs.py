import numpy as np 
    
def get_pairs(nIndividuals:int,nParents:int,parent_indx_seed=[]):
    """
        Get a list of all the pairing partners for a particular individual
        Inputs:
            nIndividuals - number of individuals
            nParents - number of parents 
            parent_indx_seed - pre-populate the parent index array
    """
    if len(parent_indx_seed)>0:
        parent_indx = parent_indx_seed
        nParents += len(parent_indx_seed)
    else:
        parent_indx = []
    rand_indx = -1        
    while(rand_indx in parent_indx or rand_indx==-1):
        rand_indx = np.random.randint(0,nIndividuals-1)
        parent_indx.append(rand_indx)
        if (len(parent_indx)>=nParents):
            break
    return parent_indx