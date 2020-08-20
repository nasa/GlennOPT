from glennopt.nsga3 import NSGA_Individual
import numpy as np
from numpy import linalg as LA
from typing import TypeVar,List
T = TypeVar('T', bound='NSGA_Individual')
individual_list = List[T]

# This is basically the crowding distance
def associate_to_reference_point(individuals:individual_list,zr:np.ndarray):
    '''
        Inputs:
            individuals - list of NSGA Individuals 
            
            zr - this is the matrix that adds up to 1, rows = number of partitions, columns = number of objectives. 
            
            Paht: (2 objectives) I think "zr" represents a pareto front. The first and last rows represent the extreme ends the middle is the compromise between the two objectives

        returns
            individuals - List of individuals with parameters associatedref and distanceToAssociatedRef populated

            d - distance
            rho - number of individuals near reference point? # TODO Not sure about this one

    '''
    
    nZr = len(zr) # Number of reference points
    rho = np.zeros((nZr,))
    nPop = len(individuals)
    d = np.zeros((nPop,nZr))

    for i in range(nPop):
        for j in range(nZr):
            w = zr[j,:]/LA.norm(zr[j,:]) # * Normalize zr and store as w. In the case of 2 objectives [0 1] or [0.25 0.75]
            w = w.reshape((len(w),1)) # because sometimes we get a vector
            z = individuals[i].normalized_cost
            z = z.reshape((len(z),1)) # because sometimes we get a vector
            a = np.matmul(w.transpose(),z)     
            a = a*w # np.transpose(w)*z*w
            d[i,j] = LA.norm(z - a) # compute distance  d(i,j) = norm(z - w'*z*w);
            # * Paht - I think this is the distance to the pareto front.

        # Get minimum distance
        min_indx = np.argmin(d[i,:]) # minimum distance to any point on the pareto front, whatever it's closest. 
        dmin = d[i,min_indx]

        individuals[i].association_ref = min_indx  # * Paht: This is which index on the pareto front the individual is closest to 
        individuals[i].distance_to_association_ref = dmin # * Paht: this is the minimum distance 
        rho[min_indx] = rho[min_indx] + 1  # * Paht: This is the number of individuals closest to each point along the pareto front. So if rho has 5 rows, that's 5 p

    return individuals, d, rho