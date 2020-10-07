from glennopt.nsga3 import NSGA_Individual
import copy
import numpy as np
from numpy import linalg as LA
from typing import TypeVar,List

def non_dominated_sorting(individuals:List[NSGA_Individual]):
    '''
        Loops through the list of individuals and checks which one
    '''
    
    def dominates(x: NSGA_Individual,y : NSGA_Individual) -> bool:
        '''
            Returns true if all the objectives of x are less than y
        '''
        b = np.all(x.objectives <= y.objectives) & np.any(x.objectives<y.objectives)
        return b

    nIndividuals = len(individuals)
    for i in range(nIndividuals):
        individuals[i].dominated_count = 0
        individuals[i].domination_set = []
    
    F = []
    # Start the sortind find the first set
    for i in range(nIndividuals):
        for j in range(i+1,nIndividuals):
            p = copy.deepcopy(individuals[i])
            q = copy.deepcopy(individuals[j])

            if (dominates(p,q)):                    # Calls Dominates 
                p.domination_set.append(j)
                q.dominated_count+=1
            elif (dominates(q,p)):
                q.domination_set.append(i)
                p.dominated_count+=1
            
            individuals[i] = p
            individuals[j] = q
        # After j loop ends 
        if (individuals[i].dominated_count==0):
            F.append(i) # This is saying Individual_1 does not dominate individual i. this vector tells which individual has no dominatant
            individuals[i].rank = 1
    F = [F] 
    k = 0   # Find subsequent fronts
    while len(F[k])!=0:
        Q = []
        for i in F[k]:            # Look at each individual that doesn't have a dominant and checks it against the other individuals that don't have a dominant 
            p = copy.deepcopy(individuals[i])
            for j in p.domination_set:  
                q = copy.deepcopy(individuals[j])
                q.dominated_count = q.dominated_count - 1
                if q.dominated_count == 0:                        
                    Q.append(j) # This tells us which are the best individuals
                    q.rank = k+1
                individuals[j] = q
        F.append(Q) # the second array in F contains the fronts. 
        k+=1
    
    return individuals,F
    # Fnew = list(filter(None, F))
    # P.sort(key=lambda x: x.rank, reverse=False) # Sort the individuals based on rank
    # P = CrowdingDistance(P,Fnew,nObj)

# This is basically the crowding distance
def associate_to_reference_point(individuals:List[NSGA_Individual],zr):
    '''
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
            w = zr[j,:]/LA.norm(zr[j,:])
            z = individuals[i].normalized_cost
            a = np.matmul(np.transpose(w),z)
            a = np.matmul(a,w) # np.transpose(w)*z*w
            d[i,j] = LA.norm(z - a) # compute distance 
        # Get minimum distance
        min_indx = np.argmin(d[i,:])
        dmin = d[i,min_indx]

        individuals[i].associatedRef = min_indx
        individuals[i].distanceToAssociatedRef = dmin
        rho[min_indx] = rho[min_indx] + 1

    return individuals, d, rho