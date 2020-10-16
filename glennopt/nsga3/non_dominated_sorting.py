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
    F = list(filter(None, F))
    return individuals,F
