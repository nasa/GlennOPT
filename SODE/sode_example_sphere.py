import matplotlib.pyplot as plt
import numpy as np
## Problem Definition
def Sphere(x):
    return np.sum(np.power(x,2))

CostFunction = lambda a: Sphere(a) # Cost Function

nVar=20            # Number of Decision Variables 20

VarSize=[1, nVar]   # Decision Variables Matrix Size

VarMin=-5          # Lower Bound of Decision Variables
VarMax= 5          # Upper Bound of Decision Variables

## DE Parameters

MaxIt=1000      # Maximum Number of Iterations

nPop=50        # Population Size 50

beta_min=0.2   # Lower Bound of Scaling Factor
beta_max=0.8   # Upper Bound of Scaling Factor

pCR=0.2        # Crossover Probability

## Initialization
class Individual:
    def __init__(self, position,cost):
        self.Position = position
        self.Cost = cost

# empty_individual = Individual(_,_)
# empty_individual.Position=[]
# empty_individual.Cost=[]

BestSol = Individual([],[])
BestSol.Cost=np.inf

NewSol = Individual([],[])

# pop=np.tile(empty_individual,nPop) ####CANNOT USE ######-- creates a pointer to the same"empty_individual"
pop =[]
for i in range(nPop):
    pop.append(Individual([],[]))
    
for i in range(nPop):

    pop[i].Position= np.random.uniform(VarMin,VarMax,VarSize) ##### Is this doing the right thing?

    # Sets the position of ith individual to a list of randomly distributed numbers b/w VarMin and VarMax

    pop[i].Cost= CostFunction(pop[i].Position)  
        
    if pop[i].Cost < BestSol.Cost:        

        BestSol.Position=pop[i].Position
        BestSol.Cost = pop[i].Cost     

BestCost=np.zeros((MaxIt,1)) #Initialize a best cost array

## DE Main Loop
for it in range(MaxIt):
    
    for i in range(nPop):
        x=np.copy(pop[i].Position)
        A=np.random.permutation(nPop).tolist() 
        
        if i in A: A.remove(i) # replaces A(A==i)=[]
        
        a=A[0]
        b=A[1]
        c=A[2]
        
        # Mutation
        #beta=unifrnd(beta_min,beta_max)
        beta = np.random.uniform(beta_min,beta_max,VarSize)
        y= pop[a].Position + (beta *(np.subtract(pop[b].Position, pop[c].Position))) #See single objective DE
        y = np.maximum(y, VarMin)
        y = np.minimum(y, VarMax)

        
        # Crossover
        numel = 1
        for dim in np.shape(x): numel *= dim #matches numel behavior
        z=np.zeros(numel)
        j0=np.random.randint(1, numel)
        for j in range(numel):
            if j==j0 or np.random.random()<=pCR:
                z[j]=y[0][j]
            else:
                z[j]=x[0][j]

        NewSol.Position=[z.tolist()]
        NewSol.Cost=CostFunction(NewSol.Position)

        if NewSol.Cost<pop[i].Cost:

            pop[i].Position=NewSol.Position
            pop[i].Cost=NewSol.Cost
        
            if pop[i].Cost<BestSol.Cost:
                BestSol.Position=pop[i].Position
                BestSol.Cost = pop[i].Cost

    # Update Best Cost
    BestCost[it]=BestSol.Cost
    
    # Show Iteration Information
#     print('Iteration ', it, ': Best Cost = ', BestCost[it])

#plot(BestCost)
plt.semilogy(BestCost)
plt.title("Single Objective DE")
plt.xlabel('Iteration')
plt.ylabel('Best Cost')
plt.grid
plt.show()