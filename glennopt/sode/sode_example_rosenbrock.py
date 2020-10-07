import matplotlib.pyplot as plt
import numpy as np

# def Sphere(x):
#     return np.sum(np.power(x,2))
# CostFunction = lambda a: Sphere(a) #Cost fn

def rosenbrock(vector):
    a = 1
    b = 100
    x = vector[0]
    y= vector[1]
    return ( a-x )**2 +b*( y - x**2 )**2
CostFunction = lambda a: rosenbrock(a) #Cost fn
print(rosenbrock([1,1]))
# nvar =2  #Descion Variables

# varSize = [1,nvar]   #Descion Variables matrix size

# varMin = -5     # Lower Bound of Decision Variables

# varMax = 5      # Upper Bound of Decision Variables

# ##DE Parameters

# maxit = 1000
# npop = 50

# beta_min = 0.2 #Lower bound for SCALING FACTOR
# beta_max = 0.8 #upper "  "   "   "  "   "   "  
# pcr = 0.2   #Probability of crossover

# class Individual:
#     def __init__(self, position, cost):
#         self.position = position
#         self.post = cost

# BestSol = Individual([],[])
# BestSol.cost = np.inf

# NewSol = Individual([],[])

# pop =[]
# for i in range(npop):
#     pop.append(Individual([],[]))
# #Define your initial population:
# for i in range(npop):
#     pop[i].position = np.random.uniform(varMin,varMax,varSize)[0]
#     # Sets the position of ith individual to a list of randomly distributed numbers b/w VarMin and VarMax
#     pop[i].cost = CostFunction(pop[i].position)
#     current_position = pop[i].position
#     current_cost = pop[i].cost
#     if current_cost < BestSol.cost:
        
#         BestSol.position = current_position
#         BestSol.cost = current_cost

# best_costs = np.zeros((maxit,1)) #to  track best cost
# best_posits = np.zeros((maxit,2))
# #Perform Differential Evolution

# for it in range(maxit):

#     for i in range(npop):
#         individuals_position = np.copy(pop[i].position) #x = individuals_position
#         A = np.random.permutation(npop).tolist()  # **
#         # print(np.random.permutation(npop).tolist())

#         a = A[0]
#         b = A[1]
#         c = A[2]
#         #***
#         # a,b, and c are some random int between 0 and npop - 1 

#         #Mutation

#         beta = np.random.uniform(beta_min,beta_max,varSize)[0] #Selecting amplification factor
#         #So you gotta permute the population via indexes provided using ** & *** and then mutate

#         y = pop[a].position + beta * np.subtract(  pop[b].position  ,  pop[c].position  )
#         y = np.maximum(y, varMin)
#         y = np.minimum(y, varMax)
#         #The above 2 lines just replace ELEMENTS (with varmin, varmax) in the position vector not within varmin, varmax

        
#         #Crossover
#         numel = 1
#         for dimension in np.shape(individuals_position):    ##### Important but subtle line
#             numel *= dimension 
#         z = np.zeros(numel)

#         j0 = np.random.randint(1,numel) # this is a random int between 1 and the 
#                                         # number of elements in your "position" variable
#                                         #allows for the variability of dimension in "position"
#                                         # so it can be some multi dimensional design var
#         for j in range(numel):

#             if j==j0 or np.random.random()<=pcr:
#                 z[j] = y[j] #changes coordinate z_j to y_j

#             else:
#                 z[j] = individuals_position[j]
        
#         NewSol.position = z.tolist()
#         NewSol.cost = CostFunction(NewSol.position)
#         #Keep the mutated individual if it has a lower cost than the original individual
#         if NewSol.cost < pop[i].cost:
#             pop[i].position = NewSol.position
#             pop[i].cost = NewSol.cost
            
#             if pop[i].cost < BestSol.cost:
#                 BestSol.position = pop[i].position 
#                 BestSol.cost = pop[i].cost
#     best_costs[it] = BestSol.cost 
#     best_posits[it] = BestSol.position
#     # print('Iteration ', it, ': Best Cost = ', BestCost[it])

# #plot(BestCost)
# plt.semilogy(best_costs)
# plt.title("Single Objective DE")
# plt.xlabel('Iteration')
# plt.ylabel('Best Cost')
# plt.grid
# plt.show()
# print(BestSol.position)
# # plot(BestPosition)
# plt.plot(best_posits[:,0])
# plt.title("Single Objective DE positions")
# plt.xlabel('Iteration')
# plt.ylabel('Best Position')
# plt.grid
# plt.show()
# print(np.shape(best_posits))
