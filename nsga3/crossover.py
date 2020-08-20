import numpy as np 

def crossover(x1,x2):
    alpha = np.random.rand(len(x1))
    y1 = alpha*x1+(1-alpha)*x2
    y2 = alpha*x2+(1-alpha)*x1
    return y1,y2