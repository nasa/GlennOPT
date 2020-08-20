import math

def rosenbrock(x,alpha=100):
    '''
        Rosenbrock function 

        Single objective multiple input function 
    '''
    f = 0
    for indx in range(len(x)-1):
        temp = alpha* math.pow(x[indx+1]-x[indx]*x[indx],2) + math.pow(1.0-x[indx],2)
        f+=temp
    return f