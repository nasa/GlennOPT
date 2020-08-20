import numpy as np 
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

def KUR(x1,x2,x3):
    '''
        Kursawe Function
        mutiple output
    '''
    f1 = (-10*math.exp(-0.2*math.sqrt(x1*x1+x2*x2))) + (-10*math.exp(-0.2*math.sqrt(x2*x2+x3*x3)))

    f2 = (abs(x1)^0.8+5*math.sin(x1*x1*x1))+(math.pow(abs(x2),0.8)+5*math.sin(x2*x2*x2))+(math.pow(abs(x3),0.8)+5*math.sin(x3*x3*x3))

    return f1,f2