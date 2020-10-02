def rosenbrock(vector):
    '''
    Rosenbrock function 

    Single objective multiple input function 
    '''
    a = 1
    b = 100
    x = vector[0]
    y= vector[1]
    return ( a-x )**2 +b*( y - x**2 )**2