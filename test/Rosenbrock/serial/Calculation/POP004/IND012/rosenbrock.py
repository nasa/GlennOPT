def rosenbrock(vector):
    '''
    Rosenbrock function 

    Single objective multiple input function 
    '''
    a = 1
    b = 100
    x = vector[0]
    y= vector[1]
    return b*(( a-x )**2 +( y - x**2 )**2)