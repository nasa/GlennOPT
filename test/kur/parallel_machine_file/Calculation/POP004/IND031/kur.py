import math

def KUR(x1,x2,x3):
    '''
        Kursawe Function
        mutiple output
    '''
    f1 = (-10*math.exp(-0.2*math.sqrt(x1*x1+x2*x2))) + (-10*math.exp(-0.2*math.sqrt(x2*x2+x3*x3)))

    f2 = (math.pow(abs(x1),0.8)+5*math.sin(x1*x1*x1))+(math.pow(abs(x2),0.8)+5*math.sin(x2*x2*x2))+(math.pow(abs(x3),0.8)+5*math.sin(x3*x3*x3))
    # Performance Parameter
    p1 = x1 + x2 + x3
    p2 = x1*x2*x3
    p3 = x1 - x2 - x3
    return f1,f2,p1,p2,p3