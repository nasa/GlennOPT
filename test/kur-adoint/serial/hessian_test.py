import numpy as np 
import math 
def func(x1,x2,x3):
    """
        function is f = 2x1 ^2 + 5x2^3 + 3e^(x3)

    Args:
        x1 ([type]): [description]
        x2 ([type]): [description]
        x3 ([type]): [description]
    """
    return 2*x1*x1 + 5*math.pow(x2,3) + 3 * math.exp(x3)

def Hessian_actual(x1:float,x2:float,x3:float):
    """This is the actual hessian matrix 

    Args:
        x1 (float): [description]
        x2 (float): [description]
        x3 (float): [description]

    Returns:
        [type]: [description]
    """
    return np.array([[4, 0, 0],
                [0, 30*x2,0],
                [0, 0, 3*math.exp(x3)]
            ])

def hessian(x:np.ndarray):
    """
        Calculate the hessian matrix with finite differences. 

        Example taken from https://stackoverflow.com/a/31207520/1599606 
    Args:
        x (np.ndarray): n,m,k dimensional array. 

    Returns:
        np.ndarray: Returns an array of shape (x.dim, x.ndim) + x.shape where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian


if __name__=="__main__":
    # Lets test this by creating a random distribution of points of x1,x2,x3
    x = np.random.random(size=(4,3,1)) # 4 test cases each with 3 variables (x1,x2,x3), 3rd dimension contains the output
    for i in range(0,4):
        x[i,func(x[i,0,0],x[i,1,0],x[i,2,0])
