import torch
import numpy as np 

def gradient(y:torch.Tensor, x:torch.Tensor, grad_outputs=None):
        """Compute dy/dx @ grad_outputs"""
        if grad_outputs is None:
            grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True)[0]
        return grad

def jacobian(y:np.ndarray, x:np.ndarray) -> np.array:
    """Compute dy/dx = dy/dx @ grad_outputs
        for grad_outputs in [1, 0, ..., 0], [0, 1, 0, ..., 0], ...., [0, ..., 0, 1]

        Source: https://discuss.pytorch.org/t/how-to-compute-jacobian-matrix-in-pytorch/14968/12 

    Args:
        y (np.ndarray): Outputs
        x (np.ndarray): Inputs

    Returns:
        np.ndarray: Matrix [number of outputs, number of eval parameters]
    """
        
    y = torch.as_tensor(y)
    x = torch.as_tensor(x) 

    jac = torch.zeros(y.shape[0], x.shape[0]) 
    for i in range(y.shape[0]):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1
        jac[i] = gradient(y, x, grad_outputs = grad_outputs)
    return jac.cpu().detach().numpy()