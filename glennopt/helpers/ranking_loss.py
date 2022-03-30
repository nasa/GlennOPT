from typing import Optional
import torch
from torch.nn.modules import *
from torch import Tensor


def rank_weighted_loss(input: Tensor, target: Tensor, rank:Tensor) -> Tensor:
    """Computes the loss but with bias towards ranking

        2/pi*arctan(5-x)+(1-2/pi*arctan(5)) - Google Search
    Args:
        input (Tensor): vector containing objectives 
        target (Tensor): objective value of best ranked individual. Worst individual has 0 assigned to weight. Best indivdual has weight of 1.
        rank (Tensor): nx1 vector describing the rank. 1 = highest rank, n = lowest rank

    Returns:
        Tensor: weighted loss based on the rank vector 
    """
    
    # Weight function
    intercepts = torch.max(input)
    weights = 2/torch.pi*torch.atan(intercepts-input)+(1-2/torch.pi*torch.atan(intercepts))
    weight_error = torch.sum(weights*(input - target))/torch.sum(weights)

    return weight_error