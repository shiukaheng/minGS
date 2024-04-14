import torch

"""
This module contains helper functions for mathematical operations.
"""

def inverse_sigmoid(x):
    return torch.log(x/(1-x))