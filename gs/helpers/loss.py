#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""
This module contains helper functions for loss computation.
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt, mask=None):
    """
    Regular pixel-wise L1 loss.
    """
    if mask is None:
        return torch.abs((network_output - gt)).mean()
    else:
        return torch.abs((network_output - gt) * mask).mean()

def l2_loss(network_output, gt):
    """
    Regular pixel-wise L2 loss.
    """
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    """
    Generate a 1-dimensional Gaussian kernel.
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """
    Creates a 2D Gaussian window of the specified size and channel.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim_loss(img1, img2, window_size=11, size_average=True, mask=None):
    """
    Calculates the Structural Similarity Index (SSIM) loss between two images.
    """
    channel = img1.size(-3) # Number of channels
    window = create_window(window_size, channel) # Create a 2D Gaussian window

    if img1.is_cuda:
        window = window.cuda(img1.get_device()) # Move the window to the same device as the input images
    window = window.type_as(img1) # Set the window type to the same as the input images

    return _ssim(img1, img2, window, window_size, channel, size_average, mask)

def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        if mask is None:
            return ssim_map.mean()
        else:
            return (ssim_map * mask).mean()
    else:
        if mask is None:
            return ssim_map.mean(1).mean(1).mean(1)
        else:
            return (ssim_map * mask).mean(1).mean(1).mean(1)
        
def mix_l1_ssim_loss(predicted, target, lambda_dssim=0.2):
    """
    A image loss function that combines L1 loss and DSSIM loss.
    L1 loss quantifies pixel-wise difference between the predicted and target images.
    DSSIM loss quantifies structural difference between the predicted and target images, which emulates human perception.
    """
    l1 = l1_loss(predicted, target)
    ssim = ssim_loss(predicted, target)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim)