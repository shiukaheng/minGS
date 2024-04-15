import numpy as np
import torch
from PIL import Image

"""
This module contains helper functions for image processing.
"""

def pil_to_torch(pil_image):
    image = torch.from_numpy(np.array(pil_image)) / 255.0
    if len(image.shape) == 3: # If the image is colored
        return image.permute(2, 0, 1)
    else: # If the image is grayscale
        return image.unsqueeze(dim=-1).permute(2, 0, 1)

def torch_to_pil(image):
    if len(image.shape) == 3: # If the image is colored
        return Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    else: # If the image is grayscale
        return Image.fromarray((image.squeeze(dim=0).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    
def torch_to_numpy(image: torch.Tensor) -> np.ndarray:
    if len(image.shape) == 3: # If the image is colored
        return image.permute(1, 2, 0).numpy()
    else: # If the image is grayscale
        return image.squeeze(dim=0).permute(1, 2, 0).numpy()