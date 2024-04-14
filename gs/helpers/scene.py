from typing import List
import torch
from gs.core.BaseCamera import BaseCamera
from gs.helpers.transforms import get_camera_center


def estimate_scene_scale(cameras: List[BaseCamera], inflation_factor: float = 1.1):
    """
    Calculate the scene radius based on the camera positions. Metric will be used to scale the learning rate.
    Larger scene sizes lead to larger step sizes in optimization.
    """
    positions = [get_camera_center(camera) for camera in cameras]
    positions = torch.vstack(positions) # (N, 3)
    center = torch.mean(positions, dim=0) # (3,)
    distances_to_center = torch.norm(positions - center, dim=1) # (N,)
    return torch.max(distances_to_center) * inflation_factor