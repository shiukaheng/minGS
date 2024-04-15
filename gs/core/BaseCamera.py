from typing import Union
import numpy as np
import torch
import torch.nn as nn
from gs.helpers.transforms import get_projection_matrix, get_world_to_view

ZFAR = 100.0
ZNEAR = 0.01

class BaseCamera(nn.Module):
    """
    Base class for a camera.
    """

    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor
    image: Union[torch.Tensor, None]

    def __init__(
            self,
            image_height: int,
            image_width: int,
            fov_x: float,
            fov_y: float,
            R: np.ndarray,
            t: np.ndarray,
            image: torch.Tensor = None
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.R = R
        self.t = t

        world_view_transform = torch.tensor(get_world_to_view(self.R, self.t)).transpose(0, 1)
        projection_matrix = get_projection_matrix(znear=ZNEAR, zfar=ZFAR, fovX=self.fov_x, fovY=self.fov_y).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        self.register_buffer("world_view_transform", world_view_transform)
        self.register_buffer("full_proj_transform", full_proj_transform)
        self.register_buffer("camera_center", camera_center)
        self.register_buffer("image", image)