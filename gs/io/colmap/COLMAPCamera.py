from typing import List
import torch
from gs.core.BaseCamera import BaseCamera


class COLMAPCamera(BaseCamera):
    def __init__(
            self,
            image_height: int,
            image_width: int,
            fov_x: float,
            fov_y: float,
            R: torch.Tensor,
            t: torch.Tensor,
            image: torch.Tensor,
            image_path: str,
            uid: int,
            point3d_ids: List[int],
    ):
        super().__init__(image_height, image_width, fov_x, fov_y, R, t, image)
        self.image_path = image_path
        self.uid = uid
        self.point3d_ids = point3d_ids