import numpy as np
from gs.core.BasePointCloud import BasePointCloud


class COLMAPPointCloud(BasePointCloud):
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    point3d_ids: np.ndarray
    def __init__(self, points, colors, normals, point3d_ids):
        super().__init__(points, colors)
        self.normals = normals
        self.point3d_ids = point3d_ids