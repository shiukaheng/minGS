import math
import numpy as np
import viser
from gs.core.BaseCamera import BaseCamera

def quaternion_to_rotation_matrix(q):
    """Convert a quaternion into a rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def convert_viser_to_colmap(position, quaternion):
    """Convert Viser camera position and quaternion to COLMAP R and t."""
    R = quaternion_to_rotation_matrix(quaternion)
    T = np.array(position)
    # Camera center in COLMAP format
    t = -np.dot(R.T, T)
    return R, t

def build_camera(camera: viser.CameraHandle, width=1920):
    aspect_ratio = camera.aspect  # width / height
    height = int(width / aspect_ratio)
    vert_fov = camera.fov  # Vertical field of view in radians
    horiz_fov = 2 * math.atan(aspect_ratio * math.tan(vert_fov / 2))

    R, t = convert_viser_to_colmap(camera.position, camera.wxyz)

    return BaseCamera(
        height,
        width,
        horiz_fov,
        vert_fov,
        R,
        t
    )