import math
import numpy as np
import torch
from typing import TYPE_CHECKING 
if TYPE_CHECKING:
    from gs.core.BaseCamera import BaseCamera

"""
This module contains helper functions for 3D transformations.
"""

def quat_to_rot(r: torch.Tensor) -> torch.Tensor:
    """
    Builds a rotation matrix from a quaternion.

    Args:
        r (torch.Tensor): A tensor of shape (N, 4) representing the quaternion.

    Returns:
        torch.Tensor: A tensor of shape (N, 3, 3) representing the rotation matrix.
    """
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def quat_to_rot_numpy(quat: np.ndarray) -> np.ndarray:
    """
    Builds a rotation matrix from a quaternion.

    Args:
        quat (np.ndarray): A numpy array of shape (4,) representing the quaternion (w, x, y, z).

    Returns:
        np.ndarray: A 3x3 numpy array representing the rotation matrix.
    """
    if quat.shape != (4,):
        raise ValueError("Quaternion must be an array of shape (4,)")

    # Normalize the quaternion
    norm = np.sqrt(np.dot(quat, quat))
    if norm == 0:
        raise ValueError("The input quaternion is a zero vector and cannot be normalized.")
    quat = quat / norm

    w, x, y, z = quat

    # Compute the rotation matrix components
    rxx = 1 - 2 * (y*y + z*z)
    rxy = 2 * (x*y - w*z)
    rxz = 2 * (x*z + w*y)
    ryx = 2 * (x*y + w*z)
    ryy = 1 - 2 * (x*x + z*z)
    ryz = 2 * (y*z - w*x)
    rzx = 2 * (x*z - w*y)
    rzy = 2 * (y*z + w*x)
    rzz = 1 - 2 * (x*x + y*y)

    return np.array([
        [rxx, rxy, rxz],
        [ryx, ryy, ryz],
        [rzx, rzy, rzz]
    ])

def build_scaling_rotation(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Builds a scaling-rotation matrix.

    Args:
        s (torch.Tensor): Scaling factors of shape (N, 3).
        r (torch.Tensor): Rotation angles of shape (N, 3).

    Returns:
        torch.Tensor: Scaling-rotation matrix of shape (N, 3, 3).
    """
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = quat_to_rot(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def strip_lowerdiag(L: torch.Tensor) -> torch.Tensor:
    """
    Strips the lower diagonal elements from a 3x3 matrix.

    Args:
        L (torch.Tensor): Input matrix of shape (N, 3, 3), where N is the number of matrices.

    Returns:
        torch.Tensor: A tensor of shape (N, 6) containing the stripped lower diagonal elements.
    """
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def build_covariance_from_scaling_rotation(scaling: torch.Tensor, scaling_modifier: float, rotation: torch.Tensor) -> torch.Tensor:
    """
    Build a covariance matrix from scaling and rotation.

    Args:
        scaling (torch.Tensor): Scaling factor.
        scaling_modifier (float): Scaling modifier.
        rotation (torch.Tensor): Rotation matrix.

    Returns:
        torch.Tensor: Symmetric covariance matrix.
    """
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_lowerdiag(actual_covariance) # Previously strip_symmetric
    return symm

def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion vector to a rotation matrix.

    Parameters:
    qvec (numpy.ndarray): The quaternion vector to be converted.

    Returns:
    numpy.ndarray: The resulting rotation matrix.

    """
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to a quaternion vector.

    Parameters:
    R (numpy.ndarray): The rotation matrix.

    Returns:
    numpy.ndarray: The quaternion vector.

    """
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def focal_to_fov(focal: float, pixels: int) -> float:
    """
    Convert focal length to field of view.
    """
    return 2*math.atan(pixels/(2*focal))

def get_world_to_view(R: np.ndarray, t: np.ndarray, translate=np.array([.0, .0, .0]), scale=1.0):
    """
    Get world to view transformation matrix.
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def get_camera_center(camera: 'BaseCamera'):
    """
    Get camera center in world coordinates.
    """
    R, t = camera.R, camera.t
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    return torch.tensor(cam_center)

def get_projection_matrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P