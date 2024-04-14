
from typing import NamedTuple
import numpy as np

class CameraModelTuple(NamedTuple):
    model_id: int
    model_name: str
    num_params: int

class CameraTuple(NamedTuple):
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

class BaseImageTuple(NamedTuple):
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray

class Point3DTuple(NamedTuple):
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2D_idxs: np.ndarray

class ImageTuple(NamedTuple):
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray

CAMERA_MODELS = {
    CameraModelTuple(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModelTuple(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModelTuple(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModelTuple(model_id=3, model_name="RADIAL", num_params=5),
    CameraModelTuple(model_id=4, model_name="OPENCV", num_params=8),
    CameraModelTuple(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModelTuple(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModelTuple(model_id=7, model_name="FOV", num_params=5),
    CameraModelTuple(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModelTuple(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModelTuple(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}

CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])

CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])