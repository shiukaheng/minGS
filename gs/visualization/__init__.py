import math
import numpy as np
import viser
from gs.core.GaussianModel import GaussianModel
from gs.core.BaseCamera import BaseCamera
from gs.helpers.image import torch_to_numpy
from gs.helpers.transforms import quat_to_rot_numpy
import threading
import time

def build_camera(camera: viser.CameraHandle, width=1920):
    aspect_ratio = camera.aspect # width / height
    height = int(width / aspect_ratio)
    vert_fov = camera.fov # Vertical field of view in radians
    horiz_fov = 2 * math.atan(aspect_ratio * math.tan(vert_fov / 2))
    return BaseCamera(
        height,
        width,
        horiz_fov,
        vert_fov,
        quat_to_rot_numpy(np.array(camera.wxyz)),
        camera.position
    )

class ModelViewer:
    def __init__(self, model: GaussianModel, frame_rate=15, width=1920):
        self.model = model
        self.viser = viser.ViserServer()
        self.running = False
        self.frame_rate = frame_rate
        self.render_thread = threading.Thread(target=self.render_loop, daemon=True)
        self.width = width
        
        @self.viser.on_client_connect
        def on_client_connect(client: viser.ClientHandle):
            print(f"Client {client.client_id} connected")

        self.start()

    def start(self):
        self.running = True
        if not self.render_thread.is_alive():
            self.render_thread.start()

    def stop(self):
        self.running = False
        self.render_thread.join()

    def render_loop(self):
        target_time = 1.0 / self.frame_rate
        while self.running:
            start_time = time.time()
            clients = self.viser.get_clients()
            for cid, client in clients.items():
                camera = build_camera(client.camera, width=self.width)
                render = torch_to_numpy(self.model.forward(camera).detach().cpu())
                client.set_background_image(render)
                del camera
            elapsed_time = time.time() - start_time
            sleep_time = target_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)