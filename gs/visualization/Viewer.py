
import torch
import viser
from gs.core.GaussianModel import GaussianModel
from gs.helpers.image import torch_to_numpy
import threading
import time
from gs.visualization.helpers import build_camera

class Viewer:
    def __init__(self, model: GaussianModel, frame_rate=15, width=1920, auto_start=True):
        self.model = model
        self.viser = viser.ViserServer()
        self.running = False
        self.frame_rate = frame_rate
        self.render_thread = threading.Thread(target=self.render_loop, daemon=True)
        self.width = width
        
        @self.viser.on_client_connect
        def on_client_connect(client: viser.ClientHandle):
            print(f"Client {client.client_id} connected")

        if auto_start:
            self.start()

    def start(self):
        """
        Start the rendering loop
        """
        self.running = True
        if not self.render_thread.is_alive():
            self.render_thread.start()

    def stop(self):
        """
        Stop the rendering loop and close the Viser server
        """
        self.running = False
        self.render_thread.join()
        self.viser.stop()

    def render_loop(self):
        """
        Start a thread that renders the model to the clients
        """
        with torch.no_grad():
            target_time = 1.0 / self.frame_rate
            while self.running:
                start_time = time.time()
                self._render_once()
                elapsed_time = time.time() - start_time
                sleep_time = target_time - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _render_once(self):
        """
        Do a single render pass
        """
        clients = self.viser.get_clients()
        renders = {}
        for cid, client in clients.items():
            camera = build_camera(client.camera, width=self.width).to(self.model.positions.device)
            render = torch_to_numpy(self.model.forward(camera).detach().cpu())
            renders[cid] = render
            del camera
        self._send_renders(renders)

    def _send_renders(self, renders):
        """
        Send the renders to the clients
        """
        for cid, client in self.viser.get_clients().items():
            if cid in renders:
                client.set_background_image(renders[cid])