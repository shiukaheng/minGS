
from gs.core.GaussianModel import GaussianModel
import threading
from gs.helpers.image import torch_to_numpy
from gs.visualization.Viewer import Viewer
import threading

from gs.visualization.helpers import build_camera

class TrainingViewer(Viewer):
    def __init__(self, model: GaussianModel, width=1920, finish_frame_rate=15):
        self.training_preview_thread = None
        self.lock = threading.Lock()
        super().__init__(model, frame_rate=finish_frame_rate, width=width, auto_start=False)

    def render_once(self):
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
        thread = threading.Thread(target=self._send_renders, args=(renders,))
        thread.start()

    def finish_training_keep_alive(self):
        """
        Keep a render loop running after training is finished.
        """
        self.start()