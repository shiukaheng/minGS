
from gs.core.GaussianModel import GaussianModel
import threading
from gs.visualization.Viewer import Viewer
import threading

class TrainingViewer(Viewer):
    def __init__(self, model: GaussianModel, width=1920, finish_frame_rate=15):
        self.training_preview_thread = None
        self.lock = threading.Lock()
        super().__init__(model, frame_rate=finish_frame_rate, width=width, auto_start=False)

    def render_once(self):
        """
        Do preview renders during the training loop off-thread to avoid blocking.
        """
        if self.training_preview_thread is None or not self.training_preview_thread.is_alive():
            self.training_preview_thread = threading.Thread(target=self._render_once)
            self.training_preview_thread.start()

    def finish_training_keep_alive(self):
        """
        Keep a render loop running after training is finished.
        """
        self.start()