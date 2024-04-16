# minGS
This is a minimalistic refactor of the original 3D Gaussian splatting codebase that follows PyTorch conventions and allow for easy customization and extension, based on the [original 3DGS official repository](https://github.com/graphdeco-inria/gaussian-splatting).

It is meant for researchers who want to experiment with 3D Gaussian splatting and need a clean and easy to understand codebase to start from.

# Features
- 🧑🏻‍💻 Typed and commented
- 📦 .devcontainer Docker provided
- 📄 Separation of model and training logic
    - A `nn.module` `GaussianModel` only for storing parameters and forward pass (rendering)
    - Reference training logic and hyperparameters is defined in `train()` in `gs.trainers.basic`
- 📸 [Viser](https://github.com/nerfstudio-project/viser) web-based frontend for viewing model during and after training

# Examples

## Minimal Custom Training Example
To customize the pipeline `GaussianModel` can be used just like any other PyTorch model and the training loop can be written from scratch. Below is a minimal example:
```python
import torch
from gs.core.GaussianModel import GaussianModel
from gs.helpers.loss import l1_loss
from gs.io.colmap import load

cameras, pointcloud = load('your_dataset/')
model = GaussianModel.from_point_cloud(pointcloud).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-15)

for i in range(5000):

    camera = cameras[i % len(cameras)]
    rendered = model.forward(camera)

    loss = l1_loss(rendered, camera.image)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True) 

    torch.cuda.empty_cache()
```

# Structure
The codebase is structured as follows:
- `gs/`: The Gaussian splatting module.
    - `core/`: Core data structures and functions for rendering 3DGS models
        - `BaseCamera.py`: Base class that represents a camera used for training 3DGS models
        - `BasePointCloud.py`: Base class for point clouds used for initializing 3DGS models
        - `GaussianModel.py`: 3DGS model refactored as a nn.Module. Use `forward` with a camera to render the model
    - `io/`: Functions for importing and exporting image and point cloud data
        - `colmap/`: Functions for importing COLMAP reconstructions into `BaseCamera` and `BasePointCloud` compliant objects
    - `trainers/`: Training scripts for 3DGS models
        - `basic/`: Re-implementations of the original training script
    - `visualization/`: Classes for visualizing 3DGS models
        - `Viewer.py`: Class for starting a web-based 3DGS viewer for a `GaussianModel`
        - `TrainingViewer.py` Extension of `Viewer` to be integrated into a training loop for live viewing during training 
    - `helpers/`: General functions for rendering and training 3DGS models
 
# Installing dependencies
Only tested for Linux, but may work for Windows too. Using devcontainers should make getting the dependencies easier.
## Method 1: Local environment (Tested on Python 3.8)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- `pip install numpy scipy torch torchvision torchaudio plyfile lpips pybind11 viser`
- Installing PyBind11 submodules
    - `pip install -e ./submodules/diff-gaussian-rasterization/`
    - `pip install -e ./submodules/simple-knn/`
## Method 2: .devcontainer
Devcontainers automatically recreate the development environment using Docker. It is mainly supported by VSCode but there is [also limited support for other editors](https://containers.dev/supporting).
Install the relevant extensions, and when you open the repository you should be prompted to enter the container environment. First time running might take around 5 minutes to build the environment.
- VSCode
- [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
