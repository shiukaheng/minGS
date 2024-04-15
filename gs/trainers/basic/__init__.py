import random
from typing import List
import torch
from tqdm import tqdm
from gs.core.BaseCamera import BaseCamera
from gs.core.GaussianModel import GaussianModel
from gs.helpers.loss import mix_l1_ssim_loss
from gs.helpers.scene import estimate_scene_scale
from gs.trainers.basic.helpers import densify, get_expon_lr_func, prune, prune_opacity_only, reset_opacities
from gs.visualization.TrainingViewer import TrainingViewer

def train(
        model: GaussianModel, 
        cameras: List[BaseCamera], 
        iterations: int = 5000, 
        randomize: bool = True, 
        positions_lr_init: float = 0.00016,
        positions_lr_final: float = 0.0000016,
        position_lr_delay_mult: float = 0.01,
        position_lr_max_steps: int = 30000,
        rotations_lr: float = 0.001,
        scales_lr: float = 0.005,
        opacities_lr: float = 0.05,
        sh_coefficients_lr: float = 0.0025,
        device: str = "cuda",
        scene_scale: float = None,
        up_sh_interval: int = 1000,
        densify_interval: int = 100,
        densify_from_iter: int = 500,
        densify_until_iter: int = 15000,
        densify_grad_threshold: float = 0.0002,
        opacity_reset_interval: int = 3000,
        opacity_threshold: float = 0.005,
        screen_size_threshold: float = 20,
        world_size_threshold_multiplier: float = 0.1,
        reset_to_opacity: float = 0.01,
    ):
    """
    This is the most basic trainer for Gaussian splatting. It mirrors the original training logic.
    """

    model.to(device)

    # Prepare model visualizer
    viewer = TrainingViewer(model)

    # We estimate the scene size, such that a larger scene will have a larger learning rate. It is a heuristic defined in the original code.
    if scene_scale is None:
        scene_scale = estimate_scene_scale(cameras).item()

    # We set different learning rates for each parameter type in a Gaussian.
    lr_groups = [
        {"params": [model.positions], "lr": positions_lr_init * scene_scale, "name": "positions"},
        {"params": [model.rotations], "lr": rotations_lr, "name": "rotations"},
        {"params": [model.scales], "lr": scales_lr, "name": "scales"},
        {"params": [model.opacities], "lr": opacities_lr, "name": "opacities"},
        {"params": [model.sh_coefficients_0], "lr": sh_coefficients_lr, "name": "sh_coefficients_0"},
        {"params": [model.sh_coefficients_rest], "lr": sh_coefficients_lr / 20.0, "name": "sh_coefficients_rest"},
    ]

    # With all this set, we can define the optimizer.
    optimizer = torch.optim.Adam(lr_groups, lr=0.0, eps=1e-15)

    # We define the learning rate scheduler for the positions parameters, such that initially it is high and decays exponentially.
    position_lr_scheduler = get_expon_lr_func(
        lr_init=positions_lr_init * scene_scale,
        lr_final=positions_lr_final * scene_scale,
        lr_delay_mult=position_lr_delay_mult,
        max_steps=position_lr_max_steps,
    )

    # We define a list of cameras to train on. If randomize is True, we shuffle the cameras. It will be filled whenever it is empty.
    train_cameras: List[BaseCamera] = []

    # We set the active SH degree to 0. For this basic trainer, each Gaussian will just have a constant color.
    active_sh_degree = 0

    pbar = tqdm(range(iterations))
    for i in pbar:

        if (i % up_sh_interval == 0) and (active_sh_degree < model.sh_degree) and (i > 0):
            active_sh_degree += 1

        # If we have no cameras to train on, we fill the list with all cameras.
        if len(train_cameras) == 0:
            train_cameras += reversed(cameras)
            if randomize:
                random.shuffle(train_cameras)

        # We update the learning rate for the positions parameters according to the scheduler.
        for group in optimizer.param_groups:
            if "positions" in group["name"]:
                group["lr"] = position_lr_scheduler(i + 1)
                break

        # We get the next camera to train on.
        camera = train_cameras.pop().to(device)

        # We perform a forward pass and compute the loss.
        rendered = model.forward(camera, active_sh_degree=active_sh_degree)
        loss = mix_l1_ssim_loss(rendered, camera.image)

        # We perform a backward pass and update the parameters.
        loss.backward()
        model.backprop_stats()

        with torch.no_grad():

            # Densification and culling
            if densify_from_iter < i < densify_until_iter:
                if i % densify_interval == 0:
                    densify(model, optimizer, scene_scale, densify_grad_threshold)
                    if i > opacity_reset_interval:
                        prune(model, optimizer, scene_scale, opacity_threshold, screen_size_threshold, world_size_threshold_multiplier)
                    else:
                        prune_opacity_only(model, optimizer, opacity_threshold)

            # Opacity reset
            if (i % opacity_reset_interval == 0) and (i > densify_from_iter):
                reset_opacities(model, optimizer, reset_to_opacity)

            # We perform the optimization step and zero the gradients
            optimizer.step()
            optimizer.zero_grad(set_to_none=True) # We zero the gradients so they do not accumulate to the next iteration.

            viewer.render_once()

            pbar.set_description(f"Loss: {loss.item()}") # We update the progress bar with the current loss.
            torch.cuda.empty_cache() # We empty the cache to avoid memory leaks.

    viewer.finish_training_keep_alive()