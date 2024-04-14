import numpy as np
import torch
from gs.core.GaussianModel import GaussianModel
from torch import nn
from gs.helpers.math import inverse_sigmoid
from gs.helpers.transforms import build_rotation

"""
This module contains helper functions for densifying and pruning Gaussian models. It is not attached as a class method since it required direct access to the optimizer.
"""

def densify(model: GaussianModel, optimizer: torch.optim.Adam, scene_scale: float, gradient_threshold: float, percent_dense: float = 0.01) -> None:
    gradients = model.mean_gradient_magnitude
    exceed_gradient_mask = torch.where(gradients > gradient_threshold, True, False).squeeze(1)
    large_gaussian_mask = (torch.max(model.scaling_activation(model.scales), dim=1).values > percent_dense * scene_scale)
    clone_mask = torch.logical_and(exceed_gradient_mask, ~large_gaussian_mask)
    clone_gaussians(model, optimizer, clone_mask)
    split_mask = torch.logical_and(exceed_gradient_mask, large_gaussian_mask)
    padded_split_mask = pad_mask(split_mask, model, model.positions.device)
    split_gaussians(model, optimizer, padded_split_mask)

def prune(model: GaussianModel, optimizer: torch.optim.Adam, scene_scale: float, opacity_threshold: float, screen_size_threshold: float, world_size_threshold_multiplier: float = 0.1) -> None:
    opacity_mask = model.opacity_activation(model.opacities) < opacity_threshold
    screen_size_mask = model.max_radii2D > screen_size_threshold
    world_size_mask = model.scaling_activation(model.scales).max(dim=1).values > world_size_threshold_multiplier * scene_scale
    final_mask = opacity_mask.squeeze(1).logical_or_(screen_size_mask.squeeze(1).logical_or_(world_size_mask))
    cull_gaussians(model, optimizer, final_mask)

def prune_opacity_only(model: GaussianModel, optimizer: torch.optim.Adam, opacity_threshold: float) -> None:
    opacity_mask = model.opacity_activation(model.opacities) < opacity_threshold
    cull_gaussians(model, optimizer, opacity_mask.squeeze())

def append_new_gaussians(
        model: GaussianModel, 
        optimizer: torch.optim.Adam, 
        positions: torch.Tensor, 
        rotations: torch.Tensor, 
        scales: torch.Tensor, 
        opacities: torch.Tensor, 
        sh_coefficients_0: torch.Tensor,
        sh_coefficients_rest: torch.Tensor
    ) -> None:
    device = model.positions.device
    extension_lookup = {
        "positions": positions,
        "rotations": rotations,
        "scales": scales,
        "opacities": opacities,
        "sh_coefficients_0": sh_coefficients_0,
        "sh_coefficients_rest": sh_coefficients_rest
    }
    for group in optimizer.param_groups:
        if len(group["params"]) != 1:
            raise ValueError("Unexpected number of parameters in optimizer group. Only one parameter is expected, as initialized in the GaussianModel.")
        if group["name"] not in extension_lookup:
            raise ValueError(f"Unexpected parameter name {group['name']} in optimizer group. Expected one of 'positions', 'rotations', 'scales', 'opacities', 'sh_coefficients_0', 'sh_coefficients_rest'.")
        extension_params = extension_lookup[group["name"]]
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_params)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_params)), dim=0)
            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_params), dim=0).requires_grad_())
            optimizer.state[group["params"][0]] = stored_state
            setattr(model, group["name"], group["params"][0])
        else:
            group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_params), dim=0).requires_grad_())
            setattr(model, group["name"], group["params"][0])
    model._gradient_accumulator = torch.zeros((model.positions.shape[0], 1), device=device)
    model._gradient_accumulator_denominator = torch.zeros((model.positions.shape[0], 1), device=device)
    model.max_radii2D = torch.zeros((model.positions.shape[0]), device=device).unsqueeze(1)

def clone_gaussians(
        model: GaussianModel, 
        optimizer: torch.optim.Adam, 
        mask: torch.Tensor
    ) -> None:
    positions = model.positions[mask]
    rotations = model.rotations[mask]
    scales = model.scales[mask]
    opacities = model.opacities[mask]
    sh_coefficients_0 = model.sh_coefficients_0[mask]
    sh_coefficients_rest = model.sh_coefficients_rest[mask]
    append_new_gaussians(model, optimizer, positions, rotations, scales, opacities, sh_coefficients_0, sh_coefficients_rest)
    
def cull_gaussians(
        model: GaussianModel,
        optimizer: torch.optim.Adam,
        mask: torch.Tensor
    ) -> None:
    keep_mask = ~mask
    for group in optimizer.param_groups:
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][keep_mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][keep_mask]
            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(group["params"][0][keep_mask].requires_grad_())
            optimizer.state[group["params"][0]] = stored_state
            setattr(model, group["name"], group["params"][0])
        else:
            group["params"][0] = nn.Parameter(group["params"][0][keep_mask].requires_grad_())
            setattr(model, group["name"], group["params"][0])
    model._gradient_accumulator = model._gradient_accumulator[keep_mask]
    model._gradient_accumulator_denominator = model._gradient_accumulator_denominator[keep_mask]
    model.max_radii2D = model.max_radii2D[keep_mask]

def split_gaussians(
        model: GaussianModel,
        optimizer: torch.optim.Adam,
        mask: torch.Tensor,
        n_samples: int = 2,
    ) -> None:
    device = model.positions.device
    positions = model.positions[mask]
    rotations = model.rotations[mask]
    scales = model.scales[mask]
    opacities = model.opacities[mask]
    sh_coefficients_0 = model.sh_coefficients_0[mask]
    sh_coefficients_rest = model.sh_coefficients_rest[mask]
    
    sds = model.scaling_activation(scales).repeat(n_samples, 1)
    means = torch.zeros((sds.size(0), 3), device=device)
    samples = torch.normal(means, sds)
    p_rotations = build_rotation(rotations).repeat(n_samples, 1, 1)

    new_positions = torch.bmm(p_rotations, samples.unsqueeze(-1)).squeeze(-1) + positions.repeat(n_samples, 1)
    new_rotations = rotations.repeat(n_samples, 1)
    new_scales = model.scaling_inverse_activation(
        model.scaling_activation(scales).repeat(n_samples, 1) / (0.8 * n_samples)
    )
    new_opacities = opacities.repeat(n_samples, 1)
    new_sh_coefficients_0 = sh_coefficients_0.repeat(n_samples, 1, 1)
    new_sh_coefficients_rest = sh_coefficients_rest.repeat(n_samples, 1, 1)

    append_new_gaussians(model, optimizer, new_positions, new_rotations, new_scales, new_opacities, new_sh_coefficients_0, new_sh_coefficients_rest)

    padded_mask = pad_mask(mask, model, device)
    cull_gaussians(model, optimizer, padded_mask)

def pad_mask(mask: torch.Tensor, model: GaussianModel, device: torch.device) -> torch.Tensor:
    mask_length = mask.size(0)
    model_length = model.positions.size(0)
    n_new_gaussians = model_length - mask_length
    return torch.cat((mask, torch.zeros(n_new_gaussians, dtype=torch.bool, device=device)))

def replace_tensor_to_optimizer(optimizer: torch.optim.Adam, tensor: torch.Tensor, name: str) -> dict:
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if group["name"] == name:
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors

def reset_opacities(model: GaussianModel, optimizer: torch.optim.Adam, opacity: float = 0.01) -> None:
    new_opacities = model.opacity_inverse_activation(
        torch.min(
            model.opacity_activation(model.opacities), 
            torch.ones_like(model.opacity_activation(model.opacities)) * opacity)
    )
    if torch.isnan(new_opacities).any():
        raise ValueError("NaNs in new opacities.")
    params = replace_tensor_to_optimizer(optimizer, new_opacities, "opacities")
    model.opacities = params["opacities"]

def get_expon_lr_func(
    lr_init: float, lr_final: float, lr_delay_steps: int = 0, 
    lr_delay_mult: float = 1.0, max_steps: int = 1000000
) -> callable:
    def helper(step: int) -> float:
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper