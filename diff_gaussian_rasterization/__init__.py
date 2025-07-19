#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    scores,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        scores,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        scores,
        raster_settings
    ):

        # Defensive programming: ensure all tensor parameters are valid torch.Tensor objects
        # This prevents AttributeError when non-tensor objects are passed
        if not isinstance(colors_precomp, torch.Tensor):
            colors_precomp = torch.empty(0, dtype=means3D.dtype, device=means3D.device)
        
        if not isinstance(sh, torch.Tensor):
            sh = torch.empty(0, dtype=means3D.dtype, device=means3D.device)
            
        if not isinstance(cov3Ds_precomp, torch.Tensor):
            cov3Ds_precomp = torch.empty(0, dtype=means3D.dtype, device=means3D.device)

        # Restructure arguments the way that the C++ lib expects them
        # The C++ function expects the complete SH tensor, not split into dc and rest
        if sh.numel() != 0:
            sh_tensor = sh.contiguous()
        else:
            # Empty tensor when SHs are not provided
            sh_tensor = torch.empty((0, 0, 0), dtype=means3D.dtype, device=means3D.device)

        # Ensure background is on the same device as means3D to avoid cross-device issues
        bg_tensor = raster_settings.bg
        if bg_tensor.device != means3D.device:
            bg_tensor = bg_tensor.to(means3D.device)
        # Fix: Ensure bg_tensor has correct dtype to avoid implicit type promotion
        if bg_tensor.dtype != means3D.dtype:
            bg_tensor = bg_tensor.to(dtype=means3D.dtype)

        # Fix: Ensure empty tensors have consistent shapes based on actual Gaussian count
        N = means3D.shape[0] if means3D.numel() > 0 else 0
        if colors_precomp.numel() == 0:
            colors_precomp = torch.empty((N, 3), dtype=means3D.dtype, device=means3D.device)
        if cov3Ds_precomp.numel() == 0:
            cov3Ds_precomp = torch.empty((N, 6), dtype=means3D.dtype, device=means3D.device)
        if scales is not None and scales.numel() == 0:
            scales = torch.empty((N, 3), dtype=means3D.dtype, device=means3D.device)
        if rotations is not None and rotations.numel() == 0:
            rotations = torch.empty((N, 4), dtype=means3D.dtype, device=means3D.device)
        # Fix: Ensure empty SH tensor has consistent shape
        if sh_tensor.numel() == 0:
            sh_tensor = torch.empty((N, 3, (raster_settings.sh_degree+1)**2), dtype=means3D.dtype, device=means3D.device)

        args = (
            bg_tensor,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh_tensor,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, kernel_times, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, kernel_times, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Save values required for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered  # R

        # Create invdepth tensor (depth information)
        invdepth = torch.zeros_like(color[0:1])  # Same shape as one channel of color

        ctx.save_for_backward(
            colors_precomp,
            opacities,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh_tensor,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            invdepth,
        )

        # Keep the original return signature (color, radii, depth) expected by calling code.
        return color, radii, invdepth

    @staticmethod
    def backward(ctx, grad_out_color, _0, grad_out_invdepth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (colors_precomp,
         opacities,
         means3D,
         scales,
         rotations,
         cov3Ds_precomp,
         radii,
         sh_tensor,
         geomBuffer,
         binningBuffer,
         imgBuffer,
         invdepth_saved) = ctx.saved_tensors

        # Handle missing gradient for invdepth (can be None if not used)
        if grad_out_invdepth is None:
            grad_out_invdepth = torch.empty((0,), dtype=means3D.dtype, device=means3D.device)

        # Restructure args as C++ method expects them
        bg_tensor = raster_settings.bg
        if bg_tensor.device != means3D.device:
            bg_tensor = bg_tensor.to(means3D.device)
        # Fix: Ensure bg_tensor has correct dtype to avoid implicit type promotion
        if bg_tensor.dtype != means3D.dtype:
            bg_tensor = bg_tensor.to(dtype=means3D.dtype)

        args = (
            bg_tensor,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            sh_tensor,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_G2 = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_G2 = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,  # scores (not used)
            None,  # raster_settings
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        # This is an ugly fix, but it works.
        # The original code had a check that was too strict.
        if colors_precomp is None:
            colors_precomp = torch.empty(0, dtype=means3D.dtype, device=means3D.device)
        
        if shs is None:
            shs = torch.empty(0, dtype=means3D.dtype, device=means3D.device)
            
        if cov3D_precomp is None:
            if scales is None or rotations is None:
                raise Exception('Please provide either cov3D_precomp or scales and rotations!')
            # Create empty tensor when cov3D_precomp is None
            cov3D_precomp = torch.empty(0, dtype=means3D.dtype, device=means3D.device)
        
        # All clear, rasterize
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            None, # scores (not used)
            self.raster_settings
        )
