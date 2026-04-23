"""Utility functions for RIFE Video Frame Interpolation (VFI).

This module contains preprocessing, postprocessing, and the core interpolation
loop logic used to generate intermediate frames between a sequence of images.

https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/blob/main/vfi_utils.py
"""

import einops
import os
import torch
import typing

import numpy as np

from comfy.model_management import soft_empty_cache, get_torch_device
from comfy.utils import ProgressBar

DEVICE = get_torch_device()

def preprocess_frames(frames: torch.Tensor) -> torch.Tensor:
    """Prepares ComfyUI image batches for model inference.

    Converts images from (Batch, Height, Width, Channels) to 
    (Batch, Channels, Height, Width) and ensures only RGB channels are present.

    Args:
        frames (torch.Tensor): Input tensor in BHWC format.

    Returns:
        torch.Tensor: Preprocessed tensor in BCHW format.
    """
    return einops.rearrange(frames[..., :3], "n h w c -> n c h w")

def postprocess_frames(frames: torch.Tensor) -> torch.Tensor:
    """Converts model output back to ComfyUI standard format.

    Converts images from (Batch, Channels, Height, Width) back to 
    (Batch, Height, Width, Channels) and moves them to CPU memory.

    Args:
        frames (torch.Tensor): Output tensor from the model in BCHW format.

    Returns:
        torch.Tensor: Postprocessed tensor in BHWC format on CPU.
    """
    return einops.rearrange(frames, "n c h w -> n h w c")[..., :3].cpu()

def generate_frames_rife(
        frames,
        clear_cache_after_n_frames,
        multiplier,
        return_middle_frame_function
        ):
    """Core interpolation loop for generating intermediate frames.

    Iterates through a sequence of frames and calls the provided inference 
    function to generate 'n' intermediate frames based on the multiplier.

    Args:
        frames (torch.Tensor): The preprocessed frame sequence (BCHW).
        clear_cache_after_n_frames (int): Interval for VRAM cache clearing.
        multiplier (int): The factor by which to increase the frame count.
        return_middle_frame_function (Callable): Function that takes two 
            frames and a timestep and returns the interpolated frame.

    Returns:
        torch.Tensor: The complete sequence including original and 
            interpolated frames.
    """
    output_frames = torch.zeros(multiplier*frames.shape[0], *frames.shape[1:], device="cpu")
    out_len = 0
    cache_counter = 0
    
    pbar = ProgressBar(len(frames))

    for frame_itr in range(len(frames) - 1): 

        frame_0 = frames[frame_itr:frame_itr+1]
        frame_1 = frames[frame_itr+1:frame_itr+2]
        
        # Store the current original frame
        output_frames[out_len] = frame_0 
        out_len += 1

        # Generate intermediate frames
        for middle_i in range(1, multiplier):
            timestep = middle_i/multiplier
            middle_frame = return_middle_frame_function(frame_0, frame_1, timestep).detach().cpu()

            # Copy middle frames to output
            output_frames[out_len] = middle_frame
            out_len +=1

            # VRAM Management
            cache_counter += 1
            if cache_counter >= clear_cache_after_n_frames:
                soft_empty_cache()
                cache_counter = 0

            pbar.update(1)

    # Append final frame
    output_frames[out_len] = frames[-1:]
    logger(f"done! - {(len(frames) -1) * (multiplier-1)} new frames generated at resolution: {output_frames[0].shape}")
    out_len += 1

    # clear cache for courtesy
    soft_empty_cache()
    
    # Slice to actual length in case of rounding or logic offsets
    return output_frames[:out_len]
