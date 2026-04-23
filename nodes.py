"""TensorRT-RTX RIFE Implementation for ComfyUI.

This module provides custom nodes for video frame interpolation (VFI) using
NVIDIA's TensorRT-RTX. It includes nodes for configuring engine resolutions,
building/loading optimized .trt engines, and performing high-speed inference.
"""

import json
import os
import sys

import folder_paths
import logging
import tensorrt_rtx as trt
import time
import torch

import comfy.model_management as mm

from pathlib import Path
from polygraphy import cuda
from .vfi_utilities import preprocess_frames, postprocess_frames, generate_frames_rife
from .trt_utilities import Engine

def update_folder_names_and_paths(key, targets=[]):
    """Registers specific folder paths within ComfyUI's path manager.

    Ensures that custom subdirectories (like onnx_rife) are recognized by 
    ComfyUI's internal file system and UI loaders.

    Args:
        key (str): The primary key to register in folder_names_and_paths.
        targets (list): Fallback keys to inherit paths from if the primary 
            key is missing.
    """
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".onnx"})
    if base and base != orig:
        logging.warning(f"Unknown file list already present on key {key}: {base}")
        
        
# Auto-detect CUDA toolkit and add DLL path before importing polygraphy
def _setup_cuda_dll_path():
    """Auto-detects the CUDA toolkit and adds cudart64 DLL path on Windows.

    This is necessary for TensorRT-RTX to correctly locate runtime libraries
    when running in a portable environment or on systems where CUDA is not 
    globally in the system PATH.
    """
    if not sys.platform.startswith("win"):
        return
    
    cuda_root = None
    
    # Check for CUDA_PATH or CUDA_HOME environment variables
    cuda_root = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    
    if not cuda_root:
        # Try default Windows install location
        program_files = os.environ.get("PROGRAMFILES")
        if program_files:
            cuda_base = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
            if cuda_base.exists():
                # Find highest version directory
                versions = sorted([d for d in cuda_base.iterdir() if d.is_dir()], reverse=True)
                if versions:
                    cuda_root = str(versions[0])
    
    if cuda_root:
        cuda_path = Path(cuda_root)
        # CUDA 13.0+ puts cudart64 in bin/x64 subdirectory
        cuda_bin_x64 = cuda_path / "bin" / "x64"
        if cuda_bin_x64.exists() and any(cuda_bin_x64.glob("cudart64*.dll")):
            os.add_dll_directory(str(cuda_bin_x64))
            return
        # Fallback to regular bin directory for older CUDA versions
        cuda_bin = cuda_path / "bin"
        if cuda_bin.exists() and any(cuda_bin.glob("cudart64*.dll")):
            os.add_dll_directory(str(cuda_bin))
            return
    
    # CUDA toolkit not found - print warning with download link
    print("[ComfyUI-Rife-TensorRT] WARNING: CUDA toolkit not found.")
    print("    Set CUDA_PATH environment variable or install CUDA toolkit.")
    print("    Download: https://developer.nvidia.com/cuda-13-0-2-download-archive")

# Initialize Environment
_setup_cuda_dll_path()
os.environ["POLYGRAPHY_USE_TENSORRT_RTX"] = "1"
update_folder_names_and_paths("onnx_rife", ["onnx"])

class RIFERTXEngineOptions:
    """Data container for RIFE shape boundaries.

    Attributes:
        min_dim (int): Minimum supported resolution.
        opt_dim (int): Targeted optimal resolution.
        max_dim (int): Maximum supported resolution.
    """
    min_dim: int = 384
    opt_dim: int = 720
    max_dim: int = 1312

class RIFERTXEngineConfig:
    """ComfyUI Node for configuring resolution profiles.

    Provides a UI interface to set the dynamic shape profiles required
    during the TensorRT engine build process.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min_dim": ("INT", {"default": 384, "min": 64, "max": 4096, "step": 8, "tooltip": "Minimum resolution dimension"}),
                "opt_dim": ("INT", {"default": 720, "min": 64, "max": 4096, "step": 8, "tooltip": "Optimal resolution dimension (most common)"}),
                "max_dim": ("INT", {"default": 1312, "min": 64, "max": 4096, "step": 8, "tooltip": "Maximum resolution dimension"}),
            }
        }

    RETURN_TYPES = ("RIFE_RTX_ENGINE_CONFIG",)
    RETURN_NAMES = ("RIFE_ENGINE_CONFIG",)
    FUNCTION = "configure"
    CATEGORY = "TensorRT_RTX/RIFE"
    DESCRIPTION = "Configure custom resolution dimensions for RIFE TensorRT engine."

    def configure(self, min_dim, opt_dim, max_dim):
        """Wraps dimensions into a configuration dictionary.

        Args:
            min_dim (int): Minimum resolution.
            opt_dim (int): Optimal resolution.
            max_dim (int): Maximum resolution.

        Returns:
            tuple: Contains the configuration dictionary.
        """
        config = {"min_d": min_dim, "opt_d": opt_dim, "max_d": max_dim,}
        return (config,)


class RIFERTXEngineLoader:
    """ComfyUI Node for loading or building RIFE TensorRT engines.

    Checks if a .trt engine matching the requested configuration exists. 
    If not, it compiles the ONNX model into a TensorRT engine.
    """
    @classmethod
    def INPUT_TYPES(cls):        
        onnx_names = [x for x in folder_paths.get_filename_list("onnx_rife") if "rife" in x.lower()]        
        return {
            "required": { "onnx_name": (onnx_names,)},
            "optional": { "config": ("RIFE_RTX_ENGINE_CONFIG", {"tooltip": "Options for building the TensorRT engine"}),}
        }

    RETURN_NAMES = ("RIFE_ENGINE",)
    RETURN_TYPES = ("RIFE_RTX_ENGINE",)
    CATEGORY = "TensorRT_RTX/RIFE"
    DESCRIPTION = "Load RIFE tensorrt_rtx models, they will be built automatically if not found."
    FUNCTION = "load_rife_tensorrt_rtx_model"

    def load_rife_tensorrt_rtx_model(self, onnx_name, config=None):
        """Loads or builds the RIFE TensorRT engine.

        Args:
            onnx_name (str): Name of the source ONNX file.
            config (dict, optional): Resolution profile. Defaults to RIFERTXEngineOptions.

        Returns:
            tuple: The loaded Engine object.
        """
        if config is None:
            config = RIFERTXEngineOptions()  
            
        tensorrt_models_dir = os.path.join(folder_paths.models_dir, "tensorrt", "rife")
        onnx_models_dir     = os.path.join(folder_paths.models_dir, "onnx")

        os.makedirs(tensorrt_models_dir, exist_ok=True)
        os.makedirs(onnx_models_dir, exist_ok=True)

        onnx_model_path = os.path.join(onnx_models_dir, f"{onnx_name}")

        # Build tensorrt model path with detailed naming (includes profile)
        engine_channel = 3
        e_min, e_opt, e_max = config.min_d, config.opt_d, config.max_d
        tensorrt_model_path = os.path.join(tensorrt_models_dir, f"{onnx_name}_1x{engine_channel}x{e_min}x{min_}_1x{engine_channel}x{e_opt}x{e_opt}_1x{engine_channel}x{e_max}x{e_max}_{trt.__version__}.trt")

        engine = Engine(tensorrt_model_path)

        if not os.path.exists(tensorrt_model_path):
            logging.info(f"Building TensorRT_RTX engine for {onnx_model_path}: {tensorrt_model_path}")
            
            mm.soft_empty_cache()
            s = time.time()
            
            result = engine.build(
                                  onnx_path=onnx_model_path,
                                  input_profile=[
                                      {
                                          "img0": [(1, engine_channel, e_min, e_min), (1, engine_channel, e_opt, e_opt), (1, engine_channel, e_max, e_max)],
                                          "img1": [(1, engine_channel, e_min, e_min), (1, engine_channel, e_opt, e_opt), (1, engine_channel, e_max, e_max)],
                                      }
                                  ],
            )
            
            if result != 0:
                raise ValueError(f"Failed to build the engine")
            e = time.time()
            logging.info(f"Time taken to build: {(e-s)} seconds")

        logging.info(f"Loading TensorRT_RTX engine: {tensorrt_model_path}")
        mm.soft_empty_cache()
        engine.load()

        return (engine,)

class RIFERTXFrameInterpolation:
    """ComfyUI Node for performing VFI inference.

    Uses the loaded TensorRT engine to interpolate frames in a video sequence.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Input frames for video frame interpolation"}),
                "engine": ("RIFE_RTX_ENGINE", {"tooltip": "Tensorrt_RTX model built and loaded"}),
                "clear_cache_after_n_frames": ("INT", {"default": 100, "min": 1, "max": 1000, "tooltip": "Clear CUDA cache after processing this many frames"}),
                "multiplier": ("INT", {"default": 2, "min": 1, "tooltip": "Frame interpolation multiplier"}),
                "use_cuda_graph": ("BOOLEAN", {"default": True, "tooltip": "Use CUDA graph for better performance"}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": "Keep model loaded in memory after processing"}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "TensorRT_RTX/RIFE"
    OUTPUT_NODE=True

    def vfi(
        self,
        frames,
        engine,
        clear_cache_after_n_frames=100,
        multiplier=2,
        use_cuda_graph=True,
        keep_model_loaded=False,
    ):
        """Interpolates frames using the TensorRT engine.

        Args:
            frames (torch.Tensor): Input image batch.
            engine (Engine): The active TensorRT engine.
            clear_cache_after_n_frames (int): Frequency of CUDA cache clearing.
            multiplier (int): Target frame rate multiplier.
            use_cuda_graph (bool): Whether to use CUDA graph optimization.
            keep_model_loaded (bool): If True, keeps engine in VRAM after use.

        Returns:
            tuple: Interpolated image batch.
        """
        B, H, W, C = frames.shape
        shape_dict = {
            "img0": {"shape": (1, 3, H, W)},
            "img1": {"shape": (1, 3, H, W)},
            "output": {"shape": (1, 3, H, W)},
        }

        cudaStream = cuda.Stream()

        # Activate and allocate buffers for the engine
        engine.activate()
        engine.allocate_buffers(shape_dict=shape_dict)

        frames = preprocess_frames(frames)

        def return_middle_frame(frame_0, frame_1, timestep):
            timestep_t = torch.tensor([timestep], dtype=torch.float32).to(mm.get_torch_device())
            output = engine.infer({"img0": frame_0, "img1": frame_1, "timestep": timestep_t}, cudaStream, use_cuda_graph)
            result = output['output']
            
            return result

        result = generate_frames_rife(frames, clear_cache_after_n_frames, multiplier, return_middle_frame)
        out    = postprocess_frames(result)

        if not keep_model_loaded:
            engine.reset()

        return (out,)


NODE_CLASS_MAPPINGS = {
    "RIFERTXFrameInterpolation": RIFERTXFrameInterpolation,
    "RIFERTXEngineLoader": RIFERTXEngineLoader,
    "RIFERTXEngineConfig": RIFERTXEngineConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RIFERTXFrameInterpolation": "RIFE RTX Frame Interpolation",
    "RIFERTXEngineLoader": "RIFE RTX Engine Loader",
    "RIFERTXEngineConfig": "RIFE RTX Engine Config",
}