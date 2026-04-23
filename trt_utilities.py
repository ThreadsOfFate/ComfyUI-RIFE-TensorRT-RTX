"""TensorRT Acceleration Utilities for RIFE.

This module provides the low-level interface for NVIDIA TensorRT, handling
engine compilation, timing caches, memory allocation, and optimized inference 
using CUDA Graphs.
"""

import copy
import io
import os

import cuda.bindings.runtime as cudart
import onnx
import numpy as np
import tensorrt_rtx as trt
import torch

from collections import OrderedDict
from logging import error, warning
from onnxconverter_common import float16
from onnxsim import simplify

# Use polygraphy TensorRTX backend
os.environ["POLYGRAPHY_USE_TENSORRT_RTX"] = "1"

from polygraphy.backend.common import bytes_from_path
from polygraphy import util
from polygraphy.backend.trt import ModifyNetworkOutputs, Profile
from polygraphy.backend.trt import (
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    network_from_onnx_bytes,
    save_engine,
)
from polygraphy.logger import G_LOGGER

from torch.cuda import nvtx
from tqdm import tqdm

G_LOGGER.module_severity = G_LOGGER.ERROR

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}

# Path for your persistent hardware profile
node_dir = os.path.dirname(os.path.realpath(__file__))
MASTER_CACHE_FILE = f"{node_dir}\\RIFE_RTX_ENGINE.cache"

def get_timing_cache(config):
    """Retrieves or creates a TensorRT timing cache.

    Args:
        config (trt.IBuilderConfig): The builder configuration object.

    Returns:
        trt.ITimingCache: The loaded or new timing cache.
    """
    if os.path.exists(MASTER_CACHE_FILE):
        with open(MASTER_CACHE_FILE, "rb") as f:
            cache_data = f.read()
            return config.create_timing_cache(cache_data)
    return config.create_timing_cache(b"")

def save_timing_cache(timing_cache):
    """Saves the current timing cache to disk for future builds.

    Args:
        timing_cache (trt.ITimingCache): The cache to serialize.
    """
    with open(MASTER_CACHE_FILE, "wb") as f:
        f.write(timing_cache.serialize())
 
def CUASSERT(cuda_ret):
    """Asserts that a CUDA call was successful.
    https://github.com/Jeff-LiangF/streamv2v/blob/18c1a3bd56ff348d54a3300605936980bb13b03c/src/streamv2v/acceleration/tensorrt/utilities.py
    
    Args:
        cuda_ret (tuple): The return value from a cuda-python call.

    Returns:
        Any: The result of the CUDA call if successful.

    Raises:
        RuntimeError: If the CUDA call returned an error.
    """
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None

class TQDMProgressMonitor(trt.IProgressMonitor):
    """Progress monitor for TensorRT engine builds using TQDM bars."""
    
    def __init__(self):
        trt.IProgressMonitor.__init__(self)
        self._active_phases = {}
        self._step_result = True
        self.max_indent = 5

    def phase_start(self, phase_name, parent_phase, num_steps):
        """Called when a build phase starts."""
        leave = False
        try:
            if parent_phase is not None:
                nbIndents = (
                    self._active_phases.get(parent_phase, {}).get(
                        "nbIndents", self.max_indent
                    )
                    + 1
                )
                if nbIndents >= self.max_indent:
                    return
            else:
                nbIndents = 0
                leave = True
            self._active_phases[phase_name] = {
                "tq": tqdm(
                    total=num_steps, desc=phase_name, leave=leave, position=nbIndents
                ),
                "nbIndents": nbIndents,
                "parent_phase": parent_phase,
            }
        except KeyboardInterrupt:
            # The phase_start callback cannot directly cancel the build, so request the cancellation from within step_complete.
            _step_result = False

    def phase_finish(self, phase_name):
        """Called when a build phase finishes."""
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    self._active_phases[phase_name]["tq"].total
                    - self._active_phases[phase_name]["tq"].n
                )

                parent_phase = self._active_phases[phase_name].get("parent_phase", None)
                while parent_phase is not None:
                    self._active_phases[parent_phase]["tq"].refresh()
                    parent_phase = self._active_phases[parent_phase].get(
                        "parent_phase", None
                    )
                if (
                    self._active_phases[phase_name]["parent_phase"]
                    in self._active_phases.keys()
                ):
                    self._active_phases[
                        self._active_phases[phase_name]["parent_phase"]
                    ]["tq"].refresh()
                del self._active_phases[phase_name]
            pass
        except KeyboardInterrupt:
            _step_result = False

    def step_complete(self, phase_name, step):
        """Called when a step within a phase completes."""
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    step - self._active_phases[phase_name]["tq"].n
                )
            return self._step_result
        except KeyboardInterrupt:
            # There is no need to propagate this exception to TensorRT. We can simply cancel the build.
            return False

class Engine:
    """Wrapper for TensorRT engine management.

    Handles loading, building, and executing inference with optional 
    CUDA Graph support.

    Attributes:
        engine_path (str): File path to the .trt engine.
    """
    
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph
        self.graph = None

    def __del__(self):
        """Ensures CUDA resources are freed on deletion."""
        if hasattr(self, 'cuda_graph_instance') and self.cuda_graph_instance is not None:
            try:
                cudart.cudaGraphDestroy(self.cuda_graph_instance)
            except:
                pass
        if hasattr(self, 'graph') and self.graph is not None:
            try:
                cudart.cudaGraphDestroy(self.graph)
            except:
                pass

        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def reset(self, engine_path=None):
        """Cleans up engine and CUDA resources.

        Args:
            engine_path (str, optional): New path to set for the engine.
        """
        if hasattr(self, 'cuda_graph_instance') and self.cuda_graph_instance is not None:
            try:
                cudart.cudaGraphDestroy(self.cuda_graph_instance)
            except:
                pass
            self.cuda_graph_instance = None
        if hasattr(self, 'graph') and self.graph is not None:
            try:
                cudart.cudaGraphDestroy(self.graph)
            except:
                pass
            self.graph = None

        if hasattr(self, 'engine') and self.engine is not None:
            del self.engine
        if hasattr(self, 'context') and self.context is not None:
            del self.context
        if hasattr(self, 'buffers'):
            del self.buffers
        if hasattr(self, 'tensors'):
            del self.tensors

        self.engine = None
        self.context = None
        self.engine_path = engine_path if engine_path else self.engine_path

        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.inputs = {}
        self.outputs = {}

    def build(
        self,
        onnx_path,
        input_profile=None,
        enable_refit=False,
        enable_preview=False,
        enable_all_tactics=False,
        timing_cache=None,
        update_output_names=None,
    ):
        """Compiles an ONNX model into a TensorRT engine.

        Args:
            onnx_path (str): Path to source ONNX file.
            input_profile (list): List of dicts defining min/opt/max shapes.
            **kwargs: Configuration flags (enable_refit, etc).

        Returns:
            int: 0 if success, 1 if failure.
        """
        p = [Profile()]
        if input_profile:
            p = [Profile() for i in range(len(input_profile))]
            for _p, i_profile in zip(p, input_profile):
                for name, dims in i_profile.items():
                    assert len(dims) == 3
                    _p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []  

        # Load ONNX file 
        onnx_bytes       = onnx.load(onnx_path)
        onnx_precision   = onnx_bytes.graph.input[0].type.tensor_type.elem_type        
        onnx_simp, check = simplify(onnx_bytes)
        
        if not check:
            error(f"Simplified ONNX model: {onnx_path} could not be validated")
            return 1
        
        # Keep simplified model in memory to not overwrite the onnx file in models directory
        # We dont want to be destructive
        onnx_buffer = io.BytesIO()
        onnx_buffer.write(onnx_simp.SerializeToString())
        onnx_buffer.seek(0)
        onnx_buffer.name = "onnx_simp_mem_rife.onnx" 
        
        #network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
        network = network_from_onnx_bytes(onnx_buffer.getvalue(), flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
        
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)

        builder = network[0]
        config = builder.create_builder_config()
        config.progress_monitor = TQDMProgressMonitor()

        #config.set_flag(trt.BuilderFlag.FP16) if fp16 else None
        config.set_flag(trt.BuilderFlag.REFIT) if enable_refit else None
        
        # Limit workspace (e.g., to 2GB or 4GB depending on your GPU)
        # 1 << 30 is 1GB
        #config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
        config.builder_optimization_level = 5
        
        # Use Master Cache
        timing_cache = get_timing_cache(config)
        config.set_timing_cache(timing_cache, False) 

        profiles = copy.deepcopy(p)
        for profile in profiles:
            # Last profile is used for set_calibration_profile.
            calib_profile = profile.fill_defaults(network[1]).to_trt(
                builder, network[1]
            )
            config.add_optimization_profile(calib_profile)

        result = 0

        try:
            engine = engine_from_network(network, config)
            save_engine(engine, path=self.engine_path)
        except Exception as e:
            error(f"Failed to build engine: {e}")            
            result = 1
            
        # Update Master Cache with new findings
        save_timing_cache(timing_cache)
        print(f"Updated Master Cache.")
        
        return result

    def load(self):
        """Loads the serialized engine from disk."""
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        """Initializes the execution context."""
        if self.engine is None:
            self.load()

        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
        #    self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        """Allocates GPU memory for input and output tensors.

        Args:
            shape_dict (dict, optional): Specific shapes for tensors.
            device (str): Device to allocate on (default "cuda").
        """
        if hasattr(self, 'cuda_graph_instance') and self.cuda_graph_instance is not None:
            try:
                cudart.cudaGraphDestroy(self.cuda_graph_instance)
            except:
                pass
            self.cuda_graph_instance = None
        if hasattr(self, 'graph') and self.graph is not None:
            try:
                cudart.cudaGraphDestroy(self.graph)
            except:
                pass
            self.graph = None

        nvtx.range_push("allocate_buffers")
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]["shape"]
            else:
                shape = self.context.get_tensor_shape(name)

            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=device)
            self.tensors[binding] = tensor
        nvtx.range_pop()

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        """Performs inference on the provided data.

        Args:
            feed_dict (dict): Input tensors.
            stream (cuda.Stream): CUDA stream for execution.
            use_cuda_graph (bool): If True, uses/captures CUDA graphs.

        Returns:
            OrderedDict: The output tensors.
        """
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream.ptr))
                CUASSERT(cudart.cudaStreamSynchronize(stream.ptr))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream.ptr)
                if not noerror:
                    raise ValueError("ERROR: inference failed.")
                # capture cuda graph
                CUASSERT(
                    cudart.cudaStreamBeginCapture(stream.ptr, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
                )
                self.context.execute_async_v3(stream.ptr)
                self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream.ptr))
                self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
        else:
            noerror = self.context.execute_async_v3(stream.ptr)
            if not noerror:
                raise ValueError("ERROR: inference failed.")

        return self.tensors

