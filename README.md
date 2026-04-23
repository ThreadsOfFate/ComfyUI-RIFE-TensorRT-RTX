# ComfyUI RIFE TensorRT-RTX

[![python](https://img.shields.io/badge/python-3.13.9-green)](https://www.python.org/downloads/release/python-31313/)
[![cuda](https://img.shields.io/badge/cuda-13.0-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT_RTX-1.4.0.76-green)](https://developer.nvidia.com/tensorrt)
[![by-nc-sa/4.0](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)

## Overview

This project is a fork of the original [ComfyUI_RIFE_TensorRT_Auto](https://github.com/silveroxides/ComfyUI_RIFE_TensorRT_Auto) by [silveroxides](https://github.com/silveroxides). 
This project uses TensorRT-RTX to do RIFE interpolation

## Features
- **NVIDIA TensorRT Acceleration**: Leverages Tensor Cores for up to 30x faster frame interpolation compared to standard local methods.
- **Dynamic Shape Support**: Configure minimum, optimal, and maximum resolutions to handle varying input sizes without rebuilding the engine for every image.
- **Automatic Engine Building**: Automatically compiles specialized `.trt` files from your `.onnx` models if a matching engine isn't found.
- **Persistent Timing Cache**: Saves hardware-specific optimization data to a local cache to speed up subsequent engine builds.
- **Memory Efficient**: Optimized for VRAM management with automatic garbage collection and cache clearing after inference.


## Examples

![Simple Example](assets/example.png)

## Installation

### ComfyUI Manager

This project has not been submitted to the ComfyUI Manager registry yet. But you can still install it this way:

1. Open ComfyUI Manager.
2. Click the `Custom Nodes Manager` to open the custom nodes manager page.
3. On the bottom right corner, click the `Install via Git URL` button.
4. Enter the URL of this repository: `https://github.com/ThreadsOfFate/ComfyUI-RIFE-TensorRT-RTX`
5. Click "Confirm".
6. Restart ComfyUI.

### Manual

1. On the github page, click on the green `<> Code` button and then "Download ZIP".
2. Extract the root folder within the downloaded ZIP file to your ComfyUI `custom_nodes` directory.
3. Using the same python environment that runs ComfyUI, install the required dependencies: `python -m pip install -r custom_nodes/ComfyUI-Upscaler-TensorRT-Advanced/requirements.txt`.
4. Restart ComfyUI.


### Requirements
pip install -r requirements.txt


### CUDA Toolkit Required
Please download NVIDIA CUDA Toolkit [https://developer.nvidia.com/cuda/toolkit](https://developer.nvidia.com/cuda/toolkit)

###Nodes Included
🟦 RIFE RTX Engine Builder (RIFERTXEngineLoader)
  Loads a TensorRT-RTX engine. If the engine file for your specific GPU and resolution settings does not exist, it will automatically build one from the source ONNX model.
🟦 RIFE RTX Frame Interpolation (RIFERTXFrameInterpolation)
  The core execution node. It takes an image and a loaded engine to perform the upscale. It supports an optional resize input for custom final dimensions.
🟦 RIFE RTX Engine Config (RIFERTXEngineConfig)
  Allows you to define the resolution bounds (min/opt/max) for the engine.
  Min/Max: The range of resolutions the engine can handle.
  Opt: The resolution the engine is most optimized for.
  
  
GPU: NVIDIA RTX Series (TensorRT-RTX requires Tensor Cores).
Software: NVIDIA Drivers and a compatible version of TensorRT (handled via requirements.txt).

###Troubleshooting
First Build Time: The first time you load a model with new shape settings, it may take several minutes to compile the engine. Check the console for progress via the TQDM bar.
VRAM Errors: If the build fails, try reducing the width_max or height_max in the Dynamic Shape Config.


## 🛠️ Supported Models

rife47_ensemble_True_scale_1_sim
rife48_ensemble_True_scale_1_sim
rife49_ensemble_True_scale_1_sim
