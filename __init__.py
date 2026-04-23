"""ComfyUI-Rife-TensorRT Extension Initialization.

This module acts as the entry point for ComfyUI, exporting the node classes 
and display names required to register the TensorRT-RTX RIFE implementation 
into the ComfyUI node graph.

Attributes:
    NODE_CLASS_MAPPINGS (dict): Maps internal lookup strings to node classes.
    NODE_DISPLAY_NAME_MAPPINGS (dict): Maps internal lookup strings to 
        human-readable UI names.
    WEB_DIRECTORY (str): Path to the directory containing frontend 
        javascript extensions.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

