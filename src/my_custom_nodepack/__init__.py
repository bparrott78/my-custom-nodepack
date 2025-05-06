from .nodes import ProjectContextNode, PixelArtGridNode
from .gpt_image import GPTImageGenerator
from .string_utils import StringListToString
from .dynamic_lora_stack import DynamicLoraStack
from .random_lora_stack import RandomLoraStack

NODE_CLASS_MAPPINGS = {
    "ProjectContextNode": ProjectContextNode,
    "PixelArtGridNode": PixelArtGridNode,
    "GPTImageGenerator": GPTImageGenerator,
    "StringListToString": StringListToString,
    "DynamicLoraStack": DynamicLoraStack,
    "RandomLoraStack": RandomLoraStack,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProjectContextNode": "SP Project Context",
    "PixelArtGridNode": "SP Pixel Art Grid",
    "GPTImageGenerator": "SP GPT Image Generator",
    "StringListToString": "SP String List to String",
    "DynamicLoraStack": "SP Dynamic LoRA Stack",
    "RandomLoraStack": "SP Random LoRA Stack",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

 