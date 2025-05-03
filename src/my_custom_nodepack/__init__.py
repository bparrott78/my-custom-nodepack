from .nodes import ProjectContextNode, PixelArtGridNode
from .gpt_image import GPTImageGenerator
from .string_utils import StringListToString

NODE_CLASS_MAPPINGS = {
    "ProjectContextNode": ProjectContextNode,
    "PixelArtGridNode": PixelArtGridNode,
    "GPTImageGenerator": GPTImageGenerator,
    "StringListToString": StringListToString,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProjectContextNode": "Project Context",
    "PixelArtGridNode": "Pixel Art Grid",
    "GPTImageGenerator": "GPT Image Generator",
    "StringListToString": "String List to String",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

 