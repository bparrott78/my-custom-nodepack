"""Top-level package for my_custom_nodepack."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Braeden"""
__email__ = "bparrott78@gmail.com"
__version__ = "0.0.1"

from .src.my_custom_nodepack import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"
