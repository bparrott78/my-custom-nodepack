import typing  # for type hints
if typing.TYPE_CHECKING:
    import torch  # type: ignore[import-unresolved]
else:
    try:
        import torch
    except ImportError:
        pass  # torch will be available in ComfyUI runtime


# os and date are no longer needed as ProjectContextNode is removed



# --- ComfyUI Registration ---
# Updated mappings to remove ProjectContextNode
NODE_CLASS_MAPPINGS = {
}

NODE_DISPLAY_NAME_MAPPINGS = {
}
