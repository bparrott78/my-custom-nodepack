import typing
if typing.TYPE_CHECKING:
    import torch
else:
    try:
        import torch
    except ImportError:
        pass

import numpy as np
from PIL import Image

def tensor_to_pil(tensor):
    """Converts a torch tensor (B, H, W, C) [0, 1] to a list of PIL Images [0, 255]."""
    if tensor is None:
        return []
    batch_size = tensor.shape[0]
    images = []
    for i in range(batch_size):
        img_np = tensor[i].cpu().numpy()  # Get i-th image and convert to numpy
        img_np = (img_np * 255).astype(np.uint8)  # Scale to [0, 255] and convert type
        pil_image = Image.fromarray(img_np, 'RGB' if img_np.shape[-1] == 3 else 'RGBA')
        images.append(pil_image)
    return images

def pil_to_tensor(pil_images):
    """Converts a list of PIL Images [0, 255] to a torch tensor (B, H, W, C) [0, 1]."""
    if not isinstance(pil_images, list):
        pil_images = [pil_images]

    tensors = []
    for img in pil_images:
        img_np = np.array(img).astype(np.float32) / 255.0  # Convert to numpy array and scale to [0, 1]
        tensor = torch.from_numpy(img_np)
        if len(tensor.shape) == 2: # Handle grayscale images by adding channel dim
             tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
        elif tensor.shape[2] == 4: # Keep RGBA
             pass
        elif tensor.shape[2] == 1: # Handle single channel images
             tensor = tensor.repeat(1, 1, 3)

        tensors.append(tensor)

    # Stack tensors along a new batch dimension
    return torch.stack(tensors)
