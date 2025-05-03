import typing  # for type hints
if typing.TYPE_CHECKING:
    import torch  # type: ignore[import-unresolved]
else:
    try:
        import torch
    except ImportError:
        pass  # torch will be available in ComfyUI runtime

import numpy as np
from PIL import Image, ImageOps
import os # Added for ProjectContextNode
from datetime import date # Added for ProjectContextNode

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


class ProjectContextNode:
    """
    A node to build project directory structure and context information.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client_name": ("STRING", {"default": "", "multiline": False, "label": "Client name"}),
                "project_sku": ("STRING", {"default": "", "multiline": False, "label": "Project/SKU"}),
                "stage": (["POC", "PROD", "REV1", "FINAL"],),
                "branch": ("STRING", {"default": "", "multiline": False, "label": "Branch/Product type"}),
                "auto_date": (["enable", "disable"],),
                "base_path": ("STRING", {"default": "", "multiline": False, "label": "Base path"}),
            },
            "optional": {
                "due_date": ("STRING", {"default": "", "multiline": False, "label": "Due date (YYYY-MM-DD)"}),
            },
        }

    RETURN_TYPES = ("STRING", "CONDITIONING") # Assuming CONDITIONING is a valid type or placeholder
    RETURN_NAMES = ("folder_path", "context")
    FUNCTION = "execute"
    CATEGORY = "ProjectContext"

    def execute(self, client_name, project_sku, stage, branch, auto_date, base_path, due_date=None):
        # prepare date prefix
        today_str = date.today().isoformat() if auto_date == "enable" else ""
        # build path parts
        parts = [base_path, client_name, project_sku]
        # stage with optional due-date suffix
        stage_part = f"{stage}_due-{due_date}" if due_date else stage
        parts.append(stage_part)
        # branch folder, prefixed by date if enabled
        branch_part = f"{today_str}_{branch}" if auto_date == "enable" else branch
        parts.append(branch_part)
        folder_path = os.path.join(*parts)
        os.makedirs(folder_path, exist_ok=True)
        # compute due date info
        days_left = None
        past_due = False
        if due_date:
            try:
                due = date.fromisoformat(due_date)
                days_left = (due - date.today()).days
                past_due = days_left < 0
            except ValueError:
                days_left = None
        # assemble context dictionary
        context = {
            "client_name": client_name,
            "project_sku": project_sku,
            "stage": stage,
            "branch": branch,
            "auto_date": auto_date == "enable",
            "base_path": base_path,
            "due_date": due_date,
            "today": today_str, # Use the string version
            "days_left": days_left,
            "past_due": past_due,
            "folder_path": folder_path,
        }
        # Note: Returning a dictionary directly might not work for CONDITIONING.
        # This might need adjustment based on how ComfyUI handles CONDITIONING types.
        # For now, returning the dictionary as the second element.
        return (folder_path, context)


# Renamed PixelPerfectGrid to PixelArtGridNode
class PixelArtGridNode:
    CROP_METHODS = ["top", "mid", "bottom", "rescale"]
    # OUTPUT_SIZES removed as it's defined implicitly by the input widget range

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                # Renamed final_pixels to pixel_count for clarity
                "pixel_count": ("INT", {"default": 16, "min": 4, "max": 128, "step": 4, "label": "Pixel Count (Square)"}),
                "output_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 128}),
                "crop_method": (s.CROP_METHODS,),
                # Added palette size slider
                "palette_size": ("INT", {"default": 256, "min": 2, "max": 256, "step": 1, "label": "Palette Size"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    # Renamed outputs for clarity
    RETURN_NAMES = ("cropped_image", "pixelated_image",)
    FUNCTION = "process"
    # Updated Category
    CATEGORY = "Pixel" # Changed category to match the old PixelArtGridNode

    # Renamed process method parameter final_pixels to pixel_count
    # Added palette_size parameter
    def process(self, image, pixel_count, output_size, crop_method, palette_size):
        # Check if torch is available
        if 'torch' not in globals():
             raise ImportError("Torch is required for PixelArtGridNode but not available.")
        pil_images = tensor_to_pil(image)
        if not pil_images:
            # Return empty tensors if input is empty
            empty_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_tensor, empty_tensor)

        processed_cropped_images = []
        processed_pixelated_images = []

        for idx, pil_image in enumerate(pil_images):
            print(f"--- Processing Image {idx+1} ---")
            width, height = pil_image.size
            min_dim = min(width, height)
            max_dim = max(width, height)
            diff = max_dim - min_dim

            # 1. Crop or Rescale (Pad) to Square
            if crop_method == "rescale":
                # Pad the smaller dimension to match the larger one
                delta_w = max_dim - width
                delta_h = max_dim - height
                padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                # Use ImageOps.expand to pad; assumes black padding, change fill if needed
                cropped_pil = ImageOps.expand(pil_image, padding, fill=(0,0,0)) # Pad with black
                print(f"Rescaling image to {cropped_pil.size[0]}x{cropped_pil.size[1]}")
            else:
                # Crop the larger dimension
                if width == height:
                    cropped_pil = pil_image # Already square
                    print(f"Image already square: {width}x{height}")
                elif width > height: # Landscape - crop width
                    left = 0
                    if crop_method == "mid":
                        left = diff // 2
                    elif crop_method == "bottom": # 'bottom' crop means keep bottom, crop top -> irrelevant for width crop
                        left = diff # Keep right side
                    right = left + min_dim
                    box = (left, 0, right, height)
                    cropped_pil = pil_image.crop(box)
                    print(f"Cropping width using '{crop_method}' method. New size: {cropped_pil.size[0]}x{cropped_pil.size[1]}")
                else: # Portrait - crop height
                    top = 0
                    if crop_method == "mid":
                        top = diff // 2
                    elif crop_method == "bottom":
                        top = diff # Keep bottom side
                    bottom = top + min_dim
                    box = (0, top, width, bottom)
                    cropped_pil = pil_image.crop(box)
                    print(f"Cropping height using '{crop_method}' method. New size: {cropped_pil.size[0]}x{cropped_pil.size[1]}")

            # Store the cropped/rescaled image before pixelation, resized to output_size
            # Use LANCZOS for quality resize
            final_cropped_pil = cropped_pil.resize((output_size, output_size), Image.Resampling.LANCZOS)
            processed_cropped_images.append(final_cropped_pil)


            # 2. Pixelate
            # Use the renamed parameter pixel_count
            print(f"Pixelating to {pixel_count}x{pixel_count} pixels.")
            # Downscale the *original cropped_pil* (before final resize) to pixel_count x pixel_count using averaging
            small_image = cropped_pil.resize((pixel_count, pixel_count), Image.Resampling.BOX)

            # 3. Resize to Output Size
            print(f"Resizing pixelated image to final output size: {output_size}x{output_size}.")
            # Upscale using nearest neighbor to get the blocky pixel effect
            pixelated_pil = small_image.resize((output_size, output_size), Image.Resampling.NEAREST)

            # 4. Quantize Palette
            print(f"Quantizing image {idx+1} to {palette_size} colors.")
            # Convert to RGB before quantizing if it has alpha, as quantize might handle alpha poorly or discard it.
            # Store original mode to convert back if needed (e.g., RGBA)
            original_mode = pixelated_pil.mode
            if original_mode == 'RGBA':
                # If alpha is important, a more complex process is needed.
                # For simplicity, we convert to RGB, quantize, then convert back.
                # This might lose nuanced alpha, but preserves basic transparency if palette includes it.
                quantized_pil = pixelated_pil.convert('RGB').quantize(colors=palette_size, method=Image.Quantize.MEDIANCUT)
                # Convert back to the original mode (e.g., RGBA) if possible
                # The palette might not perfectly support RGBA, so convert to RGBA after getting the palette
                quantized_pil = quantized_pil.convert(original_mode)
            elif original_mode == 'P': # Already paletted, requantize
                 quantized_pil = pixelated_pil.convert('RGB').quantize(colors=palette_size, method=Image.Quantize.MEDIANCUT)
                 quantized_pil = quantized_pil.convert('RGB') # Convert back to RGB
            else: # Assume RGB or L
                quantized_pil = pixelated_pil.quantize(colors=palette_size, method=Image.Quantize.MEDIANCUT)
                # Convert back to RGB mode for consistency in the tensor output
                quantized_pil = quantized_pil.convert('RGB')

            processed_pixelated_images.append(quantized_pil)
            print(f"--- Finished Image {idx+1} ---")

        # Convert back to tensors
        cropped_tensor = pil_to_tensor(processed_cropped_images)
        pixelated_tensor = pil_to_tensor(processed_pixelated_images)

        return (cropped_tensor, pixelated_tensor,)


# Removed OpenAI image generation code since we're using gpt_image.py instead


# --- ComfyUI Registration ---
# Updated mappings to remove OpenAIImageGenerator
NODE_CLASS_MAPPINGS = {
    "ProjectContextNode": ProjectContextNode,
    "PixelArtGridNode": PixelArtGridNode, # Now points to the renamed class
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProjectContextNode": "Project Context",
    "PixelArtGridNode": "Pixel Art Grid", # Using the target display name
}
