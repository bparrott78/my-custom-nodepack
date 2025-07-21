"""
GPT-Image-1 Generator Node for ComfyUI

This node integrates with OpenAI's GPT-Image-1 API to generate or edit images using up to three reference images and a text prompt.
"""
import typing  # for type hints
if typing.TYPE_CHECKING:
    import torch  # type: ignore[import-unresolved]
else:
    try:
        import torch
    except ImportError:
        pass  # torch will be available in ComfyUI runtime

import numpy as np
from PIL import Image
import io
import requests
import json
import base64
import os
import warnings

def tensor_to_pil(tensor):
    """
    Converts a torch tensor (B, H, W, C) [0, 1] to a list of PIL Images [0, 255].
    Used to convert ComfyUI image tensors to PIL format for API upload.
    """
    if tensor is None:
        return []
    batch_size = tensor.shape[0]
    images = []
    for i in range(batch_size):
        img_np = tensor[i].cpu().numpy()  # Get i-th image and convert to numpy
        img_np = (img_np * 255).astype(np.uint8)  # Scale to [0, 255] and convert type
        pil_image = Image.fromarray(img_np, "RGB" if img_np.shape[-1] == 3 else "RGBA")
        images.append(pil_image)
    return images

def pil_to_tensor(pil_images):
    """
    Converts a list of PIL Images [0, 255] to a torch tensor (B, H, W, C) [0, 1].
    Used to convert API results back to ComfyUI format.
    """
    if not isinstance(pil_images, list):
        pil_images = [pil_images]

    tensors = []
    for img in pil_images:
        img_np = np.array(img).astype(np.float32) / 255.0  # Convert to numpy array and scale to [0, 1]
        tensor = torch.from_numpy(img_np)
        if len(tensor.shape) == 2:  # Handle grayscale images by adding channel dim
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
        elif tensor.shape[2] == 4:  # Keep RGBA
            pass
        elif tensor.shape[2] == 1:  # Handle single channel images
            tensor = tensor.repeat(1, 1, 3)

        tensors.append(tensor)

    # Stack tensors along a new batch dimension
    return torch.stack(tensors)

def pil_to_base64(image):
    """
    Helper function to convert a PIL Image to a base64 string (PNG format).
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_pil(base64_string):
    """
    Helper function to convert a base64 string to a PIL Image.
    """
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of this module
CONFIG_FILE = os.path.join(MODULE_DIR, ".gpt_image_config.json")  # Config file for API key

def save_api_key_to_file(api_key):
    """
    Save API key to config file with a warning about plain text storage.
    """
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({"api_key": api_key}, f)
        warnings.warn("API key saved in plain text. This is insecure and should only be used in trusted environments.")
        print(f"API key saved to {CONFIG_FILE}")
        return True
    except Exception as e:
        print(f"Error saving API key: {e}")
        return False

def load_api_key():
    """
    Load API key from config file if it exists.
    """
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                return config.get("api_key", "")
        return ""
    except Exception as e:
        print(f"Error loading API key: {e}")
        return ""

class GPTImageGenerator:
    """
    ComfyUI node for generating or editing images using OpenAI's GPT-Image-1 API.
    Supports up to three reference images and a text prompt.
    """
    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input fields for the node, including all documented options for GPT-Image-1 API.
        """
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": load_api_key()}),
                "save_api_key": (["no", "yes"],),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
                "size": (["1024x1024", "1536x1024", "1024x1536", "auto"], {"default": "auto"}),
                "quality": (["auto", "low", "medium", "high"], {"default": "auto"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "moderation": (["auto", "low"], {"default": "auto"}),
                "background": (["auto", "transparent", "opaque"], {"default": "auto"}),
            },
            "optional": {
                "output_compression": ("INT", {"default": 0, "min": 0, "max": 100}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "MyCustomNodePack/OpenAI"

    def generate(self, api_key, save_api_key, prompt, n, size, quality, output_format, moderation, background, output_compression=0, image_1=None, image_2=None, image_3=None):
        """
        Main node function. Sends a request to OpenAI's API to generate or edit images.
        Validates input values and provides user-friendly error messages for common mistakes.
        """
        # Save API key if requested
        if save_api_key == "yes" and api_key:
            save_api_key_to_file(api_key)

        # Load API key from config if not provided
        if not api_key:
            api_key = load_api_key()
            if not api_key:
                raise ValueError("OpenAI API Key is required. Please enter an API key or load a saved one.")

        # Validate size
        allowed_sizes = ["1024x1024", "1536x1024", "1024x1536", "auto"]
        if size not in allowed_sizes:
            raise ValueError(f"size must be one of {allowed_sizes}, got '{size}'")

        # Validate quality
        allowed_quality = ["auto", "low", "medium", "high"]
        if quality not in allowed_quality:
            raise ValueError(f"quality must be one of {allowed_quality}, got '{quality}'")

        # Validate output_format
        allowed_output_format = ["png", "jpeg", "webp"]
        if output_format not in allowed_output_format:
            raise ValueError(f"output_format must be one of {allowed_output_format}, got '{output_format}'")

        # Validate moderation
        allowed_moderation = ["auto", "low"]
        if moderation not in allowed_moderation:
            raise ValueError(f"moderation must be one of {allowed_moderation}, got '{moderation}'")

        # Validate background
        allowed_background = ["auto", "transparent", "opaque"]
        if background not in allowed_background:
            raise ValueError(f"background must be one of {allowed_background}, got '{background}'")

        # Validate n
        try:
            n_int = int(n)
        except Exception:
            raise ValueError(f"n must be an integer (number of images to generate), got '{n}'")
        if not (1 <= n_int <= 4):
            raise ValueError("n must be between 1 and 4")

        # Validate output_compression
        if output_format in ["jpeg", "webp"]:
            try:
                output_compression_int = int(output_compression)
            except Exception:
                raise ValueError(f"output_compression must be an integer between 0 and 100, got '{output_compression}'")
            if not (0 <= output_compression_int <= 100):
                raise ValueError("output_compression must be between 0 and 100")
        else:
            output_compression_int = None

        headers = {"Authorization": f"Bearer {api_key}"}

        # Prepare up to three reference images for edits endpoint
        images = []
        for img_tensor in [image_1, image_2, image_3]:
            if img_tensor is not None:
                pil_img = tensor_to_pil(img_tensor)[0]
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)
                images.append(img_byte_arr)

        # Build payload for both endpoints
        payload = {
            "model": "gpt-image-1",
            "prompt": prompt,
            "n": str(n_int) if images else n_int,
            "size": size,
            "quality": quality,
            "output_format": output_format,
            "moderation": moderation,
            "background": background,
        }
        if output_compression_int is not None:
            payload["output_compression"] = output_compression_int

        if images:
            # Edits endpoint: multipart/form-data
            endpoint = "https://api.openai.com/v1/images/edits"
            files = {}
            for idx, img_bytes in enumerate(images):
                key = "image" if idx == 0 else f"image_{idx+1}"
                files[key] = (f"image_{idx+1}.png", img_bytes, "image/png")
            response = requests.post(endpoint, headers=headers, files=files, data=payload)
        else:
            # Generations endpoint: application/json
            endpoint = "https://api.openai.com/v1/images/generations"
            headers["Content-Type"] = "application/json"
            response = requests.post(endpoint, headers=headers, json=payload)

        # Handle API errors
        if response.status_code != 200:
            try:
                error_details = response.json()
                error_message = error_details.get("error", {}).get("message", response.text)
                raise Exception(f"OpenAI API Error: {response.status_code} - {error_message}")
            except Exception:
                raise Exception(f"OpenAI API Error: {response.status_code} - {response.text}")

        # Parse results and convert to tensors
        result = response.json()
        output_images = []
        for item in result["data"]:
            # Always expect b64_json for now, as OpenAI docs say API returns base64-encoded image data
            base64_data = item.get("b64_json")
            if base64_data:
                pil_image = base64_to_pil(base64_data)
                output_images.append(pil_to_tensor(pil_image))
            elif "url" in item:
                img_url = item["url"]
                img_resp = requests.get(img_url)
                pil_image = Image.open(io.BytesIO(img_resp.content))
                output_images.append(pil_to_tensor(pil_image))

        if not output_images:
            raise Exception("No images were generated by the API.")

        # Return stacked tensor of images
        return (torch.cat(output_images, dim=0),)


# --- ComfyUI Registration ---
# Register the node class and display name for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GPTImageGenerator": GPTImageGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImageGenerator": "GPT Image Generator",
}
