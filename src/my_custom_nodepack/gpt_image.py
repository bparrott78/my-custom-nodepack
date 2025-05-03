"""
GPT-Image-1 Generator Node for ComfyUI
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
    """Converts a torch tensor (B, H, W, C) [0, 1] to a list of PIL Images [0, 255]."""
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
    """Converts a list of PIL Images [0, 255] to a torch tensor (B, H, W, C) [0, 1]."""
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

# Helper function to convert PIL Image to base64
def pil_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Helper function to convert base64 to PIL Image
def base64_to_pil(base64_string):
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))

# Get the directory where this module is located
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define config file path
CONFIG_FILE = os.path.join(MODULE_DIR, ".gpt_image_config.json")

def save_api_key_to_file(api_key):
    """Save API key to config file with a warning."""
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
    """Load API key from config file if it exists."""
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
    A ComfyUI node for generating images using OpenAI"s GPT-Image-1 model.
    Supports up to three image inputs along with a text prompt.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": load_api_key()}),
                "save_api_key": (["no", "yes"],),  # Option to save API key with warning
                "background": (["auto", "transparent", "opaque"],),  # Background options
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "moderation": (["auto", "low"],),  # Moderation options
                "output_format": (["png", "jpeg", "webp"],),  # Output format options
                "quality": (["auto", "high", "medium", "low"],),  # Quality options
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),  # Number of images to generate
                "size": (["1024x1024", "1536x1024", "1024x1536", "auto"],),  # All GPT-Image-1 supported sizes
            },
            "optional": {
                "image_1": ("IMAGE",),  # First reference image
                "image_2": ("IMAGE",),  # Second reference image
                "image_3": ("IMAGE",),  # Third reference image
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "MyCustomNodePack/OpenAI"

    def generate(self, api_key, save_api_key, background, prompt, moderation, output_format, quality, n, size, image_1=None, image_2=None, image_3=None):
        # Save API key if requested
        if save_api_key == "yes" and api_key:
            save_api_key_to_file(api_key)
            
        # Use saved API key if none is provided
        if not api_key:
            api_key = load_api_key()
            if not api_key:
                raise ValueError("OpenAI API Key is required. Please enter an API key or load a saved one.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Base API request data
        data = {
            "model": "gpt-image-1",
            "prompt": prompt,
            "n": n,
            "size": size,
            "background": background,
            "moderation": moderation,
            "output_format": output_format,
            "quality": quality
        }
        
        # Process input images if provided
        reference_images = []
        
        for img_tensor in [image_1, image_2, image_3]:
            if img_tensor is not None:
                # Convert tensor to PIL image
                pil_images = tensor_to_pil(img_tensor)
                if pil_images:
                    # Take only the first image from each tensor
                    img_base64 = pil_to_base64(pil_images[0])
                    reference_images.append(f"data:image/png;base64,{img_base64}")
        
        # Add reference images to request if available
        if reference_images:
            data["reference_images"] = reference_images

        print(f"Generating {n} images with GPT-Image-1")
        print(f"Prompt: {prompt}")
        print(f"Using {len(reference_images)} reference images")
        
        # Choose the correct endpoint based on whether we have input images
        if reference_images:
            # Use the edits endpoint when reference images are provided
            endpoint = "https://api.openai.com/v1/images/edits"
            print("Using OpenAI images/edits endpoint with multipart/form-data")
            
            # For edits endpoint, we need to use multipart/form-data instead of JSON
            # Remove Content-Type header as it will be set automatically
            headers = {"Authorization": f"Bearer {api_key}"}
            
            # Convert the first reference image to bytes for multipart upload
            pil_image = tensor_to_pil(image_1)[0]
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Create multipart form data
            files = {
                'image': ('image.png', img_byte_arr, 'image/png'),
            }
            
            # For any additional images after the first, we'll use the prompt to describe them
            # since the edits endpoint only accepts one image
            if len(reference_images) > 1:
                prompt += " (Additional reference images were provided but only the first one is used for editing)"
            
            # Prepare form data
            form_data = {
                'model': "gpt-image-1",
                'prompt': prompt,
                'n': str(n),
                'size': size,
                'background': background,
                'moderation': moderation,
                'response_format': 'b64_json', # We want base64 json for consistent handling
                'output_format': output_format,
                'quality': quality
            }
            
            # Make API request with multipart form data
            response = requests.post(
                endpoint,
                headers=headers,
                files=files,
                data=form_data
            )
        else:
            # Use the generations endpoint when no reference images are provided
            endpoint = "https://api.openai.com/v1/images/generations"
            print("Using OpenAI images/generations endpoint with JSON")
            
            # For generations endpoint, we use JSON as before
            headers["Content-Type"] = "application/json"
            
            # Make API request with JSON data
            response = requests.post(
                endpoint,
                headers=headers,
                json=data
            )

        if response.status_code != 200:
            try:
                error_details = response.json()
                error_message = error_details.get("error", {}).get("message", response.text)
                raise Exception(f"OpenAI API Error: {response.status_code} - {error_message}")
            except requests.exceptions.JSONDecodeError:
                raise Exception(f"OpenAI API Error: {response.status_code} - {response.text}")
                
        # Process results
        result = response.json()
        
        # Handle base64-encoded images
        output_images = []
        for item in result["data"]:
            # Extract base64 data and convert to PIL image
            base64_data = item["b64_json"]
            pil_image = base64_to_pil(base64_data)
            
            # Convert PIL image to tensor
            output_images.append(pil_to_tensor(pil_image))

        if not output_images:
            raise Exception("No images were generated by the API.")

        # Stack all generated images into a single tensor
        return (torch.cat(output_images, dim=0),)


# --- ComfyUI Registration ---
NODE_CLASS_MAPPINGS = {
    "GPTImageGenerator": GPTImageGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImageGenerator": "GPT Image Generator",
}
