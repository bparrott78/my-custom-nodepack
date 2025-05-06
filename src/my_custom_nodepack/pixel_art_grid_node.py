import typing
if typing.TYPE_CHECKING:
    import torch
else:
    try:
        import torch
    except ImportError:
        pass

from PIL import Image, ImageOps
from .image_utils import tensor_to_pil, pil_to_tensor

class PixelArtGridNode:
    CROP_METHODS = ["top", "mid", "bottom", "rescale"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "pixel_count": ("INT", {"default": 16, "min": 4, "max": 128, "step": 4, "label": "Pixel Count (Square)"}),
                "output_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 128}),
                "crop_method": (s.CROP_METHODS,),
                "palette_size": ("INT", {"default": 256, "min": 2, "max": 256, "step": 1, "label": "Palette Size"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("cropped_image", "pixelated_image",)
    FUNCTION = "process"
    CATEGORY = "MyCustomNodePack/Image Processing"

    def process(self, image, pixel_count, output_size, crop_method, palette_size):
        if 'torch' not in globals():
             raise ImportError("Torch is required for PixelArtGridNode but not available.")
        pil_images = tensor_to_pil(image)
        if not pil_images:
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

            if crop_method == "rescale":
                delta_w = max_dim - width
                delta_h = max_dim - height
                padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                cropped_pil = ImageOps.expand(pil_image, padding, fill=(0,0,0))
                print(f"Rescaling image to {cropped_pil.size[0]}x{cropped_pil.size[1]}")
            else:
                if width == height:
                    cropped_pil = pil_image
                    print(f"Image already square: {width}x{height}")
                elif width > height:
                    left = 0
                    if crop_method == "mid":
                        left = diff // 2
                    elif crop_method == "bottom":
                        left = diff
                    right = left + min_dim
                    box = (left, 0, right, height)
                    cropped_pil = pil_image.crop(box)
                    print(f"Cropping width using '{crop_method}' method. New size: {cropped_pil.size[0]}x{cropped_pil.size[1]}")
                else:
                    top = 0
                    if crop_method == "mid":
                        top = diff // 2
                    elif crop_method == "bottom":
                        top = diff
                    bottom = top + min_dim
                    box = (0, top, width, bottom)
                    cropped_pil = pil_image.crop(box)
                    print(f"Cropping height using '{crop_method}' method. New size: {cropped_pil.size[0]}x{cropped_pil.size[1]}")

            final_cropped_pil = cropped_pil.resize((output_size, output_size), Image.Resampling.LANCZOS)
            processed_cropped_images.append(final_cropped_pil)

            small_image = cropped_pil.resize((pixel_count, pixel_count), Image.Resampling.BOX)
            pixelated_pil = small_image.resize((output_size, output_size), Image.Resampling.NEAREST)

            original_mode = pixelated_pil.mode
            if original_mode == 'RGBA':
                quantized_pil = pixelated_pil.convert('RGB').quantize(colors=palette_size, method=Image.Quantize.MEDIANCUT)
                quantized_pil = quantized_pil.convert(original_mode)
            elif original_mode == 'P':
                 quantized_pil = pixelated_pil.convert('RGB').quantize(colors=palette_size, method=Image.Quantize.MEDIANCUT)
                 quantized_pil = quantized_pil.convert('RGB')
            else:
                quantized_pil = pixelated_pil.quantize(colors=palette_size, method=Image.Quantize.MEDIANCUT)
                quantized_pil = quantized_pil.convert('RGB')

            processed_pixelated_images.append(quantized_pil)
            print(f"--- Finished Image {idx+1} ---")

        cropped_tensor = pil_to_tensor(processed_cropped_images)
        pixelated_tensor = pil_to_tensor(processed_pixelated_images)

        return (cropped_tensor, pixelated_tensor,)
