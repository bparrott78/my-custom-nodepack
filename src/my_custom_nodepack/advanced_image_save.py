\
import os
import json
import re
import numpy as np
from PIL import Image, PngImagePlugin
import folder_paths  # type: ignore # ComfyUI's path manager
from datetime import datetime # For time token

# Allowed extensions for saving
ALLOWED_EXT = {'.png', '.jpg', '.jpeg', '.gif', '.tiff', '.webp', '.bmp'}
# Extensions to strip from filename parts if requested
COMMON_STRIP_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.tiff', '.webp', '.bmp', # Common image formats
    '.safetensors', '.pt', '.ckpt', '.bin', '.lora', '.pth' # Common model/LoRA formats
}

# Simplified cstr for logging
def log_message(message, level="info"):
    """Helper function for logging messages from the node."""
    print(f"[AdvancedImageSave] [{level.upper()}] {message}")

# Simplified TextTokens - only handles [time(format)] for now
class SimpleTextTokens:
    """A simple class to parse tokens in strings, currently supporting [time(format)]."""
    def parseTokens(self, text_string):
        def replace_time(match):
            time_format = match.group(1)
            try:
                return datetime.now().strftime(time_format)
            except Exception as e:
                log_message(f"Error formatting time token with format '{time_format}': {e}", "error")
                return f"[time({time_format})]" # Return original token on error
        
        # Replace [time(FORMAT)] tokens
        text_string = re.sub(r"\\[time\\((.+?)\\)\\]", replace_time, text_string)
        return text_string

class AdvancedImageSave:
    """
    An advanced image saving node with options for path, filename, numbering,
    metadata embedding, and various format-specific settings.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = 'output'  # For ComfyUI's preview system

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "output_path": ("STRING", {"default": '[time(%Y-%m-%d)]', "multiline": False, "placeholder": "e.g., [time(%Y-%m-%d)]/project_name"}),
                # New filename parts
                "prefix_string": ("STRING", {"default": "ComfyUI", "placeholder": "e.g., ProjectA_"}),
                "name_string": ("STRING", {"default": "[time(%Y-%m-%d-%H%M%S)]", "placeholder": "e.g., ImageName"}),
                "postfix_string": ("STRING", {"default": "", "placeholder": "e.g., _Final"}),
                "strip_extensions_from_parts": (["No", "Yes"], {"default": "No", "label": "Strip Extensions from Name Parts"}),

                "filename_delimiter": ("STRING", {"default":"_"}),
                "filename_number_padding": ("INT", {"default":4, "min":1, "max":9, "step":1}),
                # New numbering position selector
                "numbering_position": (["after-name", "initial", "pre-name", "post-postfix"], {"default": "after-name"}),
                
                "extension": (['png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'bmp'], {"default": "png"}),
                "dpi": ("INT", {"default": 300, "min": 1, "max": 2400, "step": 1}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "optimize_image": (["true", "false"], {"default": "true"}),
                "lossless_webp": (["false", "true"], {"default": "false"}),
                "overwrite_mode": (["false", "prefix_as_filename"], {"default": "false"}),
                "embed_workflow": (["true", "false"], {"default": "true"}),
                "show_previews": (["true", "false"], {"default": "true"}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "saved_file_paths",)
    FUNCTION = "save_images_adv"
    OUTPUT_NODE = True
    CATEGORY = "MyCustomNodePack/IO"

    def _strip_known_extensions(self, filename_part):
        if not filename_part: return ""
        for ext in COMMON_STRIP_EXTENSIONS:
            if filename_part.lower().endswith(ext):
                return filename_part[:-len(ext)]
        return filename_part

    def _construct_filename_base(self, prefix, name, postfix, number_str, delimiter, numbering_position):
        parts = []
        # Order based on numbering_position
        if numbering_position == "initial":
            parts = [number_str, prefix, name, postfix]
        elif numbering_position == "pre-name":
            parts = [prefix, number_str, name, postfix]
        elif numbering_position == "after-name": # Default
            parts = [prefix, name, number_str, postfix]
        elif numbering_position == "post-postfix":
            parts = [prefix, name, postfix, number_str]
        else: # Fallback to default if unknown position
            parts = [prefix, name, number_str, postfix]

        # Filter out empty parts and join with delimiter
        # Only add delimiter if the previous part was not empty and the current part is not empty
        final_name = ""
        for i, part in enumerate(parts):
            if part: # If the current part is not empty
                if final_name and delimiter: # And if final_name is not empty (meaning a previous part was added)
                    final_name += delimiter
                final_name += part
        return final_name

    def save_images_adv(self, images, output_path_str, 
                        prefix_string, name_string, postfix_string, strip_extensions_from_parts,
                        filename_delimiter, filename_number_padding, numbering_position, 
                        extension, dpi, quality, optimize_image, lossless_webp, 
                        overwrite_mode, embed_workflow, show_previews,
                        prompt=None, extra_pnginfo=None):

        strip_ext_bool = (strip_extensions_from_parts == "Yes")
        optimize_image_bool = (optimize_image == "true")
        lossless_webp_bool = (lossless_webp == "true")
        embed_workflow_bool = (embed_workflow == "true")
        show_previews_bool = (show_previews == "true")

        token_parser = SimpleTextTokens()
        parsed_prefix_str = token_parser.parseTokens(prefix_string)
        parsed_name_str = token_parser.parseTokens(name_string)
        parsed_postfix_str = token_parser.parseTokens(postfix_string)

        if strip_ext_bool:
            parsed_prefix_str = self._strip_known_extensions(parsed_prefix_str)
            parsed_name_str = self._strip_known_extensions(parsed_name_str)
            parsed_postfix_str = self._strip_known_extensions(parsed_postfix_str)

        parsed_output_path_str = token_parser.parseTokens(output_path_str)
        
        # ... existing path setup and directory creation logic ...
        if not parsed_output_path_str or parsed_output_path_str.strip() in ['.', 'none', '']:
            current_output_path = self.output_dir
        else:
            if not os.path.isabs(parsed_output_path_str):
                current_output_path = os.path.join(self.output_dir, parsed_output_path_str)
            else:
                current_output_path = parsed_output_path_str
        current_output_path = os.path.normpath(current_output_path)

        if not os.path.exists(current_output_path):
            log_message(f"Path `{current_output_path}` doesn't exist. Creating directory.", "warning")
            try:
                os.makedirs(current_output_path, exist_ok=True)
            except Exception as e:
                log_message(f"Failed to create directory {current_output_path}: {e}", "error")
                return {"ui": {"images": []}, "result": (images, [],)}

        file_ext_lower = extension.lower()
        if '.' + file_ext_lower not in ALLOWED_EXT:
            log_message(f"Extension `{extension}` invalid. Valid: {ALLOWED_EXT}. Defaulting to .png", "error")
            file_ext_lower = "png"
        final_file_extension = '.' + file_ext_lower

        counter = 1
        if overwrite_mode == "false":
            # Revised counter initialization: iterate and try to parse numbers
            # This is more complex but more robust than a single regex for all numbering_positions
            existing_numbers = []
            try:
                for f_name_with_ext in os.listdir(current_output_path):
                    f_name_base = os.path.splitext(f_name_with_ext)[0]
                    # Attempt to deconstruct f_name_base based on current settings
                    # This is a simplified heuristic. A perfect deconstruction is very hard.
                    # We look for a number that could fit the padding at various positions.
                    # This part might need further refinement for very complex prefix/name/postfix combinations.
                    
                    # Try to extract number based on padding and delimiter, this is a heuristic
                    # It doesn't perfectly reverse the _construct_filename_base logic but tries to find plausible numbers
                    potential_num_str = ""
                    # Check for number at the start
                    if numbering_position == "initial" and len(f_name_base) >= filename_number_padding:
                        if f_name_base[:filename_number_padding].isdigit():
                            potential_num_str = f_name_base[:filename_number_padding]
                    # Check for number at the end
                    elif numbering_position == "post-postfix" and len(f_name_base) >= filename_number_padding:
                         if f_name_base[-filename_number_padding:].isdigit():
                            potential_num_str = f_name_base[-filename_number_padding:]
                    # For pre-name and after-name, it's harder without knowing the exact lengths of other parts
                    # A simpler approach: find any sequence of digits of the expected padding length
                    else: 
                        matches = re.findall(r'(\d{' + str(filename_number_padding) + r'})', f_name_base)
                        if matches: potential_num_str = matches[-1] # Take the last one as a guess

                    if potential_num_str.isdigit():
                        # Further check if the rest of the filename roughly matches the other parts
                        # This is a basic check to reduce false positives
                        temp_constructed_without_num = self._construct_filename_base(
                            parsed_prefix_str, parsed_name_str, parsed_postfix_str, 
                            "", filename_delimiter, numbering_position
                        ).replace(filename_delimiter+""+filename_delimiter, filename_delimiter) # clean double delim
                        
                        # Rough check: if removing the number and delimiters around it resembles the other parts
                        # This is highly heuristic
                        test_name = f_name_base
                        if numbering_position == "initial":
                            test_name = test_name[len(potential_num_str):]
                            if test_name.startswith(filename_delimiter): test_name = test_name[len(filename_delimiter):]
                        elif numbering_position == "post-postfix":
                            test_name = test_name[:-len(potential_num_str)]
                            if test_name.endswith(filename_delimiter): test_name = test_name[:-len(filename_delimiter)]
                        # For in-between positions, it's even harder to make a simple, reliable check here
                        # So we rely more on the number extraction itself for these cases.

                        # A very loose check: if the remaining parts are somewhat present
                        # This is not perfect and might need refinement for edge cases.
                        # For now, if a number of correct padding is found, we consider it.
                        existing_numbers.append(int(potential_num_str))

                if existing_numbers: counter = max(existing_numbers) + 1
            except OSError as e: log_message(f"Could not list dir {current_output_path} for counter: {e}", "warning")
            except Exception as e: log_message(f"Error initializing counter: {e}", "warning")

        output_files_list = []
        preview_results_list = []

        for image_tensor in images:
            i = 255. * image_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            pil_metadata = PngImagePlugin.PngInfo()
            if embed_workflow_bool:
                if prompt is not None: pil_metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for k, v in extra_pnginfo.items(): pil_metadata.add_text(k, json.dumps(v))
            
            current_file_name_base = ""
            if overwrite_mode == "prefix_as_filename":
                # Construct without number, using only prefix, name, postfix
                current_file_name_base = self._construct_filename_base(
                    parsed_prefix_str, parsed_name_str, parsed_postfix_str, 
                    "", filename_delimiter, "after-name" # Numbering position doesn't matter if number_str is empty
                )
            else:
                number_str = f"{counter:0{filename_number_padding}}"
                current_file_name_base = self._construct_filename_base(
                    parsed_prefix_str, parsed_name_str, parsed_postfix_str, 
                    number_str, filename_delimiter, numbering_position
                )
            
            final_file_name = current_file_name_base + final_file_extension
            full_output_file_path = os.path.join(current_output_path, final_file_name)

            if overwrite_mode == "false":
                while os.path.exists(full_output_file_path):
                    counter += 1
                    number_str = f"{counter:0{filename_number_padding}}"
                    current_file_name_base = self._construct_filename_base(
                        parsed_prefix_str, parsed_name_str, parsed_postfix_str, 
                        number_str, filename_delimiter, numbering_position
                    )
                    final_file_name = current_file_name_base + final_file_extension
                    full_output_file_path = os.path.join(current_output_path, final_file_name)
            
            # ... existing image saving logic (try/except block) ...
            try:
                save_kwargs = {}
                if file_ext_lower in ["jpg", "jpeg"]:
                    save_kwargs.update({'quality': quality, 'optimize': optimize_image_bool, 'dpi': (dpi, dpi)})
                    img.save(full_output_file_path, **save_kwargs)
                elif file_ext_lower == 'webp':
                    save_kwargs.update({'quality': quality, 'lossless': lossless_webp_bool, 'dpi': (dpi, dpi)})
                    img.save(full_output_file_path, **save_kwargs)
                elif file_ext_lower == 'png':
                    save_kwargs.update({'pnginfo': pil_metadata, 'optimize': optimize_image_bool, 'dpi': (dpi, dpi)})
                    img.save(full_output_file_path, **save_kwargs)
                elif file_ext_lower == 'bmp':
                    save_kwargs['dpi'] = (dpi, dpi)
                    img.save(full_output_file_path, **save_kwargs)
                elif file_ext_lower == 'tiff':
                    save_kwargs.update({'optimize': optimize_image_bool, 'dpi': (dpi, dpi)})
                    if embed_workflow_bool and (prompt or extra_pnginfo):
                         description = {}
                         if prompt: description["prompt"] = prompt
                         if extra_pnginfo: description["extra_pnginfo"] = extra_pnginfo
                         save_kwargs['description'] = json.dumps(description)
                    img.save(full_output_file_path, **save_kwargs)
                else: # gif or other
                    img.save(full_output_file_path)

                log_message(f"Image saved: {full_output_file_path}", "info")
                output_files_list.append(full_output_file_path)

                if show_previews_bool:
                    try:
                        common_ancestor = os.path.commonpath([self.output_dir, os.path.abspath(current_output_path)])
                        if common_ancestor == self.output_dir or common_ancestor == os.path.abspath(current_output_path):
                             relative_subfolder = os.path.relpath(os.path.abspath(current_output_path), self.output_dir)
                        else: 
                             relative_subfolder = os.path.basename(os.path.abspath(current_output_path))
                        if relative_subfolder == '.': relative_subfolder = ''
                    except ValueError: 
                        relative_subfolder = os.path.basename(os.path.abspath(current_output_path))
                    
                    preview_results_list.append({
                        "filename": final_file_name,
                        "subfolder": relative_subfolder.replace(os.sep, '/'),
                        "type": self.type
                    })
            except Exception as e:
                log_message(f"Failed to save {full_output_file_path}: {e}", "error")
            
            if overwrite_mode == "false": counter += 1
        
        ui_response = {"images": preview_results_list if show_previews_bool else []}
        if show_previews_bool: ui_response["files"] = output_files_list

        return {"ui": ui_response, "result": (images, output_files_list,)}
