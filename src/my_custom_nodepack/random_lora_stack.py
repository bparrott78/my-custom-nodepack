"""
Random LoRA Stack - Automatically selects LoRAs based on seed
"""
import folder_paths # type: ignore
import random
import os

class RandomLoraStack:
    """
    A LoRA stack node that automatically selects random LoRAs based on a seed.
    No manual selection required - just set how many LoRAs you want and a seed.
    """
    # Static variables for state persistence
    FIXED_SELECTION_DETAILS = None # Stores (seed, num_loras, list_of_loras)
    CURRENT_LORAS = None # Stores the currently selected LoRAs
    
    @classmethod
    def INPUT_TYPES(cls):
        try:
            # For the override_lora dropdown, always populate from default lora paths
            loras_for_override = [l for l in folder_paths.get_filename_list("loras") if l != "None"]
            loras_for_override = ["None"] + sorted(loras_for_override)
        except: # pylint: disable=bare-except
            loras_for_override = ["None"]
        
        required_inputs = {
            "num_loras": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1, "label": "Number of LoRAs"}),
            "load_from_custom_path": (["No", "Yes"], {"default": "No", "label": "Load from Custom Path"}),
            "lora_custom_path": ("STRING", {"default": "", "multiline": False, "label": "Custom LoRA Path (if Yes)"}),
            "text_delimiter": ("STRING", {"default": ", ", "multiline": False, "label": "Text Output Delimiter"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "selection_fix": ("BOOLEAN", {"default": False, "label": "Fix Selection"}),
            "override_lora": (loras_for_override, {"default": "None", "label": "Override Selection (from default)"}),
            "model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "label": "Fixed Model Weight"}),
            "clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "label": "Fixed Clip Weight"}),
            "randomize_model_weight": (["No", "Yes"], {"default": "No", "label": "Randomize Model Weight"}),
            "min_random_model_weight": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.001, "label": "Min Random Model Weight"}),
            "max_random_model_weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.001, "label": "Max Random Model Weight"}),
            "randomize_clip_weight": (["No", "Yes"], {"default": "No", "label": "Randomize Clip Weight"}),
            "min_random_clip_weight": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.001, "label": "Min Random Clip Weight"}),
            "max_random_clip_weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.001, "label": "Max Random Clip Weight"}),
        }
        
        return {
            "required": required_inputs,
            "optional": {"lora_stack": ("LORA_STACK",)}
        }
    
    RETURN_TYPES = ("LORA_STACK", "STRING")
    RETURN_NAMES = ("lora_stack", "lora_names")
    FUNCTION = "random_lora_stacker"
    CATEGORY = "MyCustomNodePack/LoRA"
    
    @classmethod
    def IS_CHANGED(cls, num_loras, load_from_custom_path, lora_custom_path, text_delimiter, seed, selection_fix, override_lora,
                   model_weight, clip_weight, 
                   randomize_model_weight, min_random_model_weight, max_random_model_weight,
                   randomize_clip_weight, min_random_clip_weight, max_random_clip_weight,
                   **kwargs):

        all_effective_inputs = (
            num_loras, load_from_custom_path, lora_custom_path, text_delimiter, seed, selection_fix, override_lora,
            model_weight, clip_weight,
            randomize_model_weight, min_random_model_weight, max_random_model_weight,
            randomize_clip_weight, min_random_clip_weight, max_random_clip_weight
        )
        current_hash = str(hash(all_effective_inputs))

        actual_lora_pool = []
        current_path_config_tuple = ("default", None) # (type, path_string)

        if load_from_custom_path == "Yes" and lora_custom_path and os.path.isdir(lora_custom_path):
            current_path_config_tuple = ("custom", lora_custom_path)
            try:
                lora_extensions = ['.safetensors', '.pt', '.ckpt', '.lora']
                actual_lora_pool = [
                    os.path.join(lora_custom_path, f) for f in os.listdir(lora_custom_path)
                    if os.path.isfile(os.path.join(lora_custom_path, f)) and \
                       any(f.lower().endswith(ext) for ext in lora_extensions)
                ]
            except Exception as e:
                print(f"[RandomLoraStack] Error loading LoRAs from custom path {lora_custom_path}: {e}")
                actual_lora_pool = []
        else:
            # Default behavior: load from ComfyUI's standard LoRA paths
            current_path_config_tuple = ("default", None)
            try:
                actual_lora_pool = [l for l in folder_paths.get_filename_list("loras") if l and l != "None"]
            except Exception as e:
                print(f"[RandomLoraStack] Error loading LoRAs from default paths: {e}")
                actual_lora_pool = []

        if override_lora != "None":
            # override_lora is a filename from the default list.
            # The LoRA loader should be able to resolve it.
            cls.CURRENT_LORAS = [override_lora] if override_lora else []
            cls.FIXED_SELECTION_DETAILS = None 
        elif not selection_fix:
            random.seed(seed)
            if not actual_lora_pool:
                cls.CURRENT_LORAS = []
            else:
                num_to_select = min(num_loras, len(actual_lora_pool))
                cls.CURRENT_LORAS = random.sample(actual_lora_pool, num_to_select) if num_to_select > 0 else []
            # Store details for potential future fixing, including path config
            cls.FIXED_SELECTION_DETAILS = (seed, num_loras, current_path_config_tuple, list(cls.CURRENT_LORAS or []))
        else: # selection_fix is True
            if (cls.FIXED_SELECTION_DETAILS is None or
                cls.FIXED_SELECTION_DETAILS[0] != seed or
                cls.FIXED_SELECTION_DETAILS[1] != num_loras or
                cls.FIXED_SELECTION_DETAILS[2] != current_path_config_tuple): # Check if LoRA source config changed
                
                random.seed(seed)
                if not actual_lora_pool:
                    cls.CURRENT_LORAS = []
                else:
                    num_to_select = min(num_loras, len(actual_lora_pool))
                    cls.CURRENT_LORAS = random.sample(actual_lora_pool, num_to_select) if num_to_select > 0 else []
                cls.FIXED_SELECTION_DETAILS = (seed, num_loras, current_path_config_tuple, list(cls.CURRENT_LORAS or []))
            else:
                # Path config, seed, and num_loras are the same, reuse fixed selection
                cls.CURRENT_LORAS = cls.FIXED_SELECTION_DETAILS[3]
        
        random.seed(seed) # Ensure seed is set for the main function's random weight operations
        return current_hash
    
    def random_lora_stacker(
        self,
        num_loras, load_from_custom_path, lora_custom_path, text_delimiter, seed, selection_fix, override_lora,
        model_weight, clip_weight, 
        randomize_model_weight, min_random_model_weight, max_random_model_weight,
        randomize_clip_weight, min_random_clip_weight, max_random_clip_weight,
        lora_stack=None 
    ):
        lora_list = []
        lora_names_list = []
        
        if lora_stack is not None:
            for l_name, l_model_w, l_clip_w in lora_stack:
                if l_name and l_name != "None":
                    lora_list.append((l_name, l_model_w, l_clip_w))
                    try:
                        lora_names_list.append(os.path.basename(str(l_name)))
                    except Exception:
                        lora_names_list.append(str(l_name))

        # self.CURRENT_LORAS is set by IS_CHANGED.
        # random.seed(seed) was called at the end of IS_CHANGED.
        if self.CURRENT_LORAS:
            for lora_name_or_path in self.CURRENT_LORAS: # Renamed for clarity
                if not lora_name_or_path or lora_name_or_path == "None":
                    continue

                actual_model_weight = model_weight 
                if randomize_model_weight == "Yes":
                    eff_min_model = min(min_random_model_weight, max_random_model_weight)
                    eff_max_model = max(min_random_model_weight, max_random_model_weight)
                    if eff_min_model == eff_max_model:
                        actual_model_weight = eff_min_model
                    else:
                        actual_model_weight = random.uniform(eff_min_model, eff_max_model)

                actual_clip_weight = clip_weight 
                if randomize_clip_weight == "Yes":
                    eff_min_clip = min(min_random_clip_weight, max_random_clip_weight)
                    eff_max_clip = max(min_random_clip_weight, max_random_clip_weight)
                    if eff_min_clip == eff_max_clip:
                        actual_clip_weight = eff_min_clip
                    else:
                        actual_clip_weight = random.uniform(eff_min_clip, eff_max_clip)
                
                try:
                    # os.path.basename works for both full paths and simple filenames
                    clean_name = os.path.basename(str(lora_name_or_path))
                except Exception:
                    clean_name = str(lora_name_or_path)
                    
                lora_list.append((
                    lora_name_or_path, # This will be full path if custom, filename if default
                    actual_model_weight,
                    actual_clip_weight
                ))
                lora_names_list.append(clean_name)
        
        lora_names_text = text_delimiter.join(lora_names_list)
        
        return (lora_list, lora_names_text)

# Register the node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "RandomLoraStack": RandomLoraStack
}

# Ensure NODE_DISPLAY_NAME_MAPPINGS is present if you want custom display names
# NODE_DISPLAY_NAME_MAPPINGS = {
# "RandomLoraStack": "Random LoRA Stack 🎲"
# }