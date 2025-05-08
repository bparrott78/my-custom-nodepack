"""
Random LoRA Stack - Automatically selects LoRAs based on seed
"""
import folder_paths # type: ignore
import random
import re
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
            # Try to get the list of all available loras
            loras = [l for l in folder_paths.get_filename_list("loras") if l != "None"]
            loras = ["None"] + sorted(loras)  # Add None option at the top for override
        except: # pylint: disable=bare-except
            # Fallback if folder_paths is not available
            loras = ["None"]
        
        required_inputs = {
            "num_loras": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1, "label": "Number of LoRAs"}),
            "text_delimiter": ("STRING", {"default": ", ", "multiline": False, "label": "Text Output Delimiter"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "selection_fix": ("BOOLEAN", {"default": False, "label": "Fix Selection"}),
            "override_lora": (loras, {"default": "None", "label": "Override Selection"}),
            "model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "label": "Default Model Weight"}),
            "clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "label": "Default Clip Weight"}),
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
    def IS_CHANGED(cls, num_loras, text_delimiter, seed, selection_fix, override_lora,
                   model_weight, clip_weight, # These are fixed weights
                   randomize_model_weight, min_random_model_weight, max_random_model_weight,
                   randomize_clip_weight, min_random_clip_weight, max_random_clip_weight,
                   **kwargs): # kwargs should be empty if all inputs defined in INPUT_TYPES are listed

        all_effective_inputs = (
            num_loras, text_delimiter, seed, selection_fix, override_lora,
            model_weight, clip_weight,
            randomize_model_weight, min_random_model_weight, max_random_model_weight,
            randomize_clip_weight, min_random_clip_weight, max_random_clip_weight
        )
        current_hash = str(hash(all_effective_inputs))

        if override_lora != "None":
            cls.CURRENT_LORAS = [override_lora] if override_lora else []
            cls.FIXED_SELECTION_DETAILS = None 
        elif not selection_fix:
            random.seed(seed)
            try:
                all_available_loras = [l for l in folder_paths.get_filename_list("loras") if l and l != "None"]
                if not all_available_loras:
                    cls.CURRENT_LORAS = []
                else:
                    num_to_select = min(num_loras, len(all_available_loras))
                    cls.CURRENT_LORAS = random.sample(all_available_loras, num_to_select) if num_to_select > 0 else []
            except Exception: 
                cls.CURRENT_LORAS = []
            cls.FIXED_SELECTION_DETAILS = (seed, num_loras, list(cls.CURRENT_LORAS or []))
        else: # selection_fix is True
            if cls.FIXED_SELECTION_DETAILS is None or \
               cls.FIXED_SELECTION_DETAILS[0] != seed or \
               cls.FIXED_SELECTION_DETAILS[1] != num_loras:
                random.seed(seed)
                try:
                    all_available_loras = [l for l in folder_paths.get_filename_list("loras") if l and l != "None"]
                    if not all_available_loras:
                        cls.CURRENT_LORAS = []
                    else:
                        num_to_select = min(num_loras, len(all_available_loras))
                        cls.CURRENT_LORAS = random.sample(all_available_loras, num_to_select) if num_to_select > 0 else []
                except Exception:
                    cls.CURRENT_LORAS = []
                cls.FIXED_SELECTION_DETAILS = (seed, num_loras, list(cls.CURRENT_LORAS or []))
            else:
                cls.CURRENT_LORAS = cls.FIXED_SELECTION_DETAILS[2]
        
        random.seed(seed) # Ensure seed is set for the main function's random operations
        return current_hash
    
    def random_lora_stacker(
        self,
        num_loras, text_delimiter, seed, selection_fix, override_lora,
        model_weight, clip_weight, # Fixed/default weights from INPUT_TYPES
        randomize_model_weight, min_random_model_weight, max_random_model_weight,
        randomize_clip_weight, min_random_clip_weight, max_random_clip_weight,
        lora_stack=None # Optional input
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
            for lora_name in self.CURRENT_LORAS:
                if not lora_name or lora_name == "None":
                    continue

                actual_model_weight = model_weight # Default to fixed weight
                if randomize_model_weight == "Yes":
                    eff_min_model = min(min_random_model_weight, max_random_model_weight)
                    eff_max_model = max(min_random_model_weight, max_random_model_weight)
                    if eff_min_model == eff_max_model:
                        actual_model_weight = eff_min_model
                    else:
                        actual_model_weight = random.uniform(eff_min_model, eff_max_model)

                actual_clip_weight = clip_weight # Default to fixed weight
                if randomize_clip_weight == "Yes":
                    eff_min_clip = min(min_random_clip_weight, max_random_clip_weight)
                    eff_max_clip = max(min_random_clip_weight, max_random_clip_weight)
                    if eff_min_clip == eff_max_clip:
                        actual_clip_weight = eff_min_clip
                    else:
                        actual_clip_weight = random.uniform(eff_min_clip, eff_max_clip)
                
                try:
                    clean_name = os.path.basename(str(lora_name))
                except Exception:
                    clean_name = str(lora_name)
                    
                lora_list.append((
                    lora_name,
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