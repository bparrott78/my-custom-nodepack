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
    FIXED_SELECTION = None
    CURRENT_LORAS = None
    
    @classmethod
    def INPUT_TYPES(cls):
        try:
            # Try to get the list of all available loras
            loras = [l for l in folder_paths.get_filename_list("loras") if l != "None"]
            loras = ["None"] + sorted(loras)  # Add None option at the top for override
        except:
            # Fallback if folder_paths is not available
            loras = ["None"]
        
        required_inputs = {
            "num_loras": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1, "label": "Number of LoRAs"}),
            "text_delimiter": ("STRING", {"default": ", ", "multiline": False, "label": "Text Output Delimiter"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "selection_fix": ("BOOLEAN", {"default": False, "label": "Fix Selection"}),
            "override_lora": (loras, {"default": "None", "label": "Override Selection"}),
            "model_weight": ("FLOAT", {"default": 0.3, "min": -10.0, "max": 10.0, "step": 0.01}),
            "clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
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
    def IS_CHANGED(cls, num_loras, text_delimiter, seed, selection_fix, override_lora, model_weight, clip_weight, **kwargs):
        """Determine if the node needs to update its output"""
        # Check for override first
        if override_lora != "None":
            cls.CURRENT_LORAS = [override_lora]
            cls.FIXED_SELECTION = f"override_{override_lora}"
            return cls.FIXED_SELECTION
            
        # Use the provided seed for deterministic randomization
        if not selection_fix or cls.FIXED_SELECTION is None:
            random.seed(seed)
            cls.FIXED_SELECTION = seed
            # Get all available loras
            try:
                all_loras = [l for l in folder_paths.get_filename_list("loras") if l != "None"]
                if not all_loras:
                    cls.CURRENT_LORAS = []
                    return str(seed)
                    
                # Select random loras based on the seed
                num_to_select = min(num_loras, len(all_loras))
                selected_loras = random.sample(all_loras, num_to_select)
                cls.CURRENT_LORAS = selected_loras
            except:
                cls.CURRENT_LORAS = []
            return str(seed)
        return str(cls.FIXED_SELECTION)
    
    def random_lora_stacker(
        self,
        num_loras,
        text_delimiter,
        seed,
        selection_fix,
        override_lora,
        model_weight,
        clip_weight,
        **kwargs
    ):
        """Main node function to generate the lora stack"""
        lora_list = []
        lora_names_list = []  # To collect LoRA names for text output
        
        # Include existing lora stack if provided
        lora_stack = kwargs.get('lora_stack', None)
        if lora_stack is not None:
            for l in lora_stack:
                if l[0] != "None":
                    lora_list.append(l)
                    lora_names_list.append(os.path.basename(l[0]))  # Add clean name to text list
        
        # Use the selected loras if available
        if self.CURRENT_LORAS:
            for lora_name in self.CURRENT_LORAS:
                clean_name = os.path.basename(lora_name)
                lora_list.append((
                    lora_name,  # Keep full path for ComfyUI
                    model_weight,
                    clip_weight
                ))
                lora_names_list.append(clean_name)  # Use clean name for display
        
        # Join the lora names with the specified delimiter
        lora_names_text = text_delimiter.join(lora_names_list)
        
        return (lora_list, lora_names_text)

# Register the node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "RandomLoraStack": RandomLoraStack
}