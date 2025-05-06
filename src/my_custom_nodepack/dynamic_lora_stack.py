"""
Dynamic LoRA Stack Node - A more efficient version of LoraStack with configurable slot count
"""
import folder_paths # type: ignore
import random
import re
import os

class DynamicLoraStack:
    """
    A more efficient LoRA stack node that dynamically generates slots based on user selection
    and uses a seed for deterministic randomization.
    """
    # Static variables for state persistence
    FIXED_SELECTION = None
    UsedLorasMap = {}
    StridesMap = {}
    LastHashMap = {}
    
    # Pre-compile regex pattern for better performance
    LORA_NAME_PATTERN = re.compile(r'DynamicLoraStack_\d+')
    
    @classmethod
    def INPUT_TYPES(cls):
        try:
            # Try to get the list of all available loras
            loras = ["None"] + folder_paths.get_filename_list("loras")
            # Calculate max slots based on available loras - never less than 20 slots for flexibility
            max_slots = max(len(loras), 20)
        except:
            # Fallback if folder_paths is not available
            loras = ["None"]
            max_slots = 20
        
        required_inputs = {
            "max_active_slots": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1, "label": "Max Active LoRAs"}),
            "num_slots": ("INT", {"default": 10, "min": 1, "max": max_slots, "step": 1, "label": "UI Slots"}),
            "text_delimiter": ("STRING", {"default": ", ", "multiline": False, "label": "Text Output Delimiter"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "selection_fix": ("BOOLEAN", {"default": False, "label": "Fix Selection"}),
            "exclusive_mode": (["Off", "On"],),
            "stride": (("INT", {"default": 1, "min": 1, "max": 100})),
            "force_randomize_after_stride": (["Off", "On"],),
            "model_weight": ("FLOAT", {"default": 0.3, "min": -10.0, "max": 10.0, "step": 0.01}),
            "clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
        }
        
        # Create slots based on the number of loras available (with some reasonable cap)
        display_slots = min(max_slots, 20)  # Start with 20 visible slots max for UI usability
        for i in range(1, display_slots + 1):
            required_inputs[f"lora_name_{i}"] = (loras,)
            required_inputs[f"switch_{i}"] = (["Off", "On"],)
        
        return {
            "required": required_inputs,
            "optional": {"lora_stack": ("LORA_STACK",)}
        }
    
    RETURN_TYPES = ("LORA_STACK", "STRING")
    RETURN_NAMES = ("lora_stack", "lora_names")
    FUNCTION = "random_lora_stacker"
    CATEGORY = "MyCustomNodePack/LoRA"
    
    @staticmethod
    def getIdHash(*lora_names) -> int:
        """Generate a hash from a set of lora names - used for tracking"""
        return hash(frozenset(set(lora_names)))
    
    @staticmethod
    def deduplicateLoraNames(*lora_names):
        """Ensure unique names by appending a counter to duplicates"""
        lora_names = list(lora_names)
        name_counts = {}
        for i, name in enumerate(lora_names):
            if name != "None":
                count = name_counts.get(name, 0)
                if count > 0:
                    # Add a suffix to duplicates to make them unique
                    lora_names[i] = f"{name}DynamicLoraStack_{count+1}"
                name_counts[name] = count + 1
        return lora_names
    
    @staticmethod
    def cleanLoraName(lora_name) -> str:
        """Remove the uniqueness suffix and strip path, but keep extension"""
        # First remove any internal suffix we added
        name = DynamicLoraStack.LORA_NAME_PATTERN.sub('', lora_name)
        # Strip path but keep filename with extension
        name = os.path.basename(name)
        return name
    
    @classmethod
    def IS_CHANGED(cls, max_active_slots, num_slots, text_delimiter, seed, selection_fix, exclusive_mode, stride, 
                  force_randomize_after_stride, model_weight, clip_weight, **kwargs):
        """Determine if the node needs to update its output"""
        # Use the provided seed for deterministic randomization
        random.seed(seed)
        
        if not selection_fix or cls.FIXED_SELECTION is None:
            new_selection = cls._generate_new_selection(max_active_slots, num_slots,
                                                       exclusive_mode, 
                                                       stride, 
                                                       force_randomize_after_stride, 
                                                       **kwargs)
            cls.FIXED_SELECTION = new_selection
            return new_selection
        return cls.FIXED_SELECTION
    
    @classmethod
    def _generate_new_selection(cls, max_active_slots, num_slots, exclusive_mode, stride, force_randomize_after_stride, **kwargs):
        """Generate a new selection of loras based on the current settings"""
        # Only process active slots (based on UI slots parameter)
        lora_names = []
        switches = []
        chances = []
        
        for i in range(1, num_slots + 1):
            lora_name_key = f"lora_name_{i}"
            switch_key = f"switch_{i}"
            if lora_name_key in kwargs and switch_key in kwargs:
                lora_name = kwargs.get(lora_name_key)
                switch = kwargs.get(switch_key)
                lora_names.append(lora_name)
                switches.append(switch)
                chances.append(1.0)  # Default chance
        
        # Calculate active loras
        total_on = sum(
            1 for name, sw in zip(lora_names, switches)
            if name != "None" and sw == "On"
        )
        
        # Deduplicate names to handle repeats
        lora_names = cls.deduplicateLoraNames(*lora_names)
        id_hash = cls.getIdHash(*lora_names)
        
        # Handle stride logic (reuse previous selection for N iterations)
        if id_hash not in cls.StridesMap:
            cls.StridesMap[id_hash] = 0
        cls.StridesMap[id_hash] += 1
        
        if stride > 1 and cls.StridesMap[id_hash] < stride and id_hash in cls.LastHashMap:
            return cls.LastHashMap[id_hash]
        else:
            cls.StridesMap[id_hash] = 0
        
        def perform_randomization() -> set:
            """Randomize which loras to activate"""
            _lora_set = set()
            random_values = [random.random() for _ in range(len(lora_names))]
            applies = [(random_values[i] <= chances[i]) and (switches[i] == "On") for i in range(len(lora_names))]
            
            indices = [i for i, apply in enumerate(applies) if apply]
            
            # Apply exclusive mode (only the lora with lowest random value is chosen)
            if exclusive_mode == "On" and len(indices) > 1:
                min_index = min(indices, key=lambda idx: random_values[idx])
                applies = [False] * len(lora_names)
                applies[min_index] = True
            else:
                # Limit to max_active_slots loras if more are selected (to prevent overload)
                if len(indices) > max_active_slots:
                    selected_indices = sorted(indices, key=lambda i: random_values[i])[:max_active_slots]
                    applies = [i in selected_indices for i in range(len(lora_names))]
            
            # Collect active loras
            for i in range(len(lora_names)):
                if lora_names[i] != "None" and applies[i]:
                    _lora_set.add(lora_names[i])
            
            return _lora_set
        
        # Get previous set of loras
        last_lora_set = cls.UsedLorasMap.get(id_hash, set())
        lora_set = perform_randomization()
        
        # Force new randomization if current equals previous and forcing is enabled
        if force_randomize_after_stride == "On" and len(last_lora_set) > 0 and total_on > 1:
            max_attempts = 10  # Avoid infinite loop
            attempts = 0
            while lora_set == last_lora_set and attempts < max_attempts:
                lora_set = perform_randomization()
                attempts += 1
        
        # Store results for next time
        cls.UsedLorasMap[id_hash] = lora_set
        hash_str = str(hash(frozenset(lora_set)))
        cls.LastHashMap[id_hash] = hash_str
        
        return hash_str
    
    def random_lora_stacker(
        self,
        max_active_slots,
        num_slots,
        text_delimiter,
        seed,
        selection_fix,
        exclusive_mode,
        stride,
        force_randomize_after_stride,
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
        
        # Only process the slots specified by num_slots parameter
        lora_names = []
        switches = []
        
        for i in range(1, num_slots + 1):
            lora_name_key = f"lora_name_{i}"
            switch_key = f"switch_{i}"
            if lora_name_key in kwargs and switch_key in kwargs:
                lora_name = kwargs.get(lora_name_key)
                switch = kwargs.get(switch_key)
                lora_names.append(lora_name)
                switches.append(switch)
        
        # Deduplicate names for consistent tracking
        lora_names = self.deduplicateLoraNames(*lora_names)
        id_hash = self.getIdHash(*lora_names)
        
        # Get previously selected loras
        used_loras = self.UsedLorasMap.get(id_hash, set())
        
        # Add active loras to the stack
        for i in range(len(lora_names)):
            if (
                lora_names[i] != "None"
                and switches[i] == "On"
                and lora_names[i] in used_loras
            ):
                clean_name = self.cleanLoraName(lora_names[i])
                lora_list.append((
                    lora_names[i],  # Keep full path for ComfyUI
                    model_weight,
                    clip_weight
                ))
                lora_names_list.append(clean_name)  # Use clean name for display
        
        # Join the lora names with the specified delimiter
        lora_names_text = text_delimiter.join(lora_names_list)
        
        return (lora_list, lora_names_text)

# Register the node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "DynamicLoraStack": DynamicLoraStack
}