"""
String Utility Nodes for ComfyUI
"""
import os
import re
from typing import List, Optional, Union

class StringListToString:
    """
    A node that processes a list of strings into a single string.
    Supports operations like joining, removing pathnames, etc.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "delimiter": ("STRING", {"default": "_", "multiline": False}),
                "operation": (["none", "remove_paths", "process_lora"],),
                "split_at": ("STRING", {"default": ":", "multiline": False, "label": "Split At Character"}),
            },
            "optional": {
                "string_list": ("STRING", {"forceInput": True}),  # Force input from another node
                "initial_text": ("STRING", {"default": "", "multiline": True}),  # Optional initial string
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_string",)
    FUNCTION = "process"
    CATEGORY = "MyCustomNodePack/Utils"
    
    def process(
        self,
        delimiter: str,
        operation: str,
        split_at: str,
        string_list: Optional[Union[List[str], str]] = None,
        initial_text: str = ""
    ) -> tuple:
        """
        Process a list of strings into a single string with various operations.

        Args:
            delimiter: The string to use between items when joining
            operation: The operation to perform on strings (none, remove_paths, etc)
            split_at: Character to split strings at (keeps everything before this character)
            string_list: Input list of strings or a single string
            initial_text: Optional initial string to prepend
        Returns:
            A tuple containing the processed string
        """
        final_processed_items = []  # List to hold final processed items

        if string_list is not None:
            items = []
            # 1. Convert input to a list of strings
            if isinstance(string_list, str):
                cleaned_string = string_list.strip()
                
                # Check for lora blocks first
                if '<lora:' in cleaned_string and '>' in cleaned_string:
                    items = re.split(r'(?<=>)\s*(?=<lora:|$)', cleaned_string)
                # Otherwise try standard delimiters
                elif '\n' in cleaned_string:
                    items = cleaned_string.split('\n')
                elif ',' in cleaned_string:
                    items = cleaned_string.split(',')
                else:
                    items = [cleaned_string]
                
                # Clean all items
                items = [item.replace('\r', '').replace('\n', '').strip() for item in items if item.strip()]
            elif isinstance(string_list, list):
                items = [str(i).replace('\r', '').replace('\n', '').strip() for i in string_list if str(i).strip()]

            # 2. Process each cleaned string item
            for item in items:
                if not item:
                    continue
                
                # Apply selected operation
                processed_item = item
                if operation == "remove_paths":
                    processed_item = os.path.basename(processed_item)
                elif operation == "process_lora":
                    match = re.search(r'([^\\/:]+)\.safetensors', processed_item)
                    if match:
                        processed_item = match.group(1) + ".safetensors"
                    else:
                        base = os.path.basename(processed_item)
                        if ".safetensors" in base:
                            processed_item = base.split(":")[0]
                
                # Apply split_at logic - keep only the part before the character
                if split_at and split_at in processed_item:
                    processed_item = processed_item.split(split_at)[0]
                
                # Final clean
                processed_item = processed_item.strip()
                
                if processed_item:
                    final_processed_items.append(processed_item)

            # 3. Join all processed items and prepend initial_text
            if final_processed_items:
                joined_items = delimiter.join(final_processed_items)
                if initial_text:
                    result = initial_text + delimiter + joined_items
                else:
                    result = joined_items
            else:
                result = initial_text
        else:
            result = initial_text
            
        return (result,)
