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
                "operation": (["none", "remove_paths", "process_lora", "lowercase", "uppercase", "titlecase"],),
                "trim_from_end": ("STRING", {"default": "", "multiline": False, "label": "Trim From End"}),
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
        trim_from_end: str,
        string_list: Optional[Union[List[str], str]] = None,
        initial_text: str = ""
    ) -> tuple:
        """
        Process a list of strings into a single string with various operations.

        Args:
            delimiter: The string to use between items when joining
            operation: The operation to perform on strings (none, remove_paths, etc)
            trim_from_end: Substring to trim from the end of each string (if present)
            string_list: Input list of strings or a single string
            initial_text: Optional initial string to prepend
        Returns:
            A tuple containing the processed string
        """
        processed_items = []  # List to hold processed string items

        if string_list is not None:
            items = []
            # 1. Convert input to a list of strings, cleaning each item immediately
            if isinstance(string_list, str):
                cleaned_string = string_list.strip() # Initial clean
                if '\n' in cleaned_string:
                    # Split on newlines if present
                    items = [line.replace('\r', '').strip() for line in cleaned_string.split('\n') if line.strip()]
                elif ',' in cleaned_string:
                    # Split on commas if present
                    items = [item.replace('\r', '').strip() for item in cleaned_string.split(',') if item.strip()]
                elif re.search(r'(?<=>)\s*(?=<lora:)', cleaned_string):
                     # Split between lora blocks if they are separated by whitespace
                     items = [block.replace('\r', '').strip() for block in re.split(r'(?<=>)\s*(?=<lora:)', cleaned_string) if block.strip()]
                elif cleaned_string.count('<lora:') > 1 and '>' in cleaned_string:
                     # Split between adjacent lora blocks like <lora:...><lora:...>
                     items = [block.replace('\r', '').strip() for block in re.split(r'(?<=>)(?=<lora:)', cleaned_string) if block.strip()]
                else:
                    # Treat as a single string if no split pattern is found
                    single_item = cleaned_string.replace('\r', '').replace('\n', '').strip()
                    if single_item:
                       items = [single_item]
            elif isinstance(string_list, list):
                # If already a list, clean each item
                items = [str(i).replace('\r', '').replace('\n', '').strip() for i in string_list if str(i).strip()]

            # 2. Process each cleaned string item
            final_processed_items = [] # Use a new list for final items
            for item in items:
                if not item:
                    continue  # Skip empty items

                # Apply selected operation
                processed_item = item # Start with the cleaned item
                if operation == "remove_paths":
                    # Ensure basename works correctly even if there's no path
                    processed_item = os.path.basename(processed_item if '\\\\' in processed_item or '/' in processed_item else processed_item)
                elif operation == "process_lora":
                    # Corrected Lora processing: extract safetensors filename without duplication
                    match = re.search(r'([^\\/:]+)\\.safetensors', processed_item)
                    if match:
                        processed_item = match.group(1) + ".safetensors"
                    else:
                        # Fallback if pattern doesn't match exactly but .safetensors is present
                        base = os.path.basename(processed_item)
                        if ".safetensors" in base:
                             processed_item = base.split(":")[0] # Get part before first colon
                elif operation == "lowercase":
                    processed_item = processed_item.lower()
                elif operation == "uppercase":
                    processed_item = processed_item.upper()
                elif operation == "titlecase":
                    processed_item = processed_item.title()
                # 'none' operation does nothing to processed_item

                # 3. Trim from end AFTER the operation
                if trim_from_end and processed_item.endswith(trim_from_end):
                    processed_item = processed_item[: -len(trim_from_end)]

                # Final clean after all processing for this item
                processed_item = processed_item.strip()

                if processed_item: # Add only if not empty after processing
                    final_processed_items.append(processed_item) # Add to the final list

            # 4. Join all final processed items and prepend initial_text
            if final_processed_items:
                joined_items = delimiter.join(final_processed_items)
                # Prepend initial_text ONLY HERE
                if initial_text:
                    result = initial_text + delimiter + joined_items
                else:
                    result = joined_items
            else:
                # No processed items, return initial_text (may be empty)
                result = initial_text
        else:
            # No string_list provided, return initial_text (may be empty)
            result = initial_text
        return (result,)
