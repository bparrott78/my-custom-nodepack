import os
from datetime import date

class ProjectContextNode:
    """
    A node to build project directory structure and context information.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client_name": ("STRING", {"default": "", "multiline": False, "label": "Client name"}),
                "project_sku": ("STRING", {"default": "", "multiline": False, "label": "Project/SKU"}),
                "stage": (["POC", "PROD", "REV1", "FINAL"],),
                "branch": ("STRING", {"default": "", "multiline": False, "label": "Branch/Product type"}),
                "auto_date": (["enable", "disable"],),
                "base_path": ("STRING", {"default": "", "multiline": False, "label": "Base path"}),
            },
            "optional": {
                "due_date": ("STRING", {"default": "", "multiline": False, "label": "Due date (YYYY-MM-DD)"}),
            },
        }

    RETURN_TYPES = ("STRING", "CONDITIONING") # Assuming CONDITIONING is a valid type or placeholder
    RETURN_NAMES = ("folder_path", "context")
    FUNCTION = "execute"
    CATEGORY = "MyCustomNodePack/ProjectContext"

    def execute(self, client_name, project_sku, stage, branch, auto_date, base_path, due_date=None):
        # prepare date prefix
        today_str = date.today().isoformat() if auto_date == "enable" else ""
        # build path parts
        parts = [base_path, client_name, project_sku]
        # stage with optional due-date suffix
        stage_part = f"{stage}_due-{due_date}" if due_date else stage
        parts.append(stage_part)
        # branch folder, prefixed by date if enabled
        branch_part = f"{today_str}_{branch}" if auto_date == "enable" else branch
        parts.append(branch_part)
        folder_path = os.path.join(*parts)
        os.makedirs(folder_path, exist_ok=True)
        # compute due date info
        days_left = None
        past_due = False
        if due_date:
            try:
                due = date.fromisoformat(due_date)
                days_left = (due - date.today()).days
                past_due = days_left < 0
            except ValueError:
                days_left = None
        # assemble context dictionary
        context = {
            "client_name": client_name,
            "project_sku": project_sku,
            "stage": stage,
            "branch": branch,
            "auto_date": auto_date == "enable",
            "base_path": base_path,
            "due_date": due_date,
            "today": today_str, # Use the string version
            "days_left": days_left,
            "past_due": past_due,
            "folder_path": folder_path,
        }
        # Note: Returning a dictionary directly might not work for CONDITIONING.
        # This might need adjustment based on how ComfyUI handles CONDITIONING types.
        # For now, returning the dictionary as the second element.
        return (folder_path, context)
