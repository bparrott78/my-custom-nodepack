# Copilot Instructions for My Custom Node Pack (ComfyUI)

This guide enables AI coding agents to work productively in this codebase. It summarizes architecture, workflows, and conventions specific to this project.

## Architecture Overview
- **Purpose:** Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), extending image and LoRA workflows.
- **Main Components:**
  - `src/my_custom_nodepack/` — Core node implementations. Each file is a node (e.g., `random_lora_stack.py`, `dynamic_lora_stack.py`, `gpt_image.py`).
  - `js/` — JavaScript assets for UI integration (rarely edited).
  - `tests/` — Pytest-based unit tests for nodes.
- **Node Registration:** All nodes are registered via their Python files and exposed to ComfyUI. See `nodes.py` for examples.

## Developer Workflows
- **Install for Development:**
  ```bash
  pip install -e .[dev]
  pre-commit install
  ```
  - The `-e` flag enables live reload of node changes.
- **Testing:**
  - Run all tests: `pytest tests/`
  - Tests are in `tests/test_my_custom_nodepack.py` and use Pytest conventions.
- **Linting:**
  - Pre-commit runs [ruff](https://github.com/charliermarsh/ruff) for linting.
- **Build/CI:**
  - GitHub Actions run tests and lint on PRs (`.github/workflows/build-pipeline.yml`).
  - Node compatibility is checked via `validate.yml` using [node-diff](https://github.com/Comfy-Org/node-diff).

## Project-Specific Patterns
- **Node Structure:**
  - Each node is a Python class/function with a unique category and output type.
  - LoRA nodes (`random_lora_stack.py`, `dynamic_lora_stack.py`) use seeds for reproducibility and support UI slot configuration.
  - Utility nodes (e.g., `string_utils.py`) provide string manipulation helpers.
- **Output Conventions:**
  - Nodes output both data objects (e.g., `LORA_STACK`) and human-readable string lists (with configurable delimiters).
- **Integration:**
  - Nodes interact with ComfyUI via defined interfaces; see [docs](https://docs.comfy.org/essentials/custom_node_overview).
  - Registry publishing uses fields in `pyproject.toml` under `[tool.comfy]`.

## External Dependencies
- **ComfyUI** (required)
- **Python 3.x**
- Additional packages listed in `src/my_custom_nodepack/requirements.txt` (if any)

## Key Files & Directories
- `src/my_custom_nodepack/` — Node source code
- `tests/` — Pytest tests
- `README.md` — Node descriptions, install/dev instructions
- `pyproject.toml` — Metadata for registry and packaging
- `.github/workflows/` — CI/CD pipelines

## Example: Adding a Node
- Place new node in `src/my_custom_nodepack/`.
- Register in `nodes.py`.
- Add tests in `tests/`.
- Document in `README.md`.

---
For unclear conventions or missing details, ask the user for clarification or check the [ComfyUI docs](https://docs.comfy.org/essentials/custom_node_overview).
