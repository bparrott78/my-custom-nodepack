# My Custom Node Pack for ComfyUI

A collection of custom nodes for ComfyUI to enhance your workflow.

## Installation

1.  Clone this repository or download the files.
2.  Place the `my_custom_nodepack` folder (the one containing `__init__.py`, `random_lora_stack.py`, etc.) into your `ComfyUI/custom_nodes/` directory.
3.  Restart ComfyUI.

## Nodes Included

This pack currently includes the following nodes:

### 1. Random LoRA Stack (`RandomLoraStack`)
   - **File:** `random_lora_stack.py`
   - **Category:** `MyCustomNodePack/LoRA`
   - **Description:** Automatically selects a specified number of LoRAs at random from your LoRA library based on a seed. It allows for an optional override to select a specific LoRA.
   - **Key Features:**
     - Random LoRA selection based on a seed for reproducibility.
     - Option to "Fix Selection" to keep the same random LoRAs across runs with the same seed.
     - Override specific LoRA selection.
     - Independent randomization of Model and Clip weights.
     - Configurable minimum and maximum ranges for randomized Model and Clip weights.
     - Option to use fixed Model and Clip weights if randomization is disabled for either.
     - Outputs a LORA_STACK and a string list of selected LoRA names with a configurable delimiter.

### 2. Dynamic LoRA Stack (`DynamicLoraStack`)
   - **File:** `dynamic_lora_stack.py`
   - **Category:** `MyCustomNodePack/LoRA`
   - **Description:** Provides a flexible way to stack multiple LoRAs with dynamically adjustable UI slots.
   - **Key Features:**
     - Configure the number of LoRA slots directly in the UI.
     - Maximum number of UI slots can be capped and is also determined by available LoRAs.
     - Seed input for deterministic random LoRA selection if slots are left to "Select Random".
     - Outputs a LORA_STACK and a string list of selected LoRA names with a configurable delimiter.
     - Cleans LoRA names for the output string (strips paths, keeps extensions).

### 3. GPT Image Generator (`GPTImageGenerator`)
   - **File:** `gpt_image.py`
   - **Category:** `MyCustomNodePack/Image`
   - **Description:** (Assuming functionality based on name) Likely generates or processes images using a GPT model or API. *You might want to add more specific details here.*

### 4. String List to String (`StringListToString`)
   - **File:** `string_utils.py`
   - **Category:** `MyCustomNodePack/Utils`
   - **Description:** A utility node for string manipulations.
   - **Key Features:**
     - Concatenates a list of strings into a single string.
     - Allows splitting a string at a specified character (e.g., ':').
     - *Add other specific operations if available.*

### 5. Pixel Art Grid Node (`PixelArtGridNode`)
   - **File:** `pixel_art_grid_node.py`
   - **Category:** `MyCustomNodePack/Image`
   - **Description:** (Assuming functionality based on name) Likely creates or processes images into a pixel art grid format. *You might want to add more specific details here.*

### 6. Project Context Node (`ProjectContextNode`)
   - **File:** `project_context_node.py`
   - **Category:** `MyCustomNodePack/Utils`
   - **Description:** (Assuming functionality based on name) Provides project-related context or information, possibly including dates or file paths. *You might want to add more specific details here.*

## Requirements
- ComfyUI
- Python 3.x
- (List any specific Python packages from `requirements.txt` if they are not commonly included with ComfyUI or Python standard library)

## Contributing
Contributions, issues, and feature requests are welcome. Please open an issue to discuss your ideas or report a bug.

## License
(Specify your license here, e.g., MIT License, Apache 2.0. If you have a `LICENSE` file, refer to it.)
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Quickstart

1. Install [ComfyUI](https://docs.comfy.org/get_started).
1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
1. Look up this extension in ComfyUI-Manager. If you are installing manually, clone this repository under `ComfyUI/custom_nodes`.
1. Restart ComfyUI.

# Features

- A list of features

## Develop

To install the dev dependencies and pre-commit (will run the ruff hook), do:

```bash
cd my_custom_nodepack
pip install -e .[dev]
pre-commit install
```

The `-e` flag above will result in a "live" install, in the sense that any changes you make to your node extension will automatically be picked up the next time you run ComfyUI.

## Publish to Github

Install Github Desktop or follow these [instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) for ssh.

1. Create a Github repository that matches the directory name. 
2. Push the files to Git
```
git add .
git commit -m "project scaffolding"
git push
``` 

## Writing custom nodes

An example custom node is located in [node.py](src/my_custom_nodepack/nodes.py). To learn more, read the [docs](https://docs.comfy.org/essentials/custom_node_overview).

## Tests

This repo contains unit tests written in Pytest in the `tests/` directory. It is recommended to unit test your custom node.

- [build-pipeline.yml](.github/workflows/build-pipeline.yml) will run pytest and linter on any open PRs
- [validate.yml](.github/workflows/validate.yml) will run [node-diff](https://github.com/Comfy-Org/node-diff) to check for breaking changes

## Publishing to Registry

If you wish to share this custom node with others in the community, you can publish it to the registry. We've already auto-populated some fields in `pyproject.toml` under `tool.comfy`, but please double-check that they are correct.

You need to make an account on https://registry.comfy.org and create an API key token.

- [ ] Go to the [registry](https://registry.comfy.org). Login and create a publisher id (everything after the `@` sign on your registry profile). 
- [ ] Add the publisher id into the pyproject.toml file.
- [ ] Create an api key on the Registry for publishing from Github. [Instructions](https://docs.comfy.org/registry/publishing#create-an-api-key-for-publishing).
- [ ] Add it to your Github Repository Secrets as `REGISTRY_ACCESS_TOKEN`.

A Github action will run on every git push. You can also run the Github action manually. Full instructions [here](https://docs.comfy.org/registry/publishing). Join our [discord](https://discord.com/invite/comfyorg) if you have any questions!

