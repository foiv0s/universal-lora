from .layers.dense import LoRALinear, LoRAEmbedding
from .layers.conv import LoRAConv1d, LoRAConv2d, LoRAConv3d

import torch
from torch import nn


def add_lora_utils(model):
    """
    Add utility functions to a model with LoRA layers:
    - save_lora_weights(path): Saves only LoRA parameters to disk.
    - load_lora_weights(path): Loads LoRA parameters from a saved file.

    These functions allow storing and restoring only the trainable LoRA parameters
    without affecting the frozen base model, enabling efficient fine-tuning and deployment.

    Args:
        model (nn.Module): A model with LoRA layers injected via `apply_lora`.

    Returns:
        nn.Module: The same model with two new methods: `save_lora_weights` and `load_lora_weights`.
    """

    def save_lora_weights(path):
        """
        Save only the LoRA-specific parameters of the model to the given path.

        Args:
            path (str): Path to save the LoRA parameter state dictionary.
        """
        lora_state = {
            k: v for k, v in model.state_dict().items() if 'lora_' in k
        }
        torch.save(lora_state, path)

    def load_lora_weights(path, strict=False):
        """
        Load LoRA-specific parameters from a saved file and inject them into the model.

        Args:
            path (str): Path to a file containing LoRA weights (as saved by `save_lora_weights`).
            strict (bool): Whether to allow strict loading.
        """
        lora_state = torch.load(path)
        result = model.load_state_dict(lora_state, strict=False)
        missing_lora_keys = [key for key in result.missing_keys if 'lora_' in key]
        unexpected_lora_keys = [key for key in result.unexpected_keys if 'lora_' in key]
        # Report missing or unexpected keys
        if missing_lora_keys:
            message = f"Missing LoRA keys during LoRA loading: {missing_lora_keys}"
            if strict:
                raise ValueError(message)
            print(f'[WARN]-{message}')
        if unexpected_lora_keys:
            message = f'Unexpected LoRA keys during LoRA loading: {unexpected_lora_keys}'
            if strict:
                raise ValueError(message)
            print(f"[WARN]-{message}")

        if not missing_lora_keys and not unexpected_lora_keys:
            print("[INFO] All LoRA weights loaded successfully.")

    model.save_lora_weights = save_lora_weights
    model.load_lora_weights = load_lora_weights
    return model


def apply_lora(module, r=4, alpha=1.0, dropout=0.0):
    """
    Recursively apply LoRA to all supported layers inside the module.

    Args:
        module (nn.Module): The target model or submodule.
        r (int): Rank of the LoRA decomposition.
        alpha (float): Scaling factor for LoRA path.
        dropout (float): Dropout rate applied to LoRA path.

    Returns:
        nn.Module: The modified module with LoRA layers injected.
    """
    layer_map = {
        nn.Linear: LoRALinear,
        nn.Conv1d: LoRAConv1d,
        nn.Conv2d: LoRAConv2d,
        nn.Conv3d: LoRAConv3d,
        nn.Embedding: LoRAEmbedding
    }

    for name, child in module.named_children():
        # Recursively process child modules
        module.add_module(name, apply_lora(child, r, alpha, dropout))

        # Replace matching leaf layer
        for base_layer, lora_class in layer_map.items():
            if isinstance(child, base_layer):
                module.add_module(name, lora_class(child, r, alpha, dropout))
                break

    return add_lora_utils(module)
