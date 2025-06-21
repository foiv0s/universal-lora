"""
Test suite for verifying LoRA layer replacement and original weight integrity.
Includes:
- Layer replacement check
- Original weight copy check
- Freezing check for base weights
"""

import torch
import torch.nn as nn

from universal_lora.layers.dense import LoRALinear, LoRAEmbedding
from universal_lora.layers.conv import LoRAConv1d, LoRAConv2d, LoRAConv3d


def all_params_equal(orig_child, lora_child, atol=0):
    """
    Compare parameters between an original layer and a LoRA-wrapped version.

    This function checks whether all parameters in the `orig_child` layer
    are numerically equal to the corresponding `orig.`-prefixed parameters
    inside the `lora_child` state_dict.

    Args:
        orig_child (nn.Module): The original, unmodified PyTorch layer.
        lora_child (nn.Module): The LoRA-wrapped version of the same layer.
        atol (float): Absolute tolerance for floating point comparison.

    Returns:
        bool: True if all corresponding weights match within the given tolerance, False otherwise.
    """
    sd1 = lora_child.state_dict()
    sd2 = orig_child.state_dict()
    lora_orig_keys = {k.replace('orig.', ''): k for k in sd1.keys() if 'orig.' in k}
    if lora_orig_keys.keys() != sd2.keys():
        print("Different keys in state_dict.")
        return False

    for k in sd2:
        if not torch.allclose(sd1[lora_orig_keys[k]], sd2[k], atol=atol):
            print(f"Mismatch in parameter: {k}")
            return False

    return True


def lora_replacement_and_freezing(lora_adapt, orig_model):
    """
        Recursively verifies that all expected layers in a model were correctly replaced by LoRA-wrapped versions.

        This function traverses two models in parallel (LoRA-adapted and original), and for each supported
        layer type it checks:
        - That the original layer is replaced by the correct LoRA class.
        - That the original parameters were copied correctly.
        - That the original parameters are frozen (`requires_grad = False`).

        Args:
            lora_adapt (nn.Module): Model with LoRA layers injected.
            orig_model (nn.Module): The original baseline model without LoRA.

        Raises:
            AssertionError: If a layer type was not replaced properly, weights differ, or are not frozen.
        """
    layer_map = {
        nn.Linear: LoRALinear,
        nn.Conv1d: LoRAConv1d,
        nn.Conv2d: LoRAConv2d,
        nn.Conv3d: LoRAConv3d,
        nn.Embedding: LoRAEmbedding
    }

    for (orig_name, orig_child), (lora_name, lora_child) \
            in zip(orig_model.named_children(), lora_adapt.named_children()):
        # Recursively process child modules
        lora_replacement_and_freezing(lora_child, orig_child)

        # Replace matching leaf layer
        for base_layer, lora_class in layer_map.items():
            if isinstance(orig_child, base_layer):
                assert isinstance(lora_child, layer_map[base_layer]), "Incorrect adapted LoRA layer"
                assert all_params_equal(orig_child, lora_child), "Mismatch in original weights and LoRA layer"
                # Check that original weights are frozen
                for param in lora_child.orig.parameters():
                    assert not param.requires_grad, f"{lora_name}.{param} is not frozen"
                break


def test_resnet18_lora(resnet18_lora):
    '''Test the application of LoRA to resnet18'''
    lora_replacement_and_freezing(*resnet18_lora)


def test_vit_small_lora(vit_small_lora):
    '''Test the application of LoRA to vit_small'''
    lora_replacement_and_freezing(*vit_small_lora)
