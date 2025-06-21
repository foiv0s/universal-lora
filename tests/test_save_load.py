"""
Tests for saving and loading LoRA weights, ensuring:
- Parameters are correctly serialized
- Parameters can be restored into another LoRA-injected model
"""

import tempfile, torch, copy
from universal_lora.utils import apply_lora


def save_and_load(lora_model, orig_model):
    """
    Saves LoRA weights, loads them into a new model, and checks parameter equality.
    """
    x = torch.randn(1, 3, 224, 224)
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/lora.pth"
        lora_model.save_lora_weights(path)

        y1 = lora_model(x)

        # fresh baseline model + LoRA
        m2 = apply_lora(copy.deepcopy(orig_model), r=2, alpha=4)
        m2.load_lora_weights(path)

        y2 = m2(x)
        assert torch.allclose(y1, y2), "Loaded LoRA weights give different output"


def test_resnet18_lora(resnet18_lora):
    '''Save and load resnet18 lora model'''
    save_and_load(*resnet18_lora)


def test_vit_small_lora(vit_small_lora):
    '''Save and load vit_small lora model'''
    save_and_load(*vit_small_lora)
