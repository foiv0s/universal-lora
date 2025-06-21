"""
Test configuration file for pytest.
Defines shared fixtures used across multiple test modules.
"""

import copy
import timm
import pytest
from universal_lora.utils import apply_lora
from universal_lora.utils import add_lora_utils


@pytest.fixture(scope="module")
def resnet18_lora():
    """
    Fixture that returns a tuple of:
    - LoRA-wrapped ResNet18 model
    - Original ResNet18 model (before LoRA injection)

    The LoRA model is initialized with `r=4`, `alpha=4`, and utilities attached.
    """
    model = timm.create_model("resnet18", pretrained=False)
    orig_model = copy.deepcopy(model)
    orig_model.eval()
    lora_adapt_model = apply_lora(model, r=2, alpha=4)
    lora_adapt_model = add_lora_utils(lora_adapt_model)
    lora_adapt_model.eval()
    return lora_adapt_model, orig_model


@pytest.fixture(scope="module")
def vit_small_lora():
    """
    Fixture that returns a tuple of:
    - LoRA-wrapped vit_small model
    - Original vit_small model (before LoRA injection)

    The LoRA model is initialized with `r=4`, `alpha=4`, and utilities attached.
    """
    model = timm.create_model('vit_small_patch16_224', pretrained=False)
    orig_model = copy.deepcopy(model)
    orig_model.eval()
    lora_adapt_model = apply_lora(model, r=2, alpha=4)
    lora_adapt_model = add_lora_utils(lora_adapt_model)
    lora_adapt_model.eval()
    return lora_adapt_model, orig_model
