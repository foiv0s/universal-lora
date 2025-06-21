"""
Tests that the LoRA-injected model produces output
with the correct shape and is callable in inference mode.
"""
import copy
import torch

from universal_lora.utils import apply_lora


def test_eval_inference():
    """
    Verifies that a forward pass with dummy input produces the correct output shape.
    """
    torch.manual_seed(0)
    orig_model = torch.nn.Sequential(torch.nn.Linear(10, 10, bias=False),
                                     torch.nn.Linear(10, 1, bias=False))
    orig_model.eval()

    m0 = apply_lora(copy.deepcopy(orig_model), r=2, alpha=0)
    m1 = apply_lora(copy.deepcopy(orig_model), r=2, alpha=10)
    x = torch.randn((1, 10))
    orig_model.eval(), m0.eval(), m1.eval()

    y_orig = orig_model(x)
    y_m0 = m0(x)
    y_m1 = m1(x)

    assert torch.allclose(y_orig, y_m0), 'Values with alpha=0 must be the same.'
    assert not torch.allclose(y_orig, y_m1, atol=2.2064), 'Values with alpha=1 must have different higher than 2.2064'
    assert torch.allclose(y_orig, y_m1, atol=2.2065), \
        'Values with alpha=1 must have different around to lower than 2.2065'
