from torch import nn


class LoRALayer(nn.Module):
    """
    Base class for LoRA modules.
    Provides a standard interface for future merge/unmerge logic.
    """

    def __init__(self):
        super().__init__()

    def merge(self):
        """Merge LoRA parameters into the original layer (not implemented)."""
        raise NotImplementedError

    def unmerge(self):
        """Unmerge LoRA parameters from the original layer (not implemented)."""
        raise NotImplementedError
