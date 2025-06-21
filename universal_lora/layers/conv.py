from ..layers.default_layer import LoRALayer
from torch import nn


class LoRAConv1d(LoRALayer):
    """
    LoRA wrapper for nn.Conv1d layers using 1x1 convolutions as adaptation.
    """

    def __init__(self, orig_layer, r=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.orig = orig_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Conv1d(orig_layer.in_channels, r, kernel_size=1, stride=1, padding=0, bias=False)
        self.lora_B = nn.Conv1d(r, orig_layer.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.dropout = nn.Dropout(dropout)

        for param in self.orig.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.orig(x) + self.dropout(self.lora_B(self.lora_A(x))) * self.scaling


class LoRAConv2d(LoRALayer):
    """
    LoRA wrapper for nn.Conv2d layers using 1x1 convolutions as adaptation.
    Maintains spatial alignment regardless of the original kernel/stride.
    """

    def __init__(self, orig_layer, r=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.orig = orig_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Conv2d(
            orig_layer.in_channels, r,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.lora_B = nn.Conv2d(
            r, orig_layer.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.dropout = nn.Dropout(dropout)

        for param in self.orig.parameters():
            param.requires_grad = False

    def forward(self, x):
        orig_out = self.orig(x)
        lora_out = self.lora_B(self.lora_A(x))
        if lora_out.shape != orig_out.shape:
            # Resize LoRA output to match the original output spatially
            lora_out = nn.functional.interpolate(lora_out, size=orig_out.shape[2:], mode='bilinear',
                                                 align_corners=False)
        return orig_out + self.dropout(lora_out) * self.scaling


class LoRAConv3d(LoRALayer):
    """
    LoRA wrapper for nn.Conv3d layers using 1x1x1 convolutions as adaptation.
    Maintains spatial alignment regardless of original kernel/stride.
    """

    def __init__(self, orig_layer, r=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.orig = orig_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Conv3d(
            orig_layer.in_channels, r,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.lora_B = nn.Conv3d(
            r, orig_layer.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.dropout = nn.Dropout(dropout)

        for param in self.orig.parameters():
            param.requires_grad = False

    def forward(self, x):
        orig_out = self.orig(x)
        lora_out = self.lora_B(self.lora_A(x))
        if lora_out.shape != orig_out.shape:
            lora_out = nn.functional.interpolate(lora_out, size=orig_out.shape[2:], mode='trilinear',
                                                 align_corners=False)
        return orig_out + self.dropout(lora_out) * self.scaling
