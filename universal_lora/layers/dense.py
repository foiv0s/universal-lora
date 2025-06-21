from ..layers.default_layer import LoRALayer
from torch import nn


class LoRALinear(LoRALayer):
    """
    LoRA wrapper for nn.Linear layers.
    Adds a low-rank adaptation path to the original linear transformation.
    """

    def __init__(self, orig_layer, r=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.orig = orig_layer
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(orig_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, orig_layer.out_features, bias=False)
        self.scaling = alpha / r

        for param in self.orig.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.orig(x) + self.dropout(self.lora_B(self.lora_A(x))) * self.scaling


class LoRAEmbedding(LoRALayer):
    """
    LoRA wrapper for nn.Embedding layers.
    Adds a low-rank adaptation path to the original Embedding layer
    """

    def __init__(self, orig_layer, r=4, alpha=1.0):
        super().__init__()
        self.orig = orig_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Embedding(orig_layer.num_embeddings, r)
        self.lora_B = nn.Linear(r, orig_layer.embedding_dim, bias=False)

        for param in self.orig.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.orig(x) + self.lora_B(self.lora_A(x)) * self.scaling
