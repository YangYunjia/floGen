
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    LoRA-augmented Linear layer without original linear layers.
    
    """
    def __init__(self, in_features, out_features, r=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0

        if r > 0:
            # Trainable LoRA adapters
            self.A = nn.Parameter(torch.zeros(r, in_features))
            self.B = nn.Parameter(torch.zeros(out_features, r))
            self.dropout = nn.Dropout(dropout)
            # Init
            nn.init.normal_(self.A, std=0.01)
            nn.init.zeros_(self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)
            self.dropout = nn.Identity()

    def forward(self, x):
        if self.r > 0:
            return (self.dropout(x) @ self.A.T) @ self.B.T * self.scaling
        else:
            return torch.zeros(x.shape[0], self.B.shape[0], device=x.device, dtype=x.dtype)

class QKVwithLoRA(nn.Module):
    """
    replace qkv layers with one add lora
    """
    def __init__(self, dim, qkv_layer, r=4, alpha=16, dropout=0.05):
        super().__init__()
        self.qkv_base = qkv_layer
        self.qkv_base.weight.requires_grad = False
        if self.qkv_base.bias is not None:
            self.qkv_base.bias.requires_grad = False

        # add lora for q and v
        self.lora_q = LoRALinear(dim, dim, r=r, alpha=alpha, dropout=dropout)
        self.lora_v = LoRALinear(dim, dim, r=r, alpha=alpha, dropout=dropout)

    def forward(self, x):

        qkv = self.qkv_base(x)
        lora_addition = torch.concat((self.lora_q(x), torch.zeros_like(x), self.lora_v(x)), dim=-1)
        return qkv + lora_addition

def add_lora_to_model(model, target_modules=("qkv"), r=4, alpha=16, dropout=0.05):
    """
    Recursively replace nn.Linear layers with LoRALinear in target modules.
    Args:
        model: nn.Module, the base transformer model
        target_modules: tuple of strings, module names to apply LoRA (e.g. ("qkv", "proj"))
        r, alpha, dropout: LoRA hyperparams
    Returns:
        model with LoRA layers injected
    """
    for p in model.parameters():
            p.requires_grad = False

    for name, module in model.named_children():
        # match the qkv layers
        if isinstance(module, nn.Linear) and any(t in name.lower() for t in target_modules):
            dim = module.in_features

            # establish lora QKV (original qkv is saved and set to not require gradients)
            lora_layer = QKVwithLoRA(dim, module, r=r, alpha=alpha, dropout=dropout).to(module.weight.device, dtype=module.weight.dtype)

            # replace
            setattr(model, name, lora_layer)

        else:
            # recurrent to sub models
            add_lora_to_model(module, target_modules, r, alpha, dropout)

    return model
