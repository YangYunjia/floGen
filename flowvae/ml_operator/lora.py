
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

class LoRAConv2dDP(nn.Module):
    """
    Wrap an existing Conv2d with LoRA using Depthwise + Pointwise trick.
    """
    def __init__(self, base_conv: nn.Conv2d, r: int = 4, alpha: float = 1.0, dropout: float = 0.0):

        super().__init__()
        assert isinstance(base_conv, nn.Conv2d), "base_conv must be nn.Conv2d"

        # ---- Use the given conv as base ----
        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad = False  # freeze

        in_channels = base_conv.in_channels
        out_channels = base_conv.out_channels
        kH, kW = base_conv.kernel_size
        stride = base_conv.stride
        padding = base_conv.padding
        dilation = base_conv.dilation

        # ---- LoRA path ----
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.enable_lora = r > 0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.enable_lora:
            # 1x1 conv: in -> r
            self.lora_A = nn.Conv2d(in_channels, r, kernel_size=1, bias=False)

            # depthwise conv: r -> r
            self.lora_DW = nn.Conv2d(
                r, r, kernel_size=(kH, kW),
                stride=stride, padding=padding, dilation=dilation,
                groups=r, bias=False
            )

            # 1x1 conv: r -> out
            self.lora_B = nn.Conv2d(r, out_channels, kernel_size=1, bias=False)

            # init: A small random, B zeros
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
            nn.init.zeros_(self.lora_B.weight)
            nn.init.kaiming_uniform_(self.lora_DW.weight, a=5**0.5)
        else:
            self.lora_A = nn.Identity()
            self.lora_DW = nn.Identity()
            self.lora_B = nn.Identity()

    def forward(self, x):
        y = self.base(x)
        if self.enable_lora:
            z = self.lora_A(self.dropout(x))
            z = self.lora_DW(z)
            z = self.lora_B(z)
            y = y + self.scaling * z
        return y

class LoRAQKV(nn.Module):
    """
    replace qkv layers with one add lora
    """
    def __init__(self, qkv_layer: nn.Linear, r=4, alpha=16, dropout=0.05):
        super().__init__()
        self.qkv_base = qkv_layer
        self.qkv_base.weight.requires_grad = False
        if self.qkv_base.bias is not None:
            self.qkv_base.bias.requires_grad = False

        dim = self.qkv_base.in_features

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
            # establish lora QKV (original qkv is saved and set to not require gradients)
            lora_layer = LoRAQKV(module, r=r, alpha=alpha, dropout=dropout).to(module.weight.device, dtype=module.weight.dtype)
            # replace
            setattr(model, name, lora_layer)

        elif isinstance(module, nn.Conv2d) and any(t in name.lower() for t in target_modules):
            # establish lora Conv 
            lora_layer = LoRAConv2dDP(module, r=r, alpha=alpha, dropout=dropout).to(module.weight.device, dtype=module.weight.dtype)
            # replace
            setattr(model, name, lora_layer)

        else:
            # recurrent to sub models
            add_lora_to_model(module, target_modules, r, alpha, dropout)

    return model
