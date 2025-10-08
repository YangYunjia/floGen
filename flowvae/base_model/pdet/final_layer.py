'''
2025.10.8


'''
import torch
from torch import nn
import torch.nn.functional as F

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class AttnPoolVector(nn.Module):

    def __init__(self, in_dim, num_heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, in_dim))  # learnable query
        self.attn = nn.MultiheadAttention(in_dim, num_heads, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # input B, C, H, W         
        tokens = tokens.flatten(start_dim=2).permute(0, 2, 1) # tokens: [B, N, C]
        q = self.q.expand(tokens.size(0), -1, -1)      # [B,1,C]
        out, _ = self.attn(q, tokens, tokens)  # [B,1,C]
        return out.squeeze(1)  # [B, C]

class FinalLayer(nn.Module):
    """
    The final layer of IPT.

    25.10.8 modified to a more generalized Final layer

    input: B, C, H, W
    """

    def __init__(self, hidden_size, out_channels, out_proj: str = 'conv2d', is_adp: bool = False):
        super().__init__()
        self.norm_final = LayerNorm2d(hidden_size, affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        if out_proj in ['conv2d']:
            self.reshaping = nn.Identity()
            self.out_proj = nn.Conv2d(hidden_size, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        elif out_proj in ['avg_pool']:
            self.reshaping = nn.Sequential(
                nn.Flatten(start_dim=2),                         # [B, C, H, W] -> [B, C, N]
                nn.AdaptiveAvgPool1d(1),                         # [B, C, N] -> [B, C, 1]
                nn.Flatten(start_dim=1)
            ) 
            self.out_proj = nn.Linear(hidden_size, out_channels)
        elif out_proj in ['attn_pool']:
            self.reshaping = AttnPoolVector(hidden_size)
            self.out_proj = nn.Linear(hidden_size, out_channels)
        else:
            raise NotImplementedError()
        self.is_adp = is_adp
        
    # def zero_init(self):
    #     nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
    #     nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    #     nn.init.constant_(self.out_proj.weight, 0)
    #     nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, x, c):
        if self.is_adp:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
        x = self.reshaping(x)
        x = self.out_proj(x)
        return x