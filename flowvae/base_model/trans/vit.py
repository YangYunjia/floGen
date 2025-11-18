
import torch
import torch.nn as nn
import numpy as np

from .attention import Attention
from flowvae.base_model.mlp import mlp


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

class ViT_block(nn.Module):
    '''
    ViT block
    
    '''

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float = 0.,
            act='gelu',
            mlp_ratio=4,
            is_add_mesh=0,
    ) -> None:
        
        super().__init__()
        if act == 'gelu':
            act_layer = nn.GELU
        else:
            act_layer = nn.LeakyReLU
        
        self.ln_1 = nn.LayerNorm(hidden_dim + is_add_mesh)
        self.Attn = self.fetch_attention_layer(num_heads, hidden_dim, dropout)
        self.is_add_mesh = is_add_mesh

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = mlp(in_features=hidden_dim, out_features=hidden_dim, hidden_dims=[hidden_dim * mlp_ratio], last_actv=False,
                       basic_layers={'actv': act_layer})   # , res=False, act=act

    def fetch_attention_layer(self, num_heads, hidden_dim, dropout) -> Attention:
        dim_head = hidden_dim // num_heads
        return Attention(hidden_dim, heads=num_heads, dim_head=dim_head, dropout=dropout)

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        # attention
        fx = self.ln_1(fx)
        fx = self.Attn(fx, None) + fx[..., self.is_add_mesh:]
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx

class ViT(nn.Module):
    
    def __init__(self,
                 image_size,
                 patch_size,
                 fun_dim: int = 3,
                 out_dim: int = 1,
                 n_layers: int = 5,
                 n_hidden: int = 256,
                 n_head: int = 8,
                 mlp_ratio: int = 4,
                 add_mesh: int = 0,
                 dropout=0.0,
                 pos_embedding='sincos',
                 act='gelu',
                 device: str = 'cuda:0'
                 ):

        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = fun_dim * patch_height * patch_width
        
        self.ph, self.pw, self.nh, self.nw = patch_height, patch_width, image_height // patch_height, image_width // patch_width

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, n_hidden),
            nn.LayerNorm(n_hidden),
        )

        if pos_embedding in ['trainable']:
            self.pos_embedding = nn.Parameter((1 / (n_hidden)) * torch.rand((1, self.nh * self.nw, n_hidden), dtype=torch.float))
        elif pos_embedding in ['sincos']:
            self.pos_embedding = posemb_sincos_2d(h = self.nh, w = self.nw, dim = n_hidden)
        else:
            raise KeyError()
            
        self.blocks = nn.ModuleList([ViT_block(num_heads=n_head, hidden_dim=n_hidden,
                                                    dropout=dropout,
                                                    act=act,
                                                    mlp_ratio=mlp_ratio,
                                                    slice_num=None,
                                                    mesh_type='ViT',
                                                    is_add_mesh=add_mesh)
                                    for _ in range(n_layers)])
        
        self.last_layer = nn.Sequential(nn.LayerNorm(n_hidden), nn.Linear(n_hidden, self.ph * self.pw * out_dim))
        self.out_dim = out_dim
        self.device = device

    def forward(self, img: torch.Tensor):
        
        C0_ = img.shape[1]
        img = img.reshape(-1, C0_, self.nh, self.ph, self.nw, self.pw).permute(0, 2, 4, 3, 5, 1).reshape(-1, self.nh * self.nw, self.ph * self.pw * C0_)

        fx = self.to_patch_embedding(img)
        fx += self.pos_embedding.to(self.device, dtype=fx.dtype)

        for block in self.blocks:
            fx = block(fx)
            
        fx = self.last_layer(fx)
        
        results = fx.reshape(-1, self.nh, self.nw, self.ph, self.pw, self.out_dim).permute(0, 5, 1, 3, 2, 4).reshape(-1, self.out_dim, self.nh * self.ph, self.nw * self.pw)
        
        return results
