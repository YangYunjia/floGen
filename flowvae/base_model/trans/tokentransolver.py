'''
The modules below reuse the slicing and deslicing operation in the physics-based attention
but the transolver block are in the token space

'''

import torch
import torch.nn as nn
import numpy as np

from typing import Union

from flowvae.base_model.mlp import mlp

from .transolver import Transolver
from .attention import Attention, Physics_Attention, Physics_Attention_2D, Physics_Attention_3D


class TokenSpaceBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, keywords: dict):

        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError('hidden_dim must be divisible by num_heads')
        
        self.keywords = keywords
        self.dim_head = hidden_dim // num_heads

        self.projector = self.fetch_token_projection_layer(num_heads, hidden_dim, keywords['dropout'])

        self.pre_ln = nn.LayerNorm(hidden_dim + keywords['is_add_mesh'])
        act_layer = nn.GELU if keywords['act'] == 'gelu' else nn.LeakyReLU

        self.token_ln = nn.LayerNorm(self.dim_head)
        self.token_mlp = mlp(in_features=self.dim_head, out_features=self.dim_head, hidden_dims=[self.dim_head * keywords['mlp_ratio']], last_actv=False,
                              basic_layers={'actv': act_layer})


    def fetch_token_projection_layer(self, num_heads, hidden_dim, dropout) -> Physics_Attention:

        kwargs = {k: v for k, v in self.keywords.items() if k not in ['act', 'mesh_type']}
        kwargs.update(dict(heads=num_heads, dim_head=self.dim_head))

        mesh_type = self.keywords['mesh_type']

        if mesh_type == '2d':
            return Physics_Attention_2D(hidden_dim, **kwargs)
        elif mesh_type == '3d':
            return Physics_Attention_3D(hidden_dim, **kwargs)
        elif mesh_type == 'point':
            return Physics_Attention(hidden_dim, **kwargs)
        else:
            raise KeyError(f'Unsupported mesh type {mesh_type}')

    def forward(self, fx: torch.Tensor) -> torch.Tensor:

        fx_norm = self.pre_ln(fx)   # B N C

        # calculate slices `x`, `fx` for self attention
        x_mid, fx_mid, N_, N0_ = self.projector._forward_slice(fx_norm) 
        slice_token, slice_weights = self.projector._project_token(x_mid, fx_mid, N_)  # B Nh M Ch,   B Nh N M

        # calculate `x` for cross attention, the `fx` (value) remains the same for cross attention
        slice_token_cross = slice_token

        ### (2) Attention among slice tokens
        q = self.projector.to_q(slice_token)
        k = self.projector.to_k(slice_token_cross)
        v = self.projector.to_v(slice_token_cross)
        out_slice_token = self.projector._calculate_attention(q, k, v)  # B Nh M Ch

        out_slice_token = out_slice_token + slice_token

        tokens = self.token_mlp(self.token_ln(out_slice_token)) + out_slice_token

        fx = self.projector._deslice(tokens, slice_weights, N0_)
        fx = fx + fx_norm[..., self.keywords['is_add_mesh']:]

        return fx
    
class TokenSpaceTransolver(Transolver):

    def _fetch_blocks(self):
        self.blocks = nn.ModuleList([
            TokenSpaceBlock(num_heads=self.n_head, hidden_dim=self.n_hidden, keywords=self.keywords)
            for _ in range(self.n_layers)
        ])
        return self.n_hidden
