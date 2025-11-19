'''
Oct. 27 2024 
Author: Yunjia Yang

The basic models of transformer-based models

reference
---
https://github.com/thuml/Transolver/ (Transolver_Structured_Mesh_2D.py)
https://github.com/jadore801120/attention-is-all-you-need-pytorch/ (encoder - decoder)

'''
import torch
import torch.nn as nn
import numpy as np
from functools import reduce
import math

from flowvae.base_model.mlp import mlp

from .attention import Attention, Physics_Attention, Physics_Attention_2D, Physics_Attention_3D
from .vit import ViT_block, posemb_sincos_2d
from .sampling import *

class Transolver_block(ViT_block):
    '''
    Transformer block
    
    - `num_heads`:    (Nh) default = 8
    - `hidden_dim`:   (C)  default = 256
    
    - remark Nov 5, 24 -> move last_block mark to Transolver class
    
    inputs & outputs
    ---
    
        B N0 C0 --upsampling--> B N C
                --Attention --> B N C
                --dnsampling--> B N1 C1
    
    '''

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float = 0.,
            act='gelu',
            mlp_ratio=4,
            slice_num=32,
            mesh_type='2d',
            is_add_mesh=0,
    ) -> None:
        
        self.mesh_type = mesh_type
        self.slice_num = slice_num
        self.is_add_mesh = is_add_mesh
        
        super().__init__(num_heads, hidden_dim, dropout, act, mlp_ratio, is_add_mesh)

    def fetch_attention_layer(self, num_heads, hidden_dim, dropout) -> Attention:
        dim_head = hidden_dim // num_heads
        if self.mesh_type == '2d':
            return Physics_Attention_2D(hidden_dim, heads=num_heads, dim_head=dim_head, dropout=dropout, slice_num=self.slice_num)
        elif self.mesh_type == '3d':
            return Physics_Attention_3D(hidden_dim, heads=num_heads, dim_head=dim_head, dropout=dropout, slice_num=self.slice_num)
        elif self.mesh_type == 'point':
            return Physics_Attention(hidden_dim, heads=num_heads, dim_head=dim_head, dropout=dropout, slice_num=self.slice_num, is_add_mesh=self.is_add_mesh)
        else:
            raise NotImplementedError()
    
    def get_slice_weight(self, fx: torch.Tensor) -> torch.Tensor:
        '''
        return:
        ---
        
        `torch.Tensor`:   B Nh H W M 
        
        '''
        return self.Attn.get_slice_weight(self.ln_1(fx))

class Transolver_stage(nn.Module):

    def __init__(
            self,
            num_heads: int,
            depth: int,
            hidden_dim: int,
            dropout: float = 0.,
            act='gelu',
            mlp_ratio=4,
            slice_num=32,
            mesh_type='2d',
            is_add_mesh=0,
    ) -> None:
        
        super().__init__()
        self.dim = hidden_dim
        blocks = []
        for i in range(depth):

            block = Transolver_block(
                num_heads = num_heads,
                hidden_dim = hidden_dim,
                dropout = dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                slice_num=slice_num,
                mesh_type=mesh_type,
                is_add_mesh=is_add_mesh,
            )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        for n, block in enumerate(self.blocks):
            x = block(x)
        return x.permute(0, 3, 1, 2)

class Transolver_block_cross(Transolver_block):
    
    def __init__(self, num_heads, hidden_dim, dropout = 0, act='gelu', mlp_ratio=4, slice_num=32, mesh_type='2d', is_add_mesh=0, downsampling = nn.Identity(), upsampling = nn.Identity()):
        
        super().__init__(num_heads, hidden_dim, dropout, act, mlp_ratio, slice_num, mesh_type, is_add_mesh, downsampling, upsampling)
        self.ln_1_cross = nn.LayerNorm(hidden_dim)
        self.ln_1_self  = nn.LayerNorm(hidden_dim)
        self.cross_Attn = self.fetch_attention_layer(num_heads, hidden_dim, dropout, slice_num, mesh_type, is_add_mesh=0)
        
        
    def forward(self, fx: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        # upsampling
        fx = self.upsampling(fx)
        # attention
        fx = self.ln_1(fx)
        fx = self.Attn(fx, fx) + fx[:, :, :, self.is_add_mesh:]
        
        enc = self.ln_1_cross(enc)
        fx  = self.ln_1_self(fx)
        fx = self.cross_Attn(fx, enc) + fx
        
        fx = self.mlp(self.ln_2(fx)) + fx
        # downsampling
        fx = self.dnsampling(fx)
        return fx

class Transolver(nn.Module):
    def __init__(self,
                 space_dim: int = 1,
                 fun_dim: int = 1,
                 out_dim: int = 1,
                 n_layers: int = 5,
                 n_hidden: int = 256,
                 n_head: int = 8,
                 slice_num: int = 32,
                 mlp_ratio: int = 4,
                 mesh_type: str = '2d',
                 add_mesh: int = 0,
                 dropout=0.0,
                 placeholder: dict = {'type': 'random'},
                 Time_Input=False,
                 act='gelu',
                 ref=8,
                 unified_pos=False,
                 device: str = 'cuda:0'
                 ):
        '''
        `u_shape`:
            - `0` -> original
            - `1` -> original + down/upsampling
        
        '''
        super().__init__()
        self.__name__ = 'Transolver_2D'
        # self.ref = ref
        self.unified_pos = unified_pos
        self.device = device

        if self.unified_pos:
            raise NotImplementedError
            # self.pos = self.get_grid()
            # self.preprocess = MLP(fun_dim + self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = mlp(in_features=fun_dim + space_dim, out_features=n_hidden, hidden_dims=[n_hidden * 2], last_actv=False) #res=False, act=act)

        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.n_layers = n_layers
        self.n_head = n_head
        self.slice_num = slice_num
        self.mlp_ratio = mlp_ratio
        self.mesh_type = mesh_type
        self.add_mesh = add_mesh
        self.dropout = dropout
        self.act = act

        n_hidden_out = self._fetch_blocks()

        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.last_layer = nn.Sequential(nn.LayerNorm(n_hidden_out), nn.Linear(n_hidden_out, out_dim))
        
        self.initialize_weights()
        
        if placeholder['type'] in ['random']:
            self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))
        elif placeholder['type'] in ['sincos']:
            self.placeholder = posemb_sincos_2d(h = placeholder['nh'], w = placeholder['nw'], dim = n_hidden)
        else:
            raise KeyError()
    
    def _fetch_blocks(self):
        self.blocks = nn.ModuleList([Transolver_block(num_heads=self.n_head, hidden_dim=self.n_hidden,
                                                    dropout=self.dropout,
                                                    act=self.act,
                                                    mlp_ratio=self.mlp_ratio,
                                                    slice_num=self.slice_num,
                                                    mesh_type=self.mesh_type,
                                                    is_add_mesh=self.add_mesh)
                                    for _ in range(self.n_layers)])
        return self.n_hidden

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, batchsize=1):

        size_x, size_y = self.H, self.W
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x, size_y, self.ref * self.ref).contiguous()
        return pos

    def forward(self, *args, **kwargs) -> torch.Tensor:
        '''
        B N C -> B N C
        '''

        fx = self._process(*args, **kwargs)
        for block in self.blocks:
            if self.add_mesh:
                fx = torch.cat((kwargs['mesh'], fx), dim=-1)
            fx = block(fx)
            
        fx = self.last_layer(fx)
        return fx
    
    def _process(self, x: torch.Tensor, fx: torch.Tensor, T=None) -> torch.Tensor:
        
        # if self.unified_pos:
        #     x = self.pos.repeat(x.shape[0], 1, 1, 1).reshape(x.shape[0], self.H * self.W, self.ref * self.ref)
        if fx is not None:
            fx = self.preprocess(torch.cat((x, fx), -1))
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder.to(self.device, dtype=fx.dtype)

        # if T is not None:
        #     Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
        #     Time_emb = self.time_fc(Time_emb)
        #     fx = fx + Time_emb
        
        return fx
    
    def get_slice_weight(self, *args, **kwargs) -> torch.Tensor:
        '''
        return:
        ---
        
        `torch.Tensor`:   B Nh H W M 
        
        '''
        fx = self._process(*args, **kwargs)
        slice_weights = []
        
        for block in self.blocks:
            slice_weights.append(block.get_slice_weight(fx))
            fx = block(fx)

        return slice_weights

class UTransolver(Transolver):

    def __init__(self, space_dim = 1, fun_dim = 1, out_dim = 1, depths=[2, 5, 8, 5, 2], n_hidden = 256, n_head = 8, slice_num = 32, mlp_ratio = 4, mesh_type = '2d', u_shape = 1, add_mesh = 0, dropout=0, placeholder = { 'type': 'random' }, Time_Input=False, act='gelu', ref=8, unified_pos=False, device = 'cuda:0'):
        '''
        `u_shape`
        = 1
        = 2 learnable
        
        '''
        self.u_shape = u_shape
        self.depths = depths
        super().__init__(space_dim, fun_dim, out_dim, sum(depths), n_hidden, n_head, slice_num, mlp_ratio, mesh_type, add_mesh, dropout, placeholder, Time_Input, act, ref, unified_pos, device)

    def _fetch_blocks(self):
        self.num_encoder_layers = len(self.depths) // 2
        hidden_size = self.n_hidden
        max_hidden_size = 256

        if self.u_shape == 2:
            self.learnable_samplers = nn.ModuleDict()
        else:
            self.learnable_samplers = None

        # encoder
        for i in range(self.num_encoder_layers):
            hidden_size_layer = min(hidden_size * 2 ** i, max_hidden_size)
            self.__setattr__(f"encoder_level_{i}", Transolver_stage(num_heads=self.n_head, depth=self.depths[i], hidden_dim=hidden_size_layer,
                                                        mlp_ratio=self.mlp_ratio,
                                                        slice_num=self.slice_num,
                                                        mesh_type=self.mesh_type))
            if hidden_size_layer == max_hidden_size:
                keep_dim = True
            else:
                keep_dim = False

            if self.u_shape == 2:
                sampler_key = f"{i}_{i+1}"
                sampler = LearnableSamplingMatrix()
                self.learnable_samplers[sampler_key] = sampler
                down_module = LearnablePointDownsample(hidden_size_layer, keep_dim=keep_dim, sampler=sampler)
            elif self.u_shape == 1:
                down_module = Downsample(hidden_size_layer, keep_dim=keep_dim)
            else:
                raise NotImplementedError()

            self.__setattr__(f"down{i}_{i+1}", down_module)

        # latent
        hidden_size_latent = min(hidden_size * 2 ** self.num_encoder_layers, max_hidden_size)
        self.latent = Transolver_stage(num_heads=self.n_head, depth=self.depths[self.num_encoder_layers], hidden_dim=hidden_size_latent,
                                                        mlp_ratio=self.mlp_ratio,
                                                        slice_num=self.slice_num,
                                                        mesh_type=self.mesh_type)
        self.is_decoder = True
        if self.is_decoder:

            hidden_size_layer0 = min(hidden_size * 2, max_hidden_size)
            if hidden_size_layer0 >= max_hidden_size:
                keep_dim = True
            else:
                keep_dim = False

            # double hidden size for last decoder layer 0
            if self.u_shape == 2:
                sampler_key = "0_1"
                if sampler_key not in self.learnable_samplers:
                    raise ValueError("Learnable sampler for level 0_1 was not initialized.")
                sampler = self.learnable_samplers[sampler_key]
                up_module = LearnablePointUpsample(hidden_size_layer0, keep_dim=keep_dim, sampler=sampler)
            elif self.u_shape == 1:
                up_module = Upsample(hidden_size_layer0, keep_dim=keep_dim)
            else:
                raise NotImplementedError()

            self.__setattr__("up1_0", up_module)
            self.__setattr__("reduce_chan_level0", nn.Conv2d(2 * min(hidden_size, max_hidden_size), hidden_size_layer0, kernel_size=1, bias=True))
            self.__setattr__("decoder_level_0", Transolver_stage(num_heads=self.n_head, depth=self.depths[self.num_encoder_layers + 1], hidden_dim=hidden_size_layer0,
                                                        mlp_ratio=self.mlp_ratio,
                                                        slice_num=self.slice_num,
                                                        mesh_type=self.mesh_type))

            # decoder layers 1 - num_encoder_layers
            for i in range(1, self.num_encoder_layers):

                hidden_size_layer = min(hidden_size * 2 ** i, max_hidden_size)
                if 2 * hidden_size_layer >= max_hidden_size:
                    keep_dim = True
                    hidden_size_upsample = max_hidden_size
                else:
                    keep_dim = False
                    hidden_size_upsample = 2 * hidden_size_layer

                if self.u_shape == 2:
                    sampler_key = f"{i}_{i+1}"
                    if sampler_key not in self.learnable_samplers:
                        raise ValueError(f"Learnable sampler for level {sampler_key} was not initialized.")
                    sampler = self.learnable_samplers[sampler_key]
                    up_module = LearnablePointUpsample(hidden_size_upsample, keep_dim=keep_dim, sampler=sampler)
                elif self.u_shape == 1:
                    up_module = Upsample(hidden_size_upsample, keep_dim=keep_dim)
                else:
                    raise NotImplementedError()
                
                self.__setattr__(f"up{i+1}_{i}", up_module)
                self.__setattr__(f"reduce_chan_level{i}", nn.Conv2d(hidden_size_layer * 2, hidden_size_layer, kernel_size=1, bias=True))
                self.__setattr__(f"decoder_level_{i}", Transolver_stage(num_heads=self.n_head, depth=self.depths[self.num_encoder_layers + i + 1], hidden_dim=hidden_size_layer,
                                                        mlp_ratio=self.mlp_ratio,
                                                        slice_num=self.slice_num,
                                                        mesh_type=self.mesh_type))

            hidden_size_out = min(2 * hidden_size, max_hidden_size)

        return hidden_size_out

    def forward(self, *args, **kwargs) -> torch.Tensor:
        '''
        B N C -> B N C
        '''

        x = self._process(*args, **kwargs)

        residuals_list = []
        for i in range(self.num_encoder_layers):
            # encoder
            out_enc_level = self.__getattr__(f"encoder_level_{i}")(x)
            residuals_list.append(out_enc_level)
            x = self.__getattr__(f"down{i}_{i+1}")(out_enc_level)
        x = self.latent(x)   # （PDEStage -> B C H W)

        if self.is_decoder:

            for i, residual in enumerate(residuals_list[1:][::-1]):
                # decoder
                x = self.__getattr__(f"up{self.num_encoder_layers - i}_{self.num_encoder_layers - i - 1}")(x)
                x = torch.cat([x, residual], 1)
                x = self.__getattr__(f"reduce_chan_level{self.num_encoder_layers - i - 1}")(x)
                x = self.__getattr__(f"decoder_level_{self.num_encoder_layers - i - 1}")(x)

            x = self.__getattr__(f"up1_0")(x)
            x = torch.cat([x, residuals_list[0]], 1)
            x = self.__getattr__(f"reduce_chan_level0")(x)
            x = self.__getattr__(f"decoder_level_0")(x)

            # output
            x = self.last_layer(x.permute(0, 2, 3, 1))

        else:
            raise NotImplementedError()
            x = self.final_layer(x)

        return x

class EncoderDecoderTransolver(Transolver):
    '''
    to extract information from the reference flow fields
    
    '''
    
    def __init__(self, space_dim = 1, fun_dim = 1, ref_dim = 1, out_dim = 1, n_layers_enc = 3, n_layers_dec = 3, n_hidden = 256, n_head = 8, slice_num = 32, mlp_ratio = 4, mesh_type = '2d', add_mesh = 0, dropout=0, Time_Input=False, act='gelu', ref=8, unified_pos=False, device = 'cuda:0'):
        super().__init__(space_dim, fun_dim, out_dim, 0, n_hidden, n_head, slice_num, mlp_ratio, mesh_type, -1, add_mesh, dropout, Time_Input, act, ref, unified_pos, device)
        
        self.enc_blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                        dropout=dropout,
                                        act=act,
                                        mlp_ratio=mlp_ratio,
                                        slice_num=slice_num,
                                        mesh_type=mesh_type,
                                        is_add_mesh=add_mesh)
                        for _ in range(n_layers_enc)])
        # build decoder with cross attention
        self.dec_blocks = nn.ModuleList([Transolver_block_cross(num_heads=n_head, hidden_dim=n_hidden,
                                        dropout=dropout,
                                        act=act,
                                        mlp_ratio=mlp_ratio,
                                        slice_num=slice_num,
                                        mesh_type=mesh_type,
                                        is_add_mesh=add_mesh)
                        for _ in range(n_layers_dec)])
        
        self.preprocess_ref = mlp(in_features=ref_dim, out_features=n_hidden, hidden_dims=[n_hidden * 2], last_actv=False)
        
    def forward(self, *args, **kwargs) -> torch.Tensor:
        
        fr, fx = self._process(*args, **kwargs)
        for block in self.enc_blocks:
            if self.add_mesh > 0:
                fr = torch.cat((kwargs['mesh_ref'], fr), dim=-1)
            fr = block(fr)
        
        for block in self.dec_blocks:
            if self.add_mesh > 0:
                fx = torch.cat((kwargs['mesh'], fx), dim=-1)
            fx = block(fx, fr)
        
        fx = self.last_layer(fx)
        return fx
    
    def _process(self, x: torch.Tensor, fx: torch.Tensor, fr: torch.Tensor, T=None) -> torch.Tensor:

        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        fr = self.preprocess_ref(fr)
        
        return fr, fx