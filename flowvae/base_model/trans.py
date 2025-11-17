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
from flowvae.base_model.utils import IntpConv
from flowvae.base_model.pdet.pde_transformer import Downsample, Upsample

class Attention(nn.Module):
    
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        '''
        B M C -> B M C
        M: patch number (for ViT = Mh x Mw patch number at h & w)
        C: channel for each patch (dim)
        
        '''
        
        super().__init__()
        
        inner_dim = dim_head * heads
        self.dim_head = dim_head        # Ch
        self.heads = heads              # Nh
        
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        #  from input channel (C) to multi-head and channel in each head
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, x_cross: torch.Tensor) -> torch.Tensor:
        
        if x_cross is None:
            x_cross = x
        
        M_ = x.shape[1]
        # B M C -> B M (Nh Ch)
        # q, k, v = [ly(x).reshape(-1, M_, self.heads, self.dim_head).permute(0, 2, 1, 3) for ly in [self.to_q, self.to_k, self.to_v]]
        q = self.to_q(x).reshape(-1, M_, self.heads, self.dim_head).permute(0, 2, 1, 3)
        k = self.to_k(x_cross).reshape(-1, M_, self.heads, self.dim_head).permute(0, 2, 1, 3)
        v = self.to_v(x_cross).reshape(-1, M_, self.heads, self.dim_head).permute(0, 2, 1, 3)
        
        out = self._calculate_attention(q, k, v).permute(0, 2, 1, 3).reshape(-1, M_, self.heads * self.dim_head)

        return self.to_out(out)
    
    def _calculate_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        '''
        B Nh M Ch -> B Nh M Ch
        
        '''

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)  # B Nh M Ch

class Physics_Attention(Attention):
    
    def __init__(self, dim, heads=8, dim_head=64, slice_num=64, dropout=0., is_add_mesh=0):
        '''
        `is_add_mesh`
            - `0` -> original
            - `1 ~ 3` -> add mesh (1 ~ 3 D)
        
        '''
        super().__init__(dim, heads, dim_head, dropout)
        
        inner_dim = dim_head * heads
        self.slice_num = slice_num      # M
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        # mapping bet. physical data to slices
        self.in_project_x = nn.Identity()
        self.in_project_fx = nn.Identity()
        self.is_add_mesh = is_add_mesh  # whether to add 3D mesh as the only `x` to calculate weights
        self._prepare_projection(dim, inner_dim)

        self.in_project_slice = nn.Linear(dim_head, slice_num)  # Ch -> M
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        
        # the heads are difference in getting projection weights and values to be projected
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
    
    def _prepare_projection(self, dim, inner_dim):
        if self.is_add_mesh > 0:
            self.in_project_x  = nn.Linear(dim + self.is_add_mesh, inner_dim)
            self.in_project_fx = nn.Linear(dim + self.is_add_mesh, inner_dim)
        else:
            self.in_project_x = nn.Linear(dim, inner_dim)
            self.in_project_fx = nn.Linear(dim, inner_dim)
    
    def forward(self, x: torch.Tensor, x_cross: torch.Tensor) -> torch.Tensor:

        # calculate slices `x`, `fx` for self attention
        x_mid, fx_mid, N_, N0_ = self._forward_slice(x) 
        slice_token, slice_weights = self._project_token(x_mid, fx_mid, N_)  # B Nh M Ch,   B Nh N M
        
        # calculate `x` for cross attention, the `fx` (value) remains the same for cross attention
        if x_cross is not None:
            x_mid_cross, fx_mid_cross, _, _ = self._forward_slice(x_cross)
            slice_token_cross, _ = self._project_token(x_mid_cross, fx_mid_cross, N_)   # B Nh M Ch
        else:
            slice_token_cross = slice_token

        ### (2) Attention among slice tokens
        q = self.to_q(slice_token)
        k = self.to_k(slice_token_cross)
        v = self.to_v(slice_token_cross)
        out_slice_token = self._calculate_attention(q, k, v)  # B Nh M Ch

        ### (3) Deslice
        return self._deslice(out_slice_token, slice_weights, N0_)
    
    def _forward_slice(self, x: torch.Tensor):
        '''
        expand input channel (C -> Nh x Ch), split to several head
        the output x_mid and fx_mid are actully the pre-version WEIGHT of each node point(N) to each slice(M) at each head(Nh)
                                                                                                           ^
                                                                                        the slice is expanded later from here(Ch) to slices(M)
        (two parts: x -> projection weights, and fx -> value)
        
        ### inputs:
            tensor:  B x (...) x C
            |
            expand (mlp, conv. etc. to (...)) -> then flatten (...) to N
            |
            tensor:  B x N x (Nh x Ch)
            |
            tensor:  B x Nh | N x Ch (Nh = each head) 
        
        '''
        
        N0_ = x.shape[1:-1]
        N_ = reduce(lambda x, y: x*y, N0_)
        fx_mid = self.in_project_fx(x).reshape(-1, N_, self.heads, self.dim_head).permute(0, 2, 1, 3)  # B Nh N Ch
        x_mid  = self.in_project_x(x).reshape(-1, N_, self.heads, self.dim_head).permute(0, 2, 1, 3)  # B Nh N Ch
        return x_mid, fx_mid, N_, N0_
    
    def _project_token(self, x_mid: torch.Tensor, fx_mid: torch.Tensor, N_: int) -> torch.Tensor:
        '''
        project from physical mesh data (N) to several slices (M)
        
        weights: N -> M (softmax)
        
                                     B x Nh x N x M
        projection: B x Nh x N x Ch        ->         B x Nh x M x Ch
         
        
        '''
        slice_weights = self.softmax(self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B Nh N M 
        slice_norm    = slice_weights.sum(2)  # B Nh M
        # Nov. 8 2024 -> move non-dimensional for slice here -> the weights used for reconstruct will be different
        slice_weights = slice_weights / ((slice_norm + 1e-5)[:, :, None, :].repeat(1, 1, N_, 1))
        
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)    # B Nh M Ch
        
        return slice_token, slice_weights
    
    def _deslice(self, out_slice_token: torch.Tensor, slice_weights: torch.Tensor, N0_) -> torch.Tensor:
        
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights) # B Nh N Ch
        out_x = out_x.permute(0, 2, 1, 3).reshape(-1, *N0_, self.heads * self.dim_head) # B N Nh*Ch
        return self.to_out(out_x)
        
    def get_slice_weight(self, x: torch.Tensor) -> torch.Tensor:
        '''
        return:
        ---
        
        `torch.Tensor`:   B Nh H W M 
        
        '''
        x_mid, fx_mid, N_, N0_ = self._forward_slice(x)
        slice_weights = self.softmax(self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B Nh N M 
        
        return slice_weights.reshape(-1, self.heads, *N0_, self.slice_num)

class Physics_Attention_2D(Physics_Attention):

    '''
    for structured mesh in 2D space

    Here we modified input and output to be H W, not N = H x W, so that H and W are not input as a argument
    
    The conditional parameters are added as tokens
    
    paras:
    ---
    - `dim`:    (C_out)     The channel number of output feature maps
    - `heads`:  (Nh)        The heads number
    - `dim_head`: (Ch)      The channel number of each head
    - `slice_num`: (M)      The slice number
    - `
    
    inputs & outputs:
    ---
    B H W C -> B H W C_out
    
    '''
    def __init__(self, dim, heads=8, dim_head=64, slice_num=64, kernel=3, dropout=0):
        self.kernel = kernel
        super().__init__(dim, heads, dim_head, slice_num, dropout)
        
    def _prepare_projection(self, dim, inner_dim):
        
        kernel = self.kernel
        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)

    def _forward_slice(self, x: torch.Tensor):
        
        # B N C
        _, H_, W_, _ = x.shape
        N_ = H_ * W_
        N0_ = [H_, W_]
        
        x = x.permute(0, 3, 1, 2)  # B C H W

        ### (1) Slice
        # B C H W --CNN--> B Nh*Ch H W -> B (H W) Nh*Ch -> B N Nh Ch-> B Nh N Ch
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).reshape(-1, N_, self.heads, self.dim_head).permute(0, 2, 1, 3)  # B Nh N Ch
        x_mid  = self.in_project_x( x).permute(0, 2, 3, 1).reshape(-1, N_, self.heads, self.dim_head).permute(0, 2, 1, 3)  # B Nh N Ch: `x`
        
        return x_mid, fx_mid, N_, N0_

class Physics_Attention_3D(Physics_Attention):

    def __init__(self, dim, heads=8, dim_head=64, slice_num=64, kernel=3, dropout=0):
        self.kernel = kernel
        super().__init__(dim, heads, dim_head, slice_num, dropout)
        
    def _prepare_projection(self, dim, inner_dim):
        
        kernel = self.kernel
        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)

    def _forward_slice(self, x: torch.Tensor):
        
        # B N C
        _, H_, W_, D_, _ = x.shape
        N_ = H_ * W_ * D_
        N0_ = [H_, W_, D_]
        
        x = x.permute(0, 4, 1, 2, 3)  # B C H W

        ### (1) Slice
        # B C H W -> B Nh*Ch H W -> B (H W) Nh*Ch -> B N Nh Ch-> B Nh N Ch
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 4, 1).reshape(-1, N_, self.heads, self.dim_head).permute(0, 2, 1, 3)  # B Nh N Ch
        x_mid  = self.in_project_x( x).permute(0, 2, 3, 4, 1).reshape(-1, N_, self.heads, self.dim_head).permute(0, 2, 1, 3)  # B Nh N Ch: `x`
        
        return x_mid, fx_mid, N_, N0_

class Transolver_block(nn.Module):
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
        
        super().__init__()
        if act == 'gelu':
            act_layer = nn.GELU
        else:
            act_layer = nn.LeakyReLU
        
        self.ln_1 = nn.LayerNorm(hidden_dim + is_add_mesh)
        self.Attn = self.fetch_attention_layer(num_heads, hidden_dim, dropout, slice_num, mesh_type, is_add_mesh)
        self.is_add_mesh = is_add_mesh

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = mlp(in_features=hidden_dim, out_features=hidden_dim, hidden_dims=[hidden_dim * mlp_ratio], last_actv=False,
                       basic_layers={'actv': act_layer})   # , res=False, act=act

    def fetch_attention_layer(self, num_heads, hidden_dim, dropout, slice_num, mesh_type, is_add_mesh) -> Attention:
        dim_head = hidden_dim // num_heads
        if mesh_type == '2d':
            return Physics_Attention_2D(hidden_dim, heads=num_heads, dim_head=dim_head, dropout=dropout, slice_num=slice_num)
        elif mesh_type == '3d':
            return Physics_Attention_3D(hidden_dim, heads=num_heads, dim_head=dim_head, dropout=dropout, slice_num=slice_num)
        elif mesh_type == 'point':
            return Physics_Attention(hidden_dim, heads=num_heads, dim_head=dim_head, dropout=dropout, slice_num=slice_num, is_add_mesh=is_add_mesh)
        elif mesh_type == 'ViT':
            return Attention(hidden_dim, heads=num_heads, dim_head=dim_head, dropout=dropout)

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        # attention
        fx = self.ln_1(fx)
        fx = self.Attn(fx, None) + fx[..., self.is_add_mesh:]
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx
    
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

class Downsampling(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.up = nn.AvgPool2d(kernel_size=5, stride=4, padding=2)
    
    def forward(self, x):
        return self.up(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

class Upsampling(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
    
    def forward(self, x):
        result = torch.nn.functional.interpolate(x.permute(0, 3, 1, 2), size=self.size, mode='bilinear', align_corners=False)
        
        return result.permute(0, 2, 3, 1)


class LearnableSamplingMatrix(nn.Module):
    """
    Learnable dense-to-coarse sampler that records the sampling matrix for reuse
    during upsampling.
    """

    def __init__(self, reduction: int = 2):
        super().__init__()
        self.reduction = reduction
        self.register_parameter('sampling_matrix', None)
        self.dense_shape = None
        self.coarse_shape = None

    def _initialize_if_needed(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.sampling_matrix is not None:
            if (h, w) != self.dense_shape:
                raise ValueError(
                    f"LearnableSamplingMatrix expected spatial shape {self.dense_shape} but received {(h, w)}"
                )
            return

        if h % self.reduction != 0 or w % self.reduction != 0:
            raise ValueError(
                f"Input spatial shape ({h}, {w}) must be divisible by reduction factor {self.reduction}"
            )

        coarse_h = h // self.reduction
        coarse_w = w // self.reduction
        init = torch.randn(coarse_h * coarse_w, h * w, device=device, dtype=dtype) * (1.0 / math.sqrt(h * w))
        self.sampling_matrix = nn.Parameter(init)
        self.dense_shape = (h, w)
        self.coarse_shape = (coarse_h, coarse_w)

    def _normalized_weights(self) -> torch.Tensor:
        return torch.softmax(self.sampling_matrix, dim=-1)

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        self._initialize_if_needed(h, w, x.device, x.dtype)
        weights = self._normalized_weights()  # Nc x Nd
        flat = x.reshape(b, c, -1)
        coarse = torch.einsum('bcn,kn->bck', flat, weights)
        return coarse.reshape(b, c, *self.coarse_shape)

    def upsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.sampling_matrix is None:
            raise RuntimeError("Sampling matrix has not been initialized. Call downsample before upsample.")
        b, c, h, w = x.shape
        if (h, w) != self.coarse_shape:
            raise ValueError(
                f"LearnableSamplingMatrix expected coarse shape {self.coarse_shape} but received {(h, w)}"
            )
        weights = self._normalized_weights()  # Nc x Nd
        dense = torch.einsum('bck,nk->bcn', x.reshape(b, c, -1), weights.transpose(0, 1))
        return dense.reshape(b, c, *self.dense_shape)


class LearnablePointDownsample(nn.Module):
    """
    Channel aware downsampling that first projects channels and then applies a
    learnable dense-to-coarse sampler.
    """

    def __init__(self, in_channels: int, keep_dim: bool = False, sampler: LearnableSamplingMatrix = None, reduction: int = 2):
        super().__init__()
        out_channels = in_channels if keep_dim else in_channels * 2
        self.channel_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU()
        )
        self.sampler = sampler if sampler is not None else LearnableSamplingMatrix(reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_proj(x)
        return self.sampler.downsample(x)


class LearnablePointUpsample(nn.Module):
    """
    Upsampling counterpart that reuses the sampling matrix learned during the
    downsampling stage.
    """

    def __init__(self, in_channels: int, keep_dim: bool = False, sampler: LearnableSamplingMatrix = None, reduction: int = 2):
        super().__init__()
        if not keep_dim and in_channels % 2 != 0:
            raise ValueError("in_channels must be divisible by 2 when keep_dim is False.")
        out_channels = in_channels if keep_dim else in_channels // 2
        self.sampler = sampler if sampler is not None else LearnableSamplingMatrix(reduction=reduction)
        self.channel_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sampler.upsample(x)
        return self.channel_proj(x)


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
            
        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
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