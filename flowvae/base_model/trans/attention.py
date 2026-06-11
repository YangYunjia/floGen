import torch
import torch.nn as nn
import numpy as np
from functools import reduce

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
    
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            slice_num=64,
            dropout=0.,
            add_mesh=0,
            dual_slices=False,
            downsample_strategy: str = 'matrix',
            downsample_scale: int = 2):
        '''
        `add_mesh`
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
        self.add_mesh = add_mesh  # whether to add 3D mesh as the only `x` to calculate weights
        self.use_dual_slice_projection = dual_slices
        self.downsample_strategy = downsample_strategy
        self.downsample_scale = downsample_scale
        self.weight_reduce_kernels = nn.ParameterDict()
        self.weight_expand_kernels = nn.ParameterDict()
        self._prepare_projection(dim, inner_dim)

        self.in_project_slice = nn.Linear(dim_head, slice_num)  # Ch -> M
        torch.nn.init.orthogonal_(self.in_project_slice.weight)  # use a principled initialization
        if self.use_dual_slice_projection:
            self.in_project_slice_deslice = nn.Linear(dim_head, slice_num)
            torch.nn.init.orthogonal_(self.in_project_slice_deslice.weight)
        else:
            self.in_project_slice_deslice = self.in_project_slice
        
        # the heads are difference in getting projection weights and values to be projected
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
    
    def _prepare_projection(self, dim, inner_dim):
        if self.add_mesh > 0:
            self.in_project_x  = nn.Linear(dim + self.add_mesh, inner_dim)
            self.in_project_fx = nn.Linear(dim + self.add_mesh, inner_dim)
        else:
            self.in_project_x = nn.Linear(dim, inner_dim)
            self.in_project_fx = nn.Linear(dim, inner_dim)
    
    def forward(self, x: torch.Tensor, x_cross: torch.Tensor) -> torch.Tensor:

        # calculate slices `x`, `fx` for self attention
        x_mid, fx_mid, N_, N0_ = self._forward_slice(x) 
        slice_token, slice_weights, slice_weights_deslice = self._project_token(x_mid, fx_mid, N_)  # B Nh M Ch,   B Nh N M
        
        # calculate `x` for cross attention, the `fx` (value) remains the same for cross attention
        if x_cross is not None:
            x_mid_cross, fx_mid_cross, _, _ = self._forward_slice(x_cross)
            slice_token_cross, _, _ = self._project_token(x_mid_cross, fx_mid_cross, N_)   # B Nh M Ch
        else:
            slice_token_cross = slice_token

        ### (2) Attention among slice tokens
        q = self.to_q(slice_token)
        k = self.to_k(slice_token_cross)
        v = self.to_v(slice_token_cross)
        out_slice_token = self._calculate_attention(q, k, v)  # B Nh M Ch

        ### (3) Deslice
        return self._deslice(out_slice_token, slice_weights_deslice, N0_)
    
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
        slice_weights = self._compute_slice_weights(self.in_project_slice, x_mid, N_)
        if self.use_dual_slice_projection:
            slice_weights_deslice = self._compute_slice_weights(self.in_project_slice_deslice, x_mid, N_)
        else:
            slice_weights_deslice = slice_weights

        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)    # B Nh M Ch
        
        return slice_token, slice_weights, slice_weights_deslice

    def _compute_slice_weights(self, projection_layer: nn.Module, x_mid: torch.Tensor, N_: int) -> torch.Tensor:
        
        slice_weights = self.softmax(projection_layer(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B Nh N M 
        slice_norm    = slice_weights.sum(2)  # B Nh M
        # Nov. 8 2024 -> move non-dimensional for slice here -> the weights used for reconstruct will be different
        slice_weights = slice_weights / ((slice_norm + 1e-5)[:, :, None, :].repeat(1, 1, N_, 1))
        return slice_weights
    
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
        slice_weights = self._compute_slice_weights(self.in_project_slice, x_mid, N_)  # B Nh N M 
        
        return slice_weights.reshape(-1, self.heads, *N0_, self.slice_num)

    def downsample(self, x: torch.Tensor, x_cross: torch.Tensor = None):
        '''
        Downsample the physical field using the slice projection. Two strategies
        are supported:
            - `token`: reshape slice tokens directly to a coarse grid
            - `matrix`: contract the slice-weight matrix before deslicing
        '''
        x_mid, fx_mid, N_, N0_ = self._forward_slice(x)
        slice_token, slice_weights, slice_weights_deslice = self._project_token(x_mid, fx_mid, N_)

        if x_cross is not None:
            x_mid_cross, fx_mid_cross, _, _ = self._forward_slice(x_cross)
            slice_token_cross, _, _ = self._project_token(x_mid_cross, fx_mid_cross, N_)
        else:
            slice_token_cross = slice_token

        q = self.to_q(slice_token)
        k = self.to_k(slice_token_cross)
        v = self.to_v(slice_token_cross)
        out_slice_token = self._calculate_attention(q, k, v)

        coarse_shape = self._scale_shape(N0_)
        strategy = self.downsample_strategy.lower()
        coarse_points = reduce(lambda a, b: a * b, coarse_shape) if coarse_shape else 1
        if strategy == 'token':
            if coarse_points != self.slice_num:
                raise ValueError("token strategy requires slice_num to equal the coarse spatial size.")
            coarse = self.to_out(self._slice_token_to_feature(out_slice_token, coarse_shape))
            sampling_state = {
                'mode': 'token',
                'original_shape': tuple(N0_),
                'coarse_shape': tuple(coarse_shape),
                'slice_weights': slice_weights_deslice
            }
        elif strategy == 'matrix':
            if coarse_points == 0 or N_ % coarse_points != 0:
                raise ValueError("matrix strategy requires a valid reduction factor.")
            reduction_factor = N_ // coarse_points
            reduced_weights = self._reduce_slice_weights(slice_weights_deslice, reduction_factor)
            coarse = self._deslice(out_slice_token, reduced_weights, coarse_shape)
            sampling_state = {
                'mode': 'matrix',
                'original_shape': tuple(N0_),
                'coarse_shape': tuple(coarse_shape),
                'slice_weights': reduced_weights,
                'reduction_factor': reduction_factor
            }
        else:
            raise ValueError(f"Unknown downsample strategy: {self.downsample_strategy}")

        return coarse, sampling_state

    def upsample(self, x: torch.Tensor, sampling_state: dict) -> torch.Tensor:
        '''
        Reconstruct the fine-resolution field from cached slice weights.
        '''
        if sampling_state is None:
            raise ValueError("sampling_state must be provided for upsample.")

        mode = sampling_state.get('mode', self.downsample_strategy).lower()
        original_shape = sampling_state['original_shape']
        coarse_shape = sampling_state['coarse_shape']
        if mode == 'token':
            slice_weights = sampling_state['slice_weights']
            slice_token = self._spatial_to_slice_token(x, coarse_shape)
            return self._deslice(slice_token, slice_weights, original_shape)
        elif mode == 'matrix':
            slice_weights = sampling_state['slice_weights']
            reduction_factor = sampling_state['reduction_factor']
            expanded_weights = self._expand_slice_weights(slice_weights, reduction_factor)
            slice_token = self._spatial_to_slice_token(x, coarse_shape)
            return self._deslice(slice_token, expanded_weights, original_shape)
        else:
            raise ValueError(f"Unknown upsample strategy: {mode}")

    def _scale_shape(self, original_shape):
        scale = self.downsample_scale
        if isinstance(scale, int):
            if scale <= 1:
                return list(original_shape)
            scales = [scale] * len(original_shape)
        else:
            scales = list(scale)
            if len(scales) == 1:
                scales = scales * len(original_shape)
            elif len(scales) != len(original_shape):
                raise ValueError("downsample_scale must match the spatial dimensionality.")
            if all(s <= 1 for s in scales):
                return list(original_shape)
        scaled = []
        for size, scale in zip(original_shape, scales):
            if size % scale != 0:
                raise ValueError(f"Spatial dimension {size} is not divisible by scale {scale}.")
            scaled.append(max(1, size // scale))
        return scaled

    def _slice_token_to_feature(self, slice_token: torch.Tensor, spatial_shape):
        b = slice_token.shape[0]
        return slice_token.permute(0, 2, 1, 3).reshape(b, *spatial_shape, self.heads * self.dim_head)

    def _spatial_to_slice_token(self, x: torch.Tensor, spatial_shape):
        if tuple(x.shape[1:-1]) != tuple(spatial_shape):
            raise ValueError(f"Expected spatial shape {spatial_shape}, but received {tuple(x.shape[1:-1])}.")
        if x.shape[-1] != self.heads * self.dim_head:
            raise ValueError("Channel dimension must equal heads * dim_head for reconstruction.")
        b = x.shape[0]
        tokens = x.reshape(b, -1, self.heads, self.dim_head)
        return tokens.permute(0, 2, 1, 3)

    def _reduce_slice_weights(self, slice_weights: torch.Tensor, factor: int) -> torch.Tensor:
        if factor <= 1:
            return slice_weights
        B, Nh, N, M = slice_weights.shape
        if N % factor != 0:
            raise ValueError(f"Cannot reduce {N} points with factor {factor}.")
        kernel = self._get_kernel(factor, kind='reduce')
        kernel = torch.softmax(kernel, dim=0)
        weights = slice_weights.reshape(B, Nh, N // factor, factor, M)
        reduced = (weights * kernel.view(1, 1, 1, factor, 1)).sum(dim=3)
        reduced_sum = reduced.sum(dim=2, keepdim=True) + 1e-6
        return reduced / reduced_sum

    def _expand_slice_weights(self, slice_weights: torch.Tensor, factor: int) -> torch.Tensor:
        if factor <= 1:
            return slice_weights
        kernel = self._get_kernel(factor, kind='expand')
        kernel = torch.softmax(kernel, dim=0)
        expanded = slice_weights[..., None, :] * kernel.view(1, 1, 1, factor, 1)
        expanded = expanded.reshape(slice_weights.shape[0], slice_weights.shape[1], -1, slice_weights.shape[-1])
        expanded_sum = expanded.sum(dim=2, keepdim=True) + 1e-6
        return expanded / expanded_sum

    def _get_kernel(self, factor: int, kind: str) -> nn.Parameter:
        if kind == 'reduce':
            container = self.weight_reduce_kernels
        else:
            container = self.weight_expand_kernels
        key = str(factor)
        if key not in container:
            if kind == 'reduce':
                init = torch.ones(factor) / factor
            else:
                init = torch.ones(factor)
            container[key] = nn.Parameter(init)
        return container[key]

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
    def __init__(self, dim, kernel=3, **kwargs):
        self.kernel = kernel
        super().__init__(dim, **kwargs)
        
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

    def __init__(self, dim, kernel=3, **kwargs):
        self.kernel = kernel
        super().__init__(dim, **kwargs)
        
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
