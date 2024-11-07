'''
Oct. 27 2024 
Author: Yunjia Yang

The basic models of transformer-based models

reference
---
https://github.com/thuml/Transolver/ (Transolver_Structured_Mesh_2D.py)

'''
import torch
import torch.nn as nn
import numpy as np
from functools import reduce

from flowvae.base_model.mlp import mlp
from flowvae.base_model.utils import IntpConv

class Physics_Attention(nn.Module):
    
    def __init__(self, dim, heads=8, dim_head=64, slice_num=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head        # Ch
        self.heads = heads              # Nh
        self.slice_num = slice_num      # M
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Identity()
        self.in_project_fx = nn.Identity()
        self._prepare_projection(dim, inner_dim)

        self.in_project_slice = nn.Linear(dim_head, slice_num)  # Ch -> M
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def _prepare_projection(self, dim, inner_dim):
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_mid, fx_mid, N_, N0_ = self._forward_slice(x)
        
        slice_weights = self.softmax(self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B Nh N M 
        slice_norm = slice_weights.sum(2)  # B Nh M

        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)    # B Nh M Ch
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B Nh M Ch

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights) # B Nh N Ch
        out_x = out_x.permute(0, 2, 1, 3).reshape(-1, *N0_, self.heads * self.dim_head) # B N Nh*Ch
        return self.to_out(out_x)
    
    def _forward_slice(self, x: torch.Tensor):
        N0_ = x.shape[1:-1]
        N_ = reduce(lambda x, y: x*y, N0_)
        fx_mid = self.in_project_fx(x).reshape(-1, N_, self.heads, self.dim_head).permute(0, 2, 1, 3)  # B H N C
        x_mid = self.in_project_x(x).reshape(-1, N_, self.heads, self.dim_head) .permute(0, 2, 1, 3)  # B H N C
        return x_mid, fx_mid, N_, N0_
    
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
            downsampling: nn.Module = nn.Identity(),
            upsampling: nn.Module = nn.Identity()
    ) -> None:
        
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(hidden_dim)
        if mesh_type == '2d':
            self.Attn = Physics_Attention_2D(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                         dropout=dropout, slice_num=slice_num)
        elif mesh_type == '3d':
            self.Attn = Physics_Attention_3D(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                         dropout=dropout, slice_num=slice_num)
        elif mesh_type == 'point':
            self.Attn = Physics_Attention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                         dropout=dropout, slice_num=slice_num)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = mlp(in_features=hidden_dim, out_features=hidden_dim, hidden_dims=[hidden_dim * mlp_ratio], last_actv=False)   # , res=False, act=act
        
        self.upsampling = upsampling
        self.dnsampling = downsampling

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        # upsampling
        fx = self.upsampling(fx)
        # attention
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        # downsampling
        fx = self.dnsampling(fx)
        return fx
    
    def get_slice_weight(self, fx: torch.Tensor) -> torch.Tensor:
        '''
        return:
        ---
        
        `torch.Tensor`:   B Nh H W M 
        
        '''
        return self.Attn.get_slice_weight(self.ln_1(fx))

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
                 u_shape: bool = False,
                 dropout=0.0,
                 Time_Input=False,
                 act='gelu',
                 ref=8,
                 unified_pos=False,
                 device: str = 'cuda:0'
                 ):
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

        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))
            
        
        if not u_shape:
            self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                        dropout=dropout,
                                                        act=act,
                                                        mlp_ratio=mlp_ratio,
                                                        slice_num=slice_num,
                                                        mesh_type=mesh_type)
                                        for _ in range(n_layers)])
        else:
            self.is_compiled = False
            # build the downsampling part of the model
            self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                        dropout=dropout,
                                                        act=act,
                                                        mlp_ratio=mlp_ratio,
                                                        slice_num=slice_num,
                                                        mesh_type=mesh_type,
                                                        downsampling=Downsampling())
                                        for _ in range(int(n_layers / 2))])
                    
        self.last_layer = nn.Sequential(nn.LayerNorm(n_hidden), nn.Linear(n_hidden, out_dim))
        
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

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

        fx = self._process(*args, **kwargs)
        for block in self.blocks:
            fx = block(fx)
            
        fx = self.last_layer(fx)
        return fx
    
    def _process(self, x: torch.Tensor, fx: torch.Tensor, T=None) -> torch.Tensor:
        
        # if self.unified_pos:
        #     x = self.pos.repeat(x.shape[0], 1, 1, 1).reshape(x.shape[0], self.H * self.W, self.ref * self.ref)
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        # if T is not None:
        #     Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
        #     Time_emb = self.time_fc(Time_emb)
        #     fx = fx + Time_emb
        
        return fx
    
    def compile(self, *args, **kwargs) -> None:
        '''
        Calculate the feature map size after downsampling, for initialize the upsampling path
        
        '''
        
        print('--------------------')
        print('Compiling the model with a batch of training data')
        sizes = []
        fx = self._process(*args, **kwargs)
        
        for ib in range(int(self.n_layers / 2)):
            sizes.append(fx.shape[1:-1])
            print(f'    Encoder Layer {ib:d}: {fx.shape} ->')
            fx = self.blocks[ib](fx)
            
        block_list_n = []
        if self.n_layers % 2 > 0:
            # middle block
            print(f'    Middle Layer:')
            block_list_n.append(Transolver_block(num_heads=self.n_head, hidden_dim=self.n_hidden,
                                                        mlp_ratio=self.mlp_ratio,
                                                        slice_num=self.slice_num,
                                                        mesh_type=self.mesh_type))
        for ib in range(int(self.n_layers / 2)):
            print(f'    Decoder Layer {ib:d}: -> {sizes[-ib-1]}')
            block_list_n.append(Transolver_block(num_heads=self.n_head, hidden_dim=self.n_hidden,
                                                        mlp_ratio=self.mlp_ratio,
                                                        slice_num=self.slice_num,
                                                        mesh_type=self.mesh_type,
                                                        upsampling=Upsampling(size=sizes[-ib-1])))
            
        self.blocks.extend(nn.ModuleList(block_list_n))
        self.is_compiled = True
        print('--------------------')
    
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