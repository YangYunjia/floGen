'''
    this script is to define the encoder and decoder used in this VAE
    @Author: Yang Yunjia

    Ref.
    1. github.com/julianstastny/VAE-ResNet18-PyTorch
    2. github.com/weiaicunzai/pytorch-cifar100
    3. Deng Kaiwen, Thesis for master degree, 2017

'''


import numpy as np
import copy

import torch
from torch import nn
from torch.nn import functional as F
from functools import reduce

from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')

class encoder(nn.Module):

    def __init__(self,
                 in_channels: int):
        '''
        parent class for encoder

        variables:
         - in_channels:     number of channel of the input
         - last_size:        size of the output flatten data [B x N]

        '''
        super().__init__()
        self.in_channels = in_channels
        self.last_flat_size = 0

    def forward(self, input: Tensor) -> Tensor:
        # print(input.size())

        return torch.flatten(input, start_dim=1)

class mlpEncoder(encoder):
    '''
    encoder for pressure distribution (only mlp is need)
    '''

    def __init__(self, in_channels, 
                 conv1d: dict = None,
                 hidden_dims: List = [1024, 512, 128]) -> None:
        
        super().__init__(in_channels)
        self.last_flat_size = hidden_dims[-1]

        layers = []

        h0 = in_channels

        for h in hidden_dims:
            layers.append(nn.Linear(h0, h))
            layers.append(nn.LeakyReLU())
            h0 = h

        self.mlp = nn.Sequential(*layers)

    def forward(self, input):
        return self.mlp(torch.flatten(input, start_dim=1))


class conv1dEncoder(encoder):

    def __init__(self, in_channels, last_size,
                 hidden_dims: List = [32, 64, 128],
                 kernel_sizes: List = [3, 3, 3],
                 strides: List = [2, 2, 2],
                 paddings: List = [1, 1, 1],
                 pool_kernels: List = [3, 3, 3],
                 pool_strides: List = [2, 2, 2]
                 ) -> None:
        
        super().__init__(in_channels)
        self.last_flat_size = hidden_dims[-1] * last_size[0]

        layers = []

        h0 = in_channels

        for h, k, s, p, kp, sp in zip(hidden_dims, kernel_sizes, strides, paddings, pool_kernels, pool_strides):
            
            layers.append(nn.Conv1d(in_channels=h0, out_channels=h, kernel_size=k, stride=s, padding=p, bias=True))
            layers.append(nn.LeakyReLU())
            layers.append(nn.AvgPool1d(kernel_size=kp, stride=sp))
            h0 = h

        self.convs = nn.Sequential(*layers)

    def forward(self, input):
        result = self.convs(input)
        # print(result.size())
        # raise
        return super().forward(result)



class conv2dEncoder(encoder):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List,
                 kernel_sizes: List,
                 strides: List,
                 paddings: List,
                 max_pools: List,
                 **kwargs) -> None:
        
        super().__init__(in_channels, latent_dim)
        modules = []

        # for k, s, p, mp in zip(kernel_sizes, strides, paddings, max_pools):
        # self.featuremap_size = pow(2, len(hidden_dims) + sum(max_pool is not None))

        # Build Encoder
        # print(hidden_dims, max_pools, kernel_sizes, strides, paddings)
        for h_dim, max_pool, kernel_size, stride, padding in zip(hidden_dims, max_pools, kernel_sizes, strides, paddings):
            if max_pool is not None:
                modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(h_dim),
                    nn.MaxPool2d(kernel_size=max_pool[0], stride=max_pool[1]),
                    nn.LeakyReLU())
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                kernel_size= kernel_size, stride=stride, padding=padding),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
    
    def forward(self, input):

        result = self.encoder(input)

        return super().forward(result)
    
class Resnet18Encoder(encoder):

    def __init__(self,
                 in_channels: int,
                 last_size: tuple = None,
                 hidden_dims: List = [16, 32, 64, 128, 256],
                 num_blocks: List = [2, 2, 2, 2],
                 strides: List = [2, 2, 2, 2, 2],
                 preactive: bool = False,
                 **kwargs) -> None:
        
        super().__init__(in_channels)

        self.preactive = preactive

        if num_blocks is None:
            num_blocks = [2 for _ in hidden_dims[1:]]
        
        if strides is None:
            strides = [2 for _ in hidden_dims]
        
        self.layer_in_channels = hidden_dims[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.layer_in_channels, kernel_size=3, stride=strides[0], padding=1, bias=False),
            nn.BatchNorm2d(self.layer_in_channels),
            nn.LeakyReLU()
        ) 
        
        block_layers = []
        for i in range(4):
            block_layers.append(self._make_layer(hidden_dims[i+1], num_blocks[i], strides[i+1]))
        
        self.blocks = nn.Sequential(*block_layers)

        if last_size is not None:
            self.adap = nn.AdaptiveAvgPool2d(last_size)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for st in strides:
            layers.append(BasicEncodeBlock(self.layer_in_channels, out_channels, st, self.preactive))
            self.layer_in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, input):
        result = self.conv1(input)
        result = self.blocks(result)

        # print('Encoder last layer input:', result.size())

        result = self.adap(result)

        # print('Encoder last layer input:', result.size())

        
        return super().forward(result)

        
class BasicEncodeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, preactive=False):

        super().__init__()

        self.preactive = preactive

        # out_channels = in_channels * stride

        if preactive:
            self.main = nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
                )       
        else:
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            if preactive:
                self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
    
    def forward(self, input):
        result = self.main(input) + self.shortcut(input)

        if not self.preactive:
            result = torch.relu(result)

        return result

class IntpConv2d(nn.Module):

    '''
    Use a `F.interpolate` layer and a `nn.Conv2d` layer to resize the image.

    paras
    ===


    '''

    def __init__(self, in_channels, out_channels, kernel_size, dim=2, size=None, scale_factor=None, mode='bilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode
        if dim == 1:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=1)
        elif dim == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, input):
        result = F.interpolate(input, size=self.size, scale_factor=self.scale_factor, mode=self.mode)
        result = self.conv(result)

        return result


class decoder(nn.Module):

    def __init__(self,
                 out_channels: int,
                 last_dim: int,
                 last_size: List,
                 final_layer: str = 'vanilla',
                 recon_mesh: bool = False):

        super().__init__()
        
        self.recon_mesh = recon_mesh
        self.fl_type = final_layer
        self.last_size = last_size
        self.out_channels = out_channels

        # print('last_dim', last_dim)

        # if not reconstruct mesh, minus 2 degrees from reconstruction
        fl_type = self.fl_type
        if fl_type == 'vanilla':
            fl_in_channels = last_dim
        elif fl_type == 'realmesh':
            fl_in_channels = last_dim + 2
        else:
            raise AttributeError
        
        fl_out_channels = out_channels

        self.fl = nn.Sequential(
                            nn.Conv2d(fl_in_channels, 
                                    out_channels=fl_out_channels,
                                    kernel_size=3, 
                                    padding=1))

    def forward(self, input: Tensor, **kwargs) -> Tensor:

        if self.fl_type == 'realmesh':
            input = torch.cat([kwargs['realmesh'], input], dim=1)
        
        result = self.fl(input)
        return result

class mlpDecoder(nn.Module):
    '''
    encoder for pressure distribution (only mlp is need)
    '''

    def __init__(self, in_channels, out_sizes,
                 hidden_dims: List = [128, 512, 1024]) -> None:
        
        super().__init__()
        
        layers = []
        self.last_flat_size = hidden_dims[0]
        self.out_sizes = tuple([-1] + out_sizes)

        h0 = in_channels

        for h in hidden_dims:
            layers.append(nn.Linear(h0, h))
            layers.append(nn.LeakyReLU())
            h0 = h

        self.mlp = nn.Sequential(*layers)

        out_channels = reduce(lambda x, y: x*y, out_sizes)

        self.fl = nn.Sequential(nn.Linear(h0, out_channels))

    def forward(self, input):

        result = self.fl(self.mlp(input))

        return result.view(self.out_sizes)

class conv1dDecoder(nn.Module):

    def __init__(self, out_channels, last_size,
                 hidden_dims: List = [128, 64, 32, 16],
                 sizes: List = [26, 101, 401]
                 ) -> None:
        
        super().__init__()
        self.last_size = tuple([-1] + [hidden_dims[0]] + last_size)
        self.last_flat_size = int(hidden_dims[0] * last_size[0])

        layers = []

        h0 = hidden_dims[0]

        for h, s in zip(hidden_dims[1:], sizes):
            layers.append(IntpConv2d(in_channels=h0, out_channels=h, kernel_size=3, dim=1, size=s, mode='linear'))
            layers.append(nn.LeakyReLU())
            h0 = h

        self.convs = nn.Sequential(*layers)

        self.last_conv = nn.Conv1d(hidden_dims[-1], out_channels=out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, input):
        input = input.view(self.last_size)
        result = self.convs(input)
        return self.last_conv(result)

class Resnet18Decoder(decoder):

    def __init__(self,
                 in_channels: int,
                 last_size: tuple,
                 hidden_dims: List = [16, 16, 32, 64, 128, 256],
                 num_blocks: List = [2, 2, 2, 2, 2],
                 strides: List = [2, 2, 2, 2, 2, 2],
                 preactive: bool = False,
                 **kwargs):
        
        super().__init__(in_channels, hidden_dims[0])
        
        self.preactive = preactive

        if num_blocks is None:
            num_blocks = [2 for _ in hidden_dims[1:]]
        
        if strides is None:
            strides = [2 for _ in hidden_dims]

        self.layer_in_channels = hidden_dims[-1]

        self.conv0 = nn.Sequential(
            IntpConv2d(self.layer_in_channels, self.layer_in_channels, kernel_size=3, size=(11,3)),
            nn.BatchNorm2d(self.layer_in_channels),
            nn.LeakyReLU()
        )

        block_layers = []
        for i in range(len(num_blocks)):
            block_layers.append(self._make_layer(hidden_dims[-i-2], num_blocks[-i-1], strides[-i-2]))
        
        self.blocks = nn.Sequential(*block_layers)

        self.conv1 = nn.Sequential(
            IntpConv2d(self.layer_in_channels, self.layer_in_channels, kernel_size=3, size=last_size),
            nn.BatchNorm2d(self.layer_in_channels),
            nn.LeakyReLU()
        )
        

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for st in strides:
            layers.append(BasicDecodeBlock(self.layer_in_channels, out_channels, st, self.preactive))
            self.layer_in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, input):
        input = input.view(tuple([-1] + self.last_size)) # 编码器最后全连接层之前的最后的大小，第一个数字是batch的大小

        input = self.conv0(input)

        input = self.blocks(input)

        # print('Decoder last layer input:', input.size())

        result = self.conv1(input)
        
        return super().forward(result)

class BasicDecodeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, preactive=False):

        super().__init__()

        self.preactive = preactive

        # out_channels = int(in_channels / stride)

        if stride == 1:

            if preactive:
                self.main = nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
                )
            else:
                self.main = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            
            self.shortcut = nn.Sequential()

        else:
            if preactive:
                self.main = nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.LeakyReLU(),
                    IntpConv2d(in_channels, out_channels, kernel_size=3, scale_factor=stride)
                )
                self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    nn.LeakyReLU(),
                    IntpConv2d(in_channels, out_channels, kernel_size=3, scale_factor=stride)
                )
            
            else:
                self.main = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.LeakyReLU(),
                    IntpConv2d(in_channels, out_channels, kernel_size=3, scale_factor=stride),
                    nn.BatchNorm2d(out_channels)
                )
                self.shortcut = nn.Sequential(
                    IntpConv2d(in_channels, out_channels, kernel_size=3, scale_factor=stride),
                    nn.BatchNorm2d(out_channels)
                )
    
    def forward(self, input):
        result = self.main(input) + self.shortcut(input)

        if not self.preactive:
            result = torch.relu(result)

        return result

class conv2dDecoder(decoder):

    def __init__(self,
				 in_channels: int,
                 hidden_dims: List,
                 kernel_sizes: List,
                 strides: List,
                 paddings: List,
                 max_pools: List,
                 **kwargs) -> None:

        super().__init__(in_channels=in_channels, last_dim=hidden_dims[0], **kwargs)
        
        modules = []

        for i in range(1, len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[-i],
                                       hidden_dims[-i-1],
                                       kernel_size=kernel_sizes[-i],
                                       stride=strides[-i],
                                       padding=paddings[-i],
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[-i-1]),
                    nn.LeakyReLU())
            )
        
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[1],
                                    hidden_dims[0],
                                    kernel_size=kernel_sizes[1],
                                    stride=strides[1],
                                    padding=paddings[1],
                                    output_padding=0),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.LeakyReLU()) 
        )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[0],
                                    hidden_dims[0],
                                    kernel_size=kernel_sizes[0],
                                    stride=strides[0],
                                    padding=paddings[0]),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.LeakyReLU()) 
        ) # N layer

        self.decoder = nn.Sequential(*modules)

    def forward(self, input):
        input = input.view(tuple([-1] + self.last_size)) # 编码器最后全连接层之前的最后的大小，第一个数字是batch的大小
        result = self.decoder(input)
        return super().forward(result)


def show_size(func):

    def wraped_func(*args, **kwargs):

        if 'input' in kwargs:

            size_in = kwargs['input'].size()

        out = func(*args, **kwargs)
        
        size_out = out.size()

        print('     ', size_in, '       ', size_out)

