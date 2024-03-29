'''
    this script is to define the encoder and decoder used in this VAE
    @Author: Yang Yunjia

    Ref.
    1. github.com/julianstastny/VAE-ResNet18-PyTorch
    2. github.com/weiaicunzai/pytorch-cifar100
    3. Deng Kaiwen, Thesis for master degree, 2017

'''


import torch
from torch import nn
from torch.nn import functional as F
from functools import reduce
from torch.autograd import Variable

from typing import Tuple, List, Dict, NewType, Callable
Tensor = NewType('Tensor', torch.tensor)

import copy

def _check_kernel_size_consistency(kernel_size):
    if not (isinstance(kernel_size, tuple) or
            (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
        raise ValueError('`kernel_size` must be tuple or list of tuples')

def _extend_for_multilayer(param, num_layers: int):
    if not isinstance(param, list):
        param = [param] * num_layers
    return param

class IntpConv(nn.Module):

    '''
    Use a `F.interpolate` layer and a `nn.Conv2d` layer to resize the image.

    paras
    ===


    '''

    def __init__(self, in_channels, out_channels, kernel_size, size=None, scale_factor=None, mode=None):
        super().__init__()
        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode
        self.conv = nn.Identity()
        self.padding = int((kernel_size - 1) / 2)

    def forward(self, inpt):
        result = F.interpolate(inpt, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        result = self.conv(result)
        return result

class IntpConv1d(IntpConv):

    def __init__(self, in_channels, out_channels, kernel_size, size=None, scale_factor=None, mode=None, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, size, scale_factor, mode)
        if mode is None:    self.mode = 'linear'
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=self.padding, bias=bias)

class IntpConv2d(IntpConv):

    def __init__(self, in_channels, out_channels, kernel_size, size=None, scale_factor=None, mode=None, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, size, scale_factor, mode)
        if mode is None:    self.mode = 'bilinear'
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=self.padding, bias=bias)

class Convbottleneck(nn.Module):

    def __init__(self, h0, out_channels, kernel_size = 3, stride = 1, padding = 1, last_actv = False, basic_layers = {}) -> None:
        super().__init__()

        layers = []
        layers.append(basic_layers['conv'](h0, out_channels=h0, kernel_size=kernel_size, stride=stride, padding=padding))
        if basic_layers['bn'] != nn.Identity:  layers.append(basic_layers['bn'](h0))
        layers.append(basic_layers['actv']())
        layers.append(basic_layers['conv'](h0, out_channels=out_channels, kernel_size=1, stride=1, padding=0))
        if basic_layers['bn'] != nn.Identity:  layers.append(basic_layers['bn'](h0))
        if last_actv:                       layers.append(basic_layers['actv']())

        self.convs = nn.Sequential(*layers)

    def forward(self, input):
        return self.convs(input)

default_basic_layers_1d = {
    'conv':     nn.Conv1d,
    'actv':     nn.LeakyReLU,
    'deconv':   IntpConv1d,
    'pool':     nn.AvgPool1d,
    'bn':       nn.Identity,
    'preactive':  False,
    'last_actv':  nn.Identity
}

default_basic_layers_2d = {
    'conv':     nn.Conv2d,
    'actv':     nn.LeakyReLU,
    'deconv':   IntpConv2d,
    'pool':     nn.AvgPool2d,
    'bn':       nn.Identity,
    'preactive':  False,
    'last_actv':  nn.Identity
}

def _update_basic_layer(basic_layers: dict, dimension: int = 1, batchnorm: bool = False):

    if dimension == 1:   
        basic_layers_n = copy.deepcopy(default_basic_layers_1d)
        if batchnorm:   basic_layers_n['bn'] = nn.BatchNorm1d       # if not set batchnorm to True, the batchnorm will
                                                                    # be a Identity layer that takes any arguments as 
                                                                    # input, and don't do anything to flux
    elif dimension == 2: 
        basic_layers_n = copy.deepcopy(default_basic_layers_2d)
        if batchnorm:   basic_layers_n['bn'] = nn.BatchNorm2d

    for key in basic_layers: basic_layers_n[key] = basic_layers[key]

    return basic_layers_n

class Encoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 last_size: List,
                 hidden_dims: List) -> None:
        '''
        parent class for encoder

        variables:
         - in_channels:     number of channel of the input
         - last_size:        size of the output flatten data [B x N]

        '''
        super().__init__()
        self.in_channels = in_channels
        self.last_flat_size = hidden_dims[-1] * reduce(lambda x, y: x*y, last_size)
        self.is_unet = False

    def forward(self, inpt: Tensor) -> Tensor:
        # print(inpt.size())

        return torch.flatten(inpt, start_dim=1)

class mlpEncoder(Encoder):
    '''
    encoder for pressure distribution (only mlp is need)
    '''

    def __init__(self, in_channels, 
                 hidden_dims: List = [1024, 512, 128]) -> None:
        
        super().__init__(in_channels, last_size=[1], hidden_dims=hidden_dims)
        self.last_flat_size = hidden_dims[-1]

        layers = []

        h0 = in_channels

        for h in hidden_dims:
            layers.append(nn.Linear(h0, h))
            layers.append(nn.LeakyReLU())
            h0 = h

        self.mlp = nn.Sequential(*layers)

    def forward(self, inpt):
        return self.mlp(torch.flatten(inpt, start_dim=1))

class convEncoder(Encoder):

    def __init__(self, in_channels: int, last_size: List,
                 hidden_dims: List[int],
                 kernel_sizes: List[int] = 3,
                 strides: List[int] = 2,
                 paddings: List[int] = 1,
                 pool_kernels: List[int] = 3,
                 pool_strides: List[int] = 2,
                 dimension: int = 1,
                 basic_layers: Dict = {}
                 ) -> None:
        
        super().__init__(in_channels, last_size, hidden_dims)

        h0 = in_channels

        kernel_sizes = _extend_for_multilayer(kernel_sizes, len(hidden_dims))
        strides      = _extend_for_multilayer(strides,      len(hidden_dims))
        paddings     = _extend_for_multilayer(paddings,     len(hidden_dims))
        pool_kernels = _extend_for_multilayer(pool_kernels, len(hidden_dims))
        pool_strides = _extend_for_multilayer(pool_strides, len(hidden_dims))

        self.basic_layers = _update_basic_layer(basic_layers, dimension)

        # ** for old version **
        # layers = []
        # for h, k, s, p, kp, sp in zip(hidden_dims, kernel_sizes, strides, paddings, pool_kernels, pool_strides):
        #     layers.append(basic_layers['conv'](in_channels=h0, out_channels=h, kernel_size=k, stride=s, padding=p, bias=True))
        #     if basic_layers['bn'] != nn.Identity:  layers.append(basic_layers['bn'](h))
        #     layers.append(basic_layers['actv']())
        #     if kp > 0: layers.append(basic_layers['pool'](kernel_size=kp, stride=sp))
        #     h0 = h
        # # DO NOT change to Modular list, otherwise old saves will fails
        # self.convs = nn.Sequential(*layers)
        # ** for old version **

        _convs = []
        for h, k, s, p, kp, sp in zip(hidden_dims, kernel_sizes, strides, paddings, pool_kernels, pool_strides):
            layers = []
            layers.append(self.basic_layers['conv'](in_channels=h0, out_channels=h, kernel_size=k, stride=s, padding=p, bias=True))
            if self.basic_layers['bn'] != nn.Identity:  layers.append(self.basic_layers['bn'](h))
            layers.append(self.basic_layers['actv']())
            if isinstance(kp, tuple) or (kp > 0 and sp > 0): layers.append(self.basic_layers['pool'](kernel_size=kp, stride=sp))
            h0 = h
            _convs.append(nn.Sequential(*layers))
        
            self.convs = nn.ModuleList(_convs)

    def forward(self, inpt):
        for conv in self.convs:
            inpt = conv(inpt)
        
        # ** for old version **
        # inpt = self.convs(inpt)
        # ** for old version **

        # print(inpt.size(), len(self.convs))
        # raise
        return super().forward(inpt)

class convEncoder_Unet(convEncoder):

    def __init__(self, in_channels, last_size,
                 hidden_dims: List = [32, 64, 128],
                 kernel_sizes: List = 3,
                 strides: List = 2,
                 paddings: List = 1,
                 pool_kernels: List = 3,
                 pool_strides: List = 2,
                 dimension: int = 1,
                 basic_layers: Dict = {}
                 ) -> None:
        
        super().__init__(in_channels, last_size, hidden_dims, kernel_sizes, strides, paddings, pool_kernels, pool_strides, dimension, basic_layers)
        
        # for U-net
        self.is_unet = True
        self.feature_maps = []

    def forward(self, inpt):
        self.feature_maps = []
        self.feature_maps.append(inpt)
        for conv in self.convs:
            inpt = conv(inpt)
            self.feature_maps.append(inpt)
            # print(inpt.size())
        # raise
        return torch.flatten(inpt, start_dim=1)
           
class Resnet18Encoder(Encoder):

    def __init__(self,
                 in_channels: int,
                 last_size: List[int],
                 hidden_dims: List, #  = [16, 32, 64, 128, 256]
                 num_blocks: List = 2,
                 strides: List = 2,
                 preactive: bool = False, 
                 extra_first_conv: Tuple[int] = None,
                 force_last_size: bool = False,
                 dimension: int = 2,
                 batchnorm: bool = True,
                 basic_layers: Dict = {}) -> None:
        
        super().__init__(in_channels, last_size, hidden_dims)

        self.preactive = preactive
        self.basic_layers = _update_basic_layer(basic_layers, dimension=dimension, batchnorm=batchnorm)
        self.isbias = not batchnorm

        num_blocks = _extend_for_multilayer(num_blocks,  len(hidden_dims))
        strides    = _extend_for_multilayer(strides,     len(hidden_dims))
        
        h0 = self.in_channels
        if extra_first_conv is not None:
            h, k, s, p = extra_first_conv
            self.conv1 = nn.Sequential(
                self.basic_layers['conv'](in_channels=h0, out_channels=h, kernel_size=k, stride=s, padding=p, bias=self.isbias),
                self.basic_layers['bn'](h),
                self.basic_layers['actv']()) 
            h0 = h
        else:
            self.conv1 = nn.Identity()

        block_layers = []
        for h, nb, s in zip(hidden_dims, num_blocks, strides):
            block_layers.append(self._make_layer(in_channels=h0, out_channels=h, num_blocks=nb, stride=s))
            h0 = h

        self.blocks = nn.Sequential(*block_layers)

        if force_last_size:
            self.adap = nn.AdaptiveAvgPool2d(last_size)
        else:
            self.adap = nn.Identity()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for i_block, st in enumerate(strides):
            layers.append(BasicEncodeBlock(in_channels, out_channels, st, i_block, batchnorm=not self.isbias, basic_layers=self.basic_layers))

        return nn.Sequential(*layers)

    def forward(self, inpt):
        result = self.conv1(inpt)
        result = self.blocks(result)

        # print('Encoder last layer input:', result.size())

        result = self.adap(result)

        # print('Encoder last layer input:', result.size())

        
        return super().forward(result)
     
class BasicEncodeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, i_block, batchnorm, basic_layers):

        super().__init__()

        self.preactive = basic_layers['preactive']
        self.actv      = basic_layers['actv']
        self.isbias    = not batchnorm

        # out_channels = in_channels * stride
        if i_block > 0:

            if self.preactive:
                self.main = nn.Sequential(
                        basic_layers['bn'](in_channels),
                        self.actv(),
                        basic_layers['conv'](out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=self.isbias),
                        basic_layers['bn'](out_channels),
                        self.actv(),
                        basic_layers['conv'](out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=self.isbias)
                    )       
            else:
                self.main = nn.Sequential(
                    basic_layers['conv'](out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=self.isbias),
                    basic_layers['bn'](out_channels),
                    self.actv(),
                    basic_layers['conv'](out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=self.isbias),
                    basic_layers['bn'](out_channels)
                )
            self.shortcut = nn.Sequential()

        else:
            if self.preactive:
                self.main = nn.Sequential(
                        basic_layers['bn'](in_channels),
                        self.actv(),
                        basic_layers['conv'](in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=self.isbias),
                        basic_layers['bn'](out_channels),
                        self.actv(),
                        basic_layers['conv'](out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=self.isbias)
                    )     
                self.shortcut = nn.Sequential(
                    basic_layers['bn'](in_channels),
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=self.isbias)
                )
            else:
                self.main = nn.Sequential(
                    basic_layers['conv'](in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=self.isbias),
                    basic_layers['bn'](out_channels),
                    self.actv(),
                    basic_layers['conv'](out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=self.isbias),
                    basic_layers['bn'](out_channels)
                )
                self.shortcut = nn.Sequential(
                    basic_layers['conv'](in_channels, out_channels, kernel_size=1, stride=stride, bias=self.isbias),
                    basic_layers['bn'](out_channels)
                )
    
    def forward(self, inpt):
        # print(inpt.size())
        result = self.main(inpt) + self.shortcut(inpt)

        if not self.preactive:
            result = self.actv()(result)

        return result

class Decoder(nn.Module):

    def __init__(self,
                 out_channels: int):
                #  final_layer: str = 'vanilla',
                #  recon_mesh: bool = False):

        super().__init__()

        self.last_flat_size = 0         # the flattened data input to decoder
        self.inpt_shape = None          # reshape the flattened input to this shape
        self.out_channels = out_channels
        self.is_unet = False

    '''
        #* input mesh at the last layer of decoder
        self.recon_mesh = recon_mesh
        self.fl_type = final_layer
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

    def forward(self, inpt: Tensor, **kwargs) -> Tensor:

        if self.fl_type == 'realmesh':
            inpt = torch.cat([kwargs['realmesh'], inpt], dim=1)
        
        result = self.fl(inpt)
        return result
    '''

class mlpDecoder(nn.Module):
    '''
    encoder for pressure distribution (only mlp is need)
    '''

    def __init__(self, out_sizes,
                 hidden_dims: List = [128, 512, 1024]) -> None:
        
        super().__init__()
        
        layers = []
        self.last_flat_size = hidden_dims[0]
        self.out_sizes = out_sizes

        h0 = hidden_dims[0]

        for h in hidden_dims[1:]:
            layers.append(nn.Linear(h0, h))
            layers.append(nn.LeakyReLU())
            h0 = h

        self.mlp = nn.Sequential(*layers)

        out_channels = reduce(lambda x, y: x*y, out_sizes)

        self.fl = nn.Linear(h0, out_channels)

    def forward(self, inpt):

        result = self.fl(self.mlp(inpt))

        return result.view(tuple([-1] + self.out_sizes))

class convDecoder(Decoder):

    def __init__(self, out_channels: int, 
                 last_size: List[int],   # the H x W of first layer viewed from last_flat_size
                 hidden_dims: List[int], # Tuple = (128, 64, 32, 16)
                 sizes: List[int],       # Tuple = (26, 101, 401)
                 kernel_sizes: List[int] = None,
                 dimension: int = 1, 
                 last_conv: str = 'normal',
                 basic_layers: Dict = {}
                 ) -> None:
        
        super().__init__(out_channels)
        self.inpt_shape = tuple([-1] + [hidden_dims[0]] + last_size)
        self.last_flat_size = abs(reduce(lambda x, y: x*y, self.inpt_shape))
        if kernel_sizes is None: self.kernel_sizes = [3 for _ in hidden_dims]

        self.basic_layers = _update_basic_layer(basic_layers, dimension)

        layers = []

        h0 = hidden_dims[0]

        for h, s, k in zip(hidden_dims[1:], sizes, self.kernel_sizes[:-1]):
            layers.append(self.basic_layers['deconv'](in_channels=h0, out_channels=h, kernel_size=k, size=s))
            layers.append(self.basic_layers['actv']())
            h0 = h
        self.convs = nn.Sequential(*layers)

        if last_conv == 'normal':
            self.last_conv = self.basic_layers['conv'](hidden_dims[-1], out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        elif last_conv == 'bottleneck':
            self.last_conv = Convbottleneck(hidden_dims[-1], out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                                            last_actv=False, basic_layers=self.basic_layers)


    def forward(self, inpt: torch.Tensor):
        inpt = inpt.view(self.inpt_shape)
        result = self.convs(inpt)
        return self.last_conv(result)

class convDecoder_Unet(convDecoder):

    def __init__(self, out_channels,
                 last_size: List[int], 
                 hidden_dims: List[int], 
                 sizes: List[int], 
                 encoder_hidden_dims: List[int],
                 kernel_sizes: List[int] = None,
                 dimension: int = 1,
                 basic_layers: Dict = {}) -> None:
        
        unet_hidden_dims = [hidden_dims[i] + encoder_hidden_dims[i] for i in range(len(hidden_dims))]
        super().__init__(out_channels, last_size, unet_hidden_dims, sizes, kernel_sizes, dimension, basic_layers)

        # The input shape and last flatten size should use non-unet hidden dimension
        # Here cover the value from the initialization of the super class
        self.inpt_shape = tuple([-1] + [hidden_dims[0]] + last_size)
        self.last_flat_size = abs(reduce(lambda x, y: x*y, self.inpt_shape))

        _convs = []

        for uh, h, s, k in zip(unet_hidden_dims[:-1], hidden_dims[1:], sizes, self.kernel_sizes[:-1]):
            layers = []
            layers.append(self.basic_layers['deconv'](in_channels=uh, out_channels=h, kernel_size=k, size=s))
            if self.basic_layers['bn'] != nn.Identity:  layers.append(self.basic_layers['bn'](h))
            layers.append(self.basic_layers['actv']())
            _convs.append(nn.Sequential(*layers))
            
        self.convs = nn.ModuleList(_convs)
        self.last_conv = self.basic_layers['conv'](unet_hidden_dims[-1], out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        self.is_unet = True

    def forward(self, inpt: torch.Tensor, encoder_feature_map: List[nn.Module]):
        inpt = inpt.view(self.inpt_shape)
        idx = len(encoder_feature_map) - 1

        for conv in self.convs:
            # print(inpt.size(), encoder_feature_map[idx].size())
            inpt = torch.cat((inpt, encoder_feature_map[idx]), dim=1)
            idx -= 1
            inpt = conv(inpt)

        # print(inpt.size(), encoder_feature_map[idx].size())
        inpt = torch.cat((inpt, encoder_feature_map[idx]), dim=1)
        return self.last_conv(inpt)

class Resnet18Decoder(Decoder):

    def __init__(self,
                 out_channels: int,
                 last_size: List[int],
                 hidden_dims: List, #  = [16, 16, 32, 64, 128, 256], # The order is reversed!
                 num_blocks: List = 2, # = [2, 2, 2, 2, 2],
                 scales: List = 2, # = [2, 2, 2, 2, 2, 2],
                 output_size: List[int] = None, # if is not None, addition layer to interpolate
                 dimension: int = 2,
                 batchnorm: bool = True,
                 basic_layers: Dict = {}):
        
        super().__init__(out_channels)
        
        self.inpt_shape = tuple([-1] + [hidden_dims[0]] + last_size)
        self.last_flat_size = abs(reduce(lambda x, y: x*y, self.inpt_shape))

        self.basic_layers = _update_basic_layer(basic_layers, dimension=dimension, batchnorm=batchnorm)
        self.isbias = not batchnorm

        num_blocks = _extend_for_multilayer(num_blocks, len(hidden_dims[1:]))
        scales     = _extend_for_multilayer(scales,     len(hidden_dims[1:]))

        h0 = hidden_dims[0]

        block_layers = []
        for h, nb, s in zip(hidden_dims[1:], num_blocks, scales):
            block_layers.append(self._make_layer(in_channels=h0, out_channels=h, num_blocks=nb, scale=s))
            h0 = h
        
        self.blocks = nn.Sequential(*block_layers)

        if output_size is not None:
            self.conv1 = nn.Sequential(
                self.basic_layers['deconv'](h0, h0, kernel_size=3, size=output_size),
                self.basic_layers['bn'](h0),
                self.basic_layers['actv']())
        else:
            self.conv1 = nn.Identity()

        # Since the output not strictly inside [0,1], the activation layer 
        # after the last conv (self.fl) is removed
        self.fl = nn.Sequential(
                    self.basic_layers['conv'](h0, out_channels=out_channels, kernel_size=3, padding=1),
                    self.basic_layers['last_actv']())

    def _make_layer(self, in_channels, out_channels, num_blocks, scale):
        scales = [scale] + [1] * (num_blocks - 1)
        layers = []

        for st in scales:
            layers.append(BasicDecodeBlock(in_channels, out_channels, st, not self.isbias, self.basic_layers))
            # self.layer_in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, inpt: Tensor):
        inpt = inpt.view(self.inpt_shape)
        inpt = self.blocks(inpt)
        # print('Decoder last layer input:', inpt.size())

        result = self.conv1(inpt)
        result = self.fl(result)
        
        return result

class BasicDecodeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale, batchnorm, basic_layers):

        super().__init__()

        self.preactive = basic_layers['preactive']
        self.actv      = basic_layers['actv']
        self.isbias    = not batchnorm

        # out_channels = int(in_channels / scale)

        if scale == 1:

            if self.preactive:
                self.main = nn.Sequential(
                    nn.BatchNorm2d(out_channels),
                    self.actv(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    self.actv(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
                )
            else:
                self.main = nn.Sequential(
                    basic_layers['conv'](out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=self.isbias),
                    basic_layers['bn'](out_channels),
                    self.actv(),
                    basic_layers['conv'](out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=self.isbias),
                    basic_layers['bn'](out_channels)
                )
            
            self.shortcut = nn.Sequential()

        else:
            if self.preactive:
                self.main = nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    self.actv(),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    self.actv(),
                    IntpConv2d(in_channels, out_channels, kernel_size=3, scale_factor=scale)
                )
                self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    self.actv(),
                    IntpConv2d(in_channels, out_channels, kernel_size=3, scale_factor=scale)
                )
            
            else:
                self.main = nn.Sequential(
                    basic_layers['conv'](in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=self.isbias),
                    basic_layers['bn'](in_channels),
                    self.actv(),
                    basic_layers['deconv'](in_channels, out_channels, kernel_size=3, scale_factor=scale, bias=self.isbias),
                    basic_layers['bn'](out_channels)
                )
                self.shortcut = nn.Sequential(
                    basic_layers['deconv'](in_channels, out_channels, kernel_size=3, scale_factor=scale, bias=self.isbias),
                    basic_layers['bn'](out_channels)
                )
    
    def forward(self, inpt):

        result = self.main(inpt) + self.shortcut(inpt)

        if not self.preactive:
            result = self.actv()(result)

        return result

def mlp(in_features: int, out_features: int, hidden_dims: List[int], basic_layers: dict = {}, drop_out: float = 0.) -> nn.Module:
    '''
    return a multi layer percetron model

    ```text
    in_features -> hidden_dims[0] -> ... -> hidden_dims[-1] -> out_features
    ```
    
    - `basic_layers`
    - `drop_out`: (float) if > 0, add a dropout layer after each linear layer with rate = `drop_out`

    '''

    return _decoder_input(hidden_dims, in_features, out_features, basic_layers, drop_out)

def _decoder_input(typ: float, ld: int, lfd: int, basic_layers: dict = {}, drop_out: float = 0.) -> nn.Module:

    basic_layers = _update_basic_layer(basic_layers)

    if isinstance(typ, int):
        if typ == 0:
            return nn.Identity()
            
        elif typ == 1:
            return nn.Linear(ld, lfd)
            
        elif typ == 1.5:
            return nn.Sequential(nn.Linear(ld, lfd), nn.BatchNorm1d(lfd), nn.LeakyReLU())
        
        elif typ == 2:
            return nn.Sequential(
                nn.Linear(ld, ld*2), nn.BatchNorm1d(ld*2), nn.LeakyReLU(),
                nn.Linear(ld*2, lfd), nn.BatchNorm1d(lfd), nn.LeakyReLU())

        elif typ == 2.5:
            return nn.Sequential(
                nn.Linear(ld, ld), nn.BatchNorm1d(ld), nn.LeakyReLU(),
                nn.Linear(ld, lfd), nn.BatchNorm1d(lfd), nn.LeakyReLU())

        elif typ == 3:                      
            return nn.Sequential(
                nn.Linear(ld, ld), nn.BatchNorm1d(ld), nn.LeakyReLU(),
                nn.Linear(ld, ld*2), nn.BatchNorm1d(ld*2), nn.LeakyReLU(),
                nn.Linear(ld*2, lfd), nn.BatchNorm1d(lfd), nn.LeakyReLU())

        else:
            raise KeyError()
    
    elif isinstance(typ, list):
        layers = []
        h0 = ld
        for h in typ + [lfd]:
            layers.append(nn.Linear(h0, h))
            if basic_layers['bn'] != nn.Identity:   layers.append(basic_layers['bn'](h))
            # layers.append(nn.BatchNorm1d(h))
            if drop_out > 0.:   layers.append(nn.Dropout(p=drop_out))
            layers.append(basic_layers['actv']())
            h0 = h

        return nn.Sequential(*layers)

    elif isinstance(typ, nn.Module):
        return typ

    else:
        raise KeyError('not a valid type for decoder input, choose from `float`, `list[int]`, `nn.Module`')

'''
Citation: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

'''

class BasicLSTMCell(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gate = nn.Identity()
        self.drop_out = nn.Identity()

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([self.drop_out(input_tensor), h_cur], dim=1)  # concatenate along channel axis

        combined_gate = self.gate(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_gate, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return [h_next, c_next]

    def init_hidden(self, batch_size, image_size):
        hidden_sizes = tuple([batch_size, self.hidden_dim] + image_size)
        return (Variable(torch.zeros(hidden_sizes, device=self.gate.weight.device)),
                Variable(torch.zeros(hidden_sizes, device=self.gate.weight.device)))

class LSTMCell(BasicLSTMCell):

    def __init__(self, input_dim: int, hidden_dim: int, drop_out: float = 0.2, bias: bool = True):

        super().__init__(input_dim, hidden_dim)

        self.gate = nn.Linear(in_features=self.input_dim + self.hidden_dim,
                              out_features=4 * self.hidden_dim,
                              bias=bias)
        if drop_out > 0.:
            self.drop_out = nn.Dropout(p=drop_out)

class ConvLSTMCell(BasicLSTMCell):

    def __init__(self, input_dim: int,
                       hidden_dim: int,
                       kernel_size: int = 3, 
                       stride: int = 1, 
                       padding: int = 1, 
                       bias: bool = True):
        """
        Initialize ConvLSTM cell. (for 1D conv)

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super().__init__(input_dim, hidden_dim)

        self.gate = nn.Conv1d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)

class BiConvLSTMCell(ConvLSTMCell):

    '''
    
    
    '''
    def __init__(self, input_dim: int, hidden_dim: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1, 
                 kernel_size_cat: int = 1, stride_cat: int = 1, padding_cat: int = 1,
                 bias=True):
        super().__init__(input_dim, hidden_dim, kernel_size, stride, padding, bias)

        self.concat = nn.Conv1d(in_channels=2 * hidden_dim,
                                out_channels=hidden_dim,
                                kernel_size=kernel_size_cat,
                                stride=stride_cat,
                                padding=padding_cat,
                                bias=bias)

class BasicGRUCell(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv_gates = nn.Identity()  # for update_gate,reset_gate respectively
        self.conv_can  = nn.Identity()  # for candidate neural memory 
        self.drop_out = nn.Identity()

    def init_hidden(self, batch_size, image_size):
        hidden_sizes = tuple([batch_size, self.hidden_dim] + image_size)
        return [Variable(torch.zeros(hidden_sizes, device=self.conv_gates.weight.device))]
    
    def forward(self, input_tensor, cur_state):
        """

        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        h_cur = cur_state[0]
        input_tensor = self.drop_out(input_tensor)
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return [h_next]

class GRUCell(BasicGRUCell):

    def __init__(self, input_dim: int, hidden_dim: int, drop_out: float = 0.2, bias: bool = True) -> None:
        super().__init__(input_dim, hidden_dim)

        self.conv_gates = nn.Linear(in_features=input_dim+hidden_dim, out_features=2*self.hidden_dim, bias=bias)
        self.conv_can   = nn.Linear(in_features=input_dim+hidden_dim, out_features=self.hidden_dim, bias=bias)

        if drop_out > 0.:
            self.drop_out = nn.Dropout(p=drop_out)

class BiGRUCell(GRUCell):

    def __init__(self, input_dim: int, hidden_dim: int, drop_out: float = 0.2, bias: bool = True) -> None:
        super().__init__(input_dim, hidden_dim, drop_out, bias)

        self.conv_concat = nn.Sequential(
            nn.Linear(in_features=2*hidden_dim, out_features=hidden_dim, bias=bias),
            nn.Tanh())

class ConvGRUCell(BasicGRUCell):

    def __init__(self, input_dim: int,
                       hidden_dim: int,
                       kernel_size: int = 3, 
                       stride: int = 1, 
                       padding: int = 1, 
                       bias: bool = True) -> None:
        super().__init__(input_dim, hidden_dim)

        self.conv_gates = nn.Conv1d(in_channels=input_dim + hidden_dim,
                                    out_channels=2 * self.hidden_dim,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    bias=bias)

        self.conv_can  = nn.Conv1d(in_channels=input_dim + hidden_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)
        
class BiConvGRUCell(ConvGRUCell):

    '''
    
    
    '''
    def __init__(self, input_dim: int, hidden_dim: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1, 
                 kernel_size_cat: int = 1, stride_cat: int = 1, padding_cat: int = 0,
                 bias=True):
        super().__init__(input_dim, hidden_dim, kernel_size, stride, padding, bias)

        self.conv_concat = nn.Sequential(
            nn.Conv1d(in_channels=2 * hidden_dim,
                                out_channels=hidden_dim,
                                kernel_size=kernel_size_cat,
                                stride=stride_cat,
                                padding=padding_cat,
                                bias=bias),
            nn.Tanh())

class LSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dims, cell_type: str, bi_direction: bool = True, kernel_sizes: List = None,
                 image_size = [], batch_first=False, bias=True, return_all_layers=False):
        super().__init__()

        # _check_kernel_size_consistency(kernel_sizes)
        self.num_layers = len(hidden_dims)
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        # kernel_size = _extend_for_multilayer(kernel_sizes, self.num_layers)
        # hidden_dim = _extend_for_multilayer(hidden_dims, self.num_layers)
        # if not len(kernel_sizes) == len(hidden_dims) == self.num_layers:
        #     raise ValueError('Inconsistent list length.')

        # self.input_dim = input_dim
        # self.hidden_dim = hidden_dim
        # self.kernel_size = kernel_size
        # self.num_layers = num_layers
        # self.batch_first = batch_first
        # self.bias = bias
        self.return_all_layers = return_all_layers
        if bi_direction:
            # if cell_type in ['LSTM']:       cell_class = BiLSTMCell
            if cell_type in ['GRU']:        cell_class = BiGRUCell
            if cell_type in ['ConvLSTM']:   cell_class = BiConvLSTMCell
            if cell_type in ['ConvGRU']:    cell_class = BiConvGRUCell
            self.iter_method = self.iter_inner_bi
        else:
            if cell_type in ['LSTM']:       cell_class = LSTMCell
            if cell_type in ['GRU']:        cell_class = GRUCell
            if cell_type in ['ConvLSTM']:   cell_class = ConvLSTMCell
            if cell_type in ['ConvGRU']:    cell_class = ConvGRUCell
            self.iter_method = self.iter_inner

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            if cell_type in ['ConvGRU', 'ConvLSTM']:
                cell_list.append(cell_class(input_dim=cur_input_dim,
                                            hidden_dim=hidden_dims[i],
                                            kernel_size=kernel_sizes[i],
                                            bias=bias))
            else:
                cell_list.append(cell_class(input_dim=cur_input_dim,
                                            hidden_dim=hidden_dims[i],
                                            bias=bias))                

        self.cell_list = nn.ModuleList(cell_list)
        self.image_size = image_size

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        # if not self.batch_first:
        #     # (t, b, c, h, w) -> (b, t, c, h, w)
        #     input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b = input_tensor.size(0)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=self.image_size)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            layer_output, out_hidden_state = self.iter_method(cur_layer_input, layer_idx, seq_len, hidden_state[layer_idx])
            layer_output_list.append(layer_output)
            last_state_list.append(out_hidden_state)
            cur_layer_input = layer_output

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]

        return layer_output_list, last_state_list

    def iter_inner(self, cur_layer_input, layer_idx, seq_len, hidden_state):

        output_inner = []
        for t in range(seq_len):
            hidden_state = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :],
                                                cur_state=hidden_state)
            output_inner.append(hidden_state[0])

        layer_output = torch.stack(output_inner, dim=1)
        return layer_output, hidden_state

    def iter_inner_bi(self, cur_layer_input, layer_idx, seq_len, hidden_state):
        
        output_inner = []
        backward_states = []
        forward_states = []

        hidden_state_back = hidden_state
        hidden_state_forward = hidden_state
        for t in range(seq_len):
            hidden_state_back = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, seq_len - t - 1],
                                                cur_state=hidden_state_back)
            backward_states.append(hidden_state_back[0])

        for t in range(seq_len):
            hidden_state_forward = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t],
                                                cur_state=hidden_state_forward)
            forward_states.append(hidden_state_forward[0])
        
        for t in range(seq_len):
            h = self.cell_list[layer_idx].conv_concat(torch.cat((forward_states[t], backward_states[seq_len - t - 1]), dim=1))
            output_inner.append(h)

        layer_output = torch.stack(output_inner, dim=1)
        return layer_output, [backward_states, forward_states]

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

