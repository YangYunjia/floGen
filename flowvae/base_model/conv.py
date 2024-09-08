

import torch
from torch import nn
from torch.nn import functional as F
from functools import reduce
from torch.autograd import Variable

from .utils import Encoder, Decoder, _extend_for_multilayer, _update_basic_layer, _make_aux_layers, Convbottleneck

from typing import Tuple, List, Dict, NewType, Callable
Tensor = NewType('Tensor', torch.Tensor)


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
            layers += _make_aux_layers(self.basic_layers, h)
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
        return inpt

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
        return inpt
    
class convDecoder(Decoder):

    def __init__(self, out_channels: int, 
                 last_size: List[int],   # the H x W of first layer viewed from last_flat_size
                 hidden_dims: List[int], # Tuple = (128, 64, 32, 16)
                 sizes: List[int],       # Tuple = (26, 101, 401)
                 kernel_sizes: List[int] = 3,
                 dimension: int = 1, 
                 last_conv: str = 'normal',
                 basic_layers: Dict = {}
                 ) -> None:
        
        super().__init__(out_channels)
        self.inpt_shape = tuple([-1] + [hidden_dims[0]] + last_size)
        self.last_flat_size = abs(reduce(lambda x, y: x*y, self.inpt_shape))

        self.kernel_sizes = _extend_for_multilayer(kernel_sizes,  len(hidden_dims))

        self.basic_layers = _update_basic_layer(basic_layers, dimension)

        layers = []

        h0 = hidden_dims[0]

        for h, s, k in zip(hidden_dims[1:], sizes, self.kernel_sizes[:-1]):
            layers.append(self.basic_layers['deconv'](in_channels=h0, out_channels=h, kernel_size=k, size=s))
            layers += _make_aux_layers(self.basic_layers, h)
            layers.append(self.basic_layers['actv']())
            h0 = h
        self.convs = nn.Sequential(*layers)

        if last_conv == 'normal':
            self.last_conv = self.basic_layers['conv'](hidden_dims[-1], out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        elif last_conv == 'bottleneck':
            self.last_conv = Convbottleneck(hidden_dims[-1], out_channels=out_channels, kernel_size=3, stride=1, padding=1, basic_layers=self.basic_layers)


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
                 kernel_sizes: List[int] = 3,
                 dimension: int = 1,
                 last_conv: str = 'normal',
                 basic_layers: Dict = {}) -> None:
        
        unet_hidden_dims = [hidden_dims[i] + encoder_hidden_dims[i] for i in range(len(hidden_dims))]
        super().__init__(out_channels, last_size, unet_hidden_dims, sizes, kernel_sizes, dimension, last_conv, basic_layers)

        # The input shape and last flatten size should use non-unet hidden dimension
        # Here cover the value from the initialization of the super class
        self.inpt_shape = tuple([-1] + [hidden_dims[0]] + last_size)
        self.last_flat_size = abs(reduce(lambda x, y: x*y, self.inpt_shape))

        _convs = []

        for uh, h, s, k in zip(unet_hidden_dims[:-1], hidden_dims[1:], sizes, self.kernel_sizes[:-1]):
            layers = []
            layers.append(self.basic_layers['deconv'](in_channels=uh, out_channels=h, kernel_size=k, size=s))
            layers += _make_aux_layers(self.basic_layers, h)
            layers.append(self.basic_layers['actv']())
            _convs.append(nn.Sequential(*layers))
            
        self.convs = nn.ModuleList(_convs)
        # self.last_conv = self.basic_layers['conv'](unet_hidden_dims[-1], out_channels=out_channels, kernel_size=3, stride=1, padding=1)

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