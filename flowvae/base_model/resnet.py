
import torch
from torch import nn
from torch.nn import functional as F
from functools import reduce
from torch.autograd import Variable

from .utils import Encoder, Decoder, _extend_for_multilayer, _update_basic_layer, _make_aux_layers

from typing import Tuple, List, Dict, NewType, Callable
Tensor = NewType('Tensor', torch.tensor)

class Resnet18Encoder(Encoder):

    def __init__(self,
                 in_channels: int,
                 last_size: List[int],
                 hidden_dims: List, #  = [16, 32, 64, 128, 256]
                 num_blocks: List = 2,
                 strides: List = 2,
                 extra_first_conv: Tuple[int] = None,
                 force_last_size: bool = False,
                 dimension: int = 2,
                 basic_layers: Dict = {}) -> None:
        
        super().__init__(in_channels, last_size, hidden_dims)

        self.basic_layers = _update_basic_layer(basic_layers, dimension=dimension)
        self.isbias = not self.basic_layers['batchnorm']

        num_blocks = _extend_for_multilayer(num_blocks,  len(hidden_dims))
        strides    = _extend_for_multilayer(strides,     len(hidden_dims))
        
        h0 = self.in_channels
        if extra_first_conv is not None:
            h, k, s, p = extra_first_conv
            self.conv1 = nn.Sequential(
                self.basic_layers['conv'](in_channels=h0, out_channels=h, kernel_size=k, stride=s, padding=p, bias=self.isbias),
                *_make_aux_layers(self.basic_layers, h),
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
            layers.append(BasicEncodeBlock(in_channels, out_channels, st, i_block, basic_layers=self.basic_layers))

        return nn.Sequential(*layers)

    def forward(self, inpt):
        result = self.conv1(inpt)
        result = self.blocks(result)

        # print('Encoder last layer input:', result.size())

        result = self.adap(result)

        # print('Encoder last layer input:', result.size())

        return result
     
class BasicEncodeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, i_block, basic_layers):

        super().__init__()

        self.preactive = basic_layers['preactive']
        self.actv      = basic_layers['actv']
        self.isbias    = not basic_layers['batchnorm']

        # out_channels = in_channels * stride
        if i_block > 0:

            if self.preactive:
                self.main = nn.Sequential(
                    *_make_aux_layers(basic_layers, in_channels),
                    self.actv(),
                    basic_layers['conv'](out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=self.isbias),
                    *_make_aux_layers(basic_layers, out_channels),
                    self.actv(),
                    basic_layers['conv'](out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=self.isbias)
                )       
            else:
                self.main = nn.Sequential(
                    basic_layers['conv'](out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=self.isbias),
                    *_make_aux_layers(basic_layers, out_channels),
                    self.actv(),
                    basic_layers['conv'](out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=self.isbias),
                    *_make_aux_layers(basic_layers, out_channels)
                )
            self.shortcut = nn.Sequential()

        else:
            if self.preactive:
                self.main = nn.Sequential(
                    *_make_aux_layers(basic_layers, in_channels),
                    self.actv(),
                    basic_layers['conv'](in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=self.isbias),
                    *_make_aux_layers(basic_layers, out_channels),
                    self.actv(),
                    basic_layers['conv'](out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=self.isbias)
                )     
                self.shortcut = nn.Sequential(
                    *_make_aux_layers(basic_layers, in_channels),
                    self.actv(),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=self.isbias)
                )
            else:
                self.main = nn.Sequential(
                    basic_layers['conv'](in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=self.isbias),
                    *_make_aux_layers(basic_layers, out_channels),
                    self.actv(),
                    basic_layers['conv'](out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=self.isbias),
                    *_make_aux_layers(basic_layers, out_channels)
                )
                self.shortcut = nn.Sequential(
                    basic_layers['conv'](in_channels, out_channels, kernel_size=1, stride=stride, bias=self.isbias),
                    *_make_aux_layers(basic_layers, out_channels)
                )
    
    def forward(self, inpt):
        # print(inpt.size())
        result = self.main(inpt) + self.shortcut(inpt)

        if not self.preactive:
            result = self.actv()(result)

        return result
    
class Resnet18Decoder(Decoder):

    def __init__(self,
                 out_channels: int,
                 last_size: List[int],
                 hidden_dims: List, #  = [16, 16, 32, 64, 128, 256], # The order is reversed!
                 num_blocks: List = 2, # = [2, 2, 2, 2, 2],
                 scales: List = 2, # = [2, 2, 2, 2, 2, 2],
                 output_size: List[int] = None, # if is not None, addition layer to interpolate
                 dimension: int = 2,
                 basic_layers: Dict = {}):
        
        super().__init__(out_channels)
        
        self.inpt_shape = tuple([-1] + [hidden_dims[0]] + last_size)
        self.last_flat_size = abs(reduce(lambda x, y: x*y, self.inpt_shape))

        self.basic_layers = _update_basic_layer(basic_layers, dimension=dimension)
        self.isbias = not self.basic_layers['batchnorm']

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
                *_make_aux_layers(self.basic_layers, h0),
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
            layers.append(BasicDecodeBlock(in_channels, out_channels, st, self.basic_layers))
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

    def __init__(self, in_channels, out_channels, scale, basic_layers):

        super().__init__()

        self.preactive = basic_layers['preactive']
        self.actv      = basic_layers['actv']
        self.isbias    = not basic_layers['batchnorm']

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
                    *_make_aux_layers(basic_layers, out_channels),
                    self.actv(),
                    basic_layers['conv'](out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=self.isbias),
                    *_make_aux_layers(basic_layers, out_channels)
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
                    basic_layers['deconv'](in_channels, out_channels, kernel_size=3, scale_factor=scale)
                )
                self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    self.actv(),
                    basic_layers['deconv'](in_channels, out_channels, kernel_size=3, scale_factor=scale)
                )
            
            else:
                self.main = nn.Sequential(
                    basic_layers['conv'](in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=self.isbias),
                    *_make_aux_layers(basic_layers, in_channels),
                    self.actv(),
                    basic_layers['deconv'](in_channels, out_channels, kernel_size=3, scale_factor=scale, bias=self.isbias),
                    *_make_aux_layers(basic_layers, out_channels)
                )
                self.shortcut = nn.Sequential(
                    basic_layers['deconv'](in_channels, out_channels, kernel_size=3, scale_factor=scale, bias=self.isbias),
                    *_make_aux_layers(basic_layers, out_channels)
                )
    
    def forward(self, inpt):

        result = self.main(inpt) + self.shortcut(inpt)

        if not self.preactive:
            result = self.actv()(result)

        return result