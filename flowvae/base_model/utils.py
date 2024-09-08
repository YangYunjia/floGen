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

from typing import Tuple, List, Dict, TypeVar, Callable
Tensor = TypeVar('Tensor', torch.tensor)

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

    def __init__(self, h0, out_channels, kernel_size = 3, stride = 1, padding = 1, basic_layers = {}) -> None:
        super().__init__()

        layers = []
        layers.append(basic_layers['conv'](h0, out_channels=h0, kernel_size=kernel_size, stride=stride, padding=padding))
        layers += _make_aux_layers(basic_layers, h0)
        layers.append(basic_layers['actv']())
        layers.append(basic_layers['conv'](h0, out_channels=out_channels, kernel_size=1, stride=1, padding=0))
        layers += _make_aux_layers(basic_layers, out_channels)
        if basic_layers['last_actv'] != nn.Identity: layers.append(basic_layers['last_actv']())

        self.convs = nn.Sequential(*layers)

    def forward(self, input):
        return self.convs(input)

default_basic_layers_0d = {
    'actv':     nn.LeakyReLU,
    'batchnorm':  False,
    'dropout':    0.,
}

default_basic_layers_1d = {
    'conv':     nn.Conv1d,
    'actv':     nn.LeakyReLU,
    'deconv':   IntpConv1d,
    'pool':     nn.AvgPool1d,
    'batchnorm':  True,
    'dropout':    0.,
    'preactive':  False,
    'last_actv':  nn.Identity
}

default_basic_layers_2d = {
    'conv':     nn.Conv2d,
    'actv':     nn.LeakyReLU,
    'deconv':   IntpConv2d,
    'pool':     nn.AvgPool2d,
    'batchnorm':  True,
    'dropout':    0.,
    'preactive':  False,
    'last_actv':  nn.Identity
}

def _update_basic_layer(basic_layers: dict, dimension: int = 1):

    if dimension == 0:
        basic_layers_n = copy.deepcopy(default_basic_layers_0d)
        for key in basic_layers: basic_layers_n[key] = basic_layers[key]

        if basic_layers_n['batchnorm']:   basic_layers_n['_bn'] = nn.BatchNorm1d
        if basic_layers_n['dropout'] > 0.:   basic_layers_n['_drop'] = nn.Dropout(basic_layers_n['dropout'])

    elif dimension == 1:   
        basic_layers_n = copy.deepcopy(default_basic_layers_1d)
        for key in basic_layers: basic_layers_n[key] = basic_layers[key]

        if basic_layers_n['batchnorm']:   basic_layers_n['_bn'] = nn.BatchNorm1d
        if basic_layers_n['dropout'] > 0.:   basic_layers_n['_drop'] = nn.Dropout1d(basic_layers_n['dropout'])

    elif dimension == 2: 
        basic_layers_n = copy.deepcopy(default_basic_layers_2d)
        for key in basic_layers: basic_layers_n[key] = basic_layers[key]

        if basic_layers_n['batchnorm']:   basic_layers_n['_bn'] = nn.BatchNorm2d
        if basic_layers_n['dropout'] > 0.:   basic_layers_n['_drop'] = nn.Dropout2d(basic_layers_n['dropout'])

    return basic_layers_n

def _make_aux_layers(basic_layers, h):
    aux_layers = []
    if '_bn' in basic_layers.keys():  
        aux_layers.append(basic_layers['_bn'](h))
        # print('add bn')
    if '_drop' in basic_layers.keys():  
        aux_layers.append(basic_layers['_drop'])
        # print('add drop')

    return  aux_layers


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

        return inpt
    
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
