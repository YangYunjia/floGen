
import torch
from torch import nn
from torch.nn import functional as F
from functools import reduce
from torch.autograd import Variable

from .utils import Encoder, _extend_for_multilayer, _update_basic_layer, _make_aux_layers

from typing import Union, List, NewType
Tensor = NewType('Tensor', torch.Tensor)


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
    

def mlp(in_features: int, out_features: int, hidden_dims: List[int], last_actv: bool = True, basic_layers: dict = {}) -> nn.Module:
    '''
    return a multi layer percetron model

    ```text
    in_features -> hidden_dims[0] -> ... -> hidden_dims[-1] -> out_features
    ```
    
    - `basic_layers`
        - `dropout`:   (float, default = 0.) if > 0, add a dropout layer after each linear layer with rate = `dropout`
        - `batchnorm`: (bool, default = True)
        - `actv`:      (nn.Module, default = nn.LeakyReLU)

    - `last_actv`: (bool, default = True) whether to add the aux layers and activation layers after the last linear layer

    '''

    return _decoder_input(hidden_dims, in_features, out_features, last_actv, basic_layers)

def _decoder_input(typ: Union[float, List[int]], ld: int, lfd: int, 
                   last_actv: bool = True,
                   basic_layers: dict = {}) -> nn.Module:

    basic_layers = _update_basic_layer(basic_layers, dimension=0)

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
        for h in typ:
            layers.append(nn.Linear(h0, h))
            layers += _make_aux_layers(basic_layers, h)
            layers.append(basic_layers['actv']())
            h0 = h

        layers.append(nn.Linear(h0, lfd))

        if last_actv:    
            layers += _make_aux_layers(basic_layers, lfd)
            layers.append(basic_layers['actv']())


        return nn.Sequential(*layers)

    else:
        raise KeyError('not a valid type for decoder input, choose from `float`, `list[int]`')