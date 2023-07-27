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

from typing import Tuple, List, Dict, NewType, Callable
Tensor = NewType('Tensor', torch.tensor)

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
        self.last_flat_size = hidden_dims[-1] * last_size[0]
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
                 kernel_sizes: List[int] = None,
                 strides: List[int] = None,
                 paddings: List[int] = None,
                 pool_kernels: List[int] = None,
                 pool_strides: List[int] = None,
                 dimension: int = 1,
                 basic_layers: Dict[nn.Module] = {}
                 ) -> None:
        
        super().__init__(in_channels, last_size, hidden_dims)

        _convs = []

        h0 = in_channels

        if kernel_sizes is None: kernel_sizes = [3 for _ in hidden_dims]
        if strides is None:      strides =      [2 for _ in hidden_dims]
        if paddings is None:     paddings =     [1 for _ in hidden_dims]
        if pool_kernels is None:     pool_kernels =     [3 for _ in hidden_dims]
        if pool_strides is None:     pool_strides =     [2 for _ in hidden_dims]

        if dimension == 1:
            if 'conv' not in basic_layers.keys():   basic_layers['conv'] = nn.Conv1d
            if 'actv' not in basic_layers.keys():   basic_layers['actv'] = nn.LeakyReLU
            if 'pool' not in basic_layers.keys():   basic_layers['pool'] = nn.AvgPool1d
            if 'bn'   not in basic_layers.keys():   basic_layers['bn']   = None
        elif dimension == 2:
            if 'conv' not in basic_layers.keys():   basic_layers['conv'] = nn.Conv2d
            if 'actv' not in basic_layers.keys():   basic_layers['actv'] = nn.LeakyReLU
            if 'pool' not in basic_layers.keys():   basic_layers['pool'] = nn.AvgPool2d
            if 'bn'   not in basic_layers.keys():   basic_layers['bn']   = None

        for h, k, s, p, kp, sp in zip(hidden_dims, kernel_sizes, strides, paddings, pool_kernels, pool_strides):
            layers = []
            layers.append(basic_layers['conv'](in_channels=h0, out_channels=h, kernel_size=k, stride=s, padding=p, bias=True))
            if basic_layers['bn'] is not None:  layers.append(basic_layers['bn'](h))
            layers.append(basic_layers['actv']())
            if kp > 0: layers.append(basic_layers['pool'](kernel_size=kp, stride=sp))
            h0 = h
            _convs.append(nn.Sequential(*layers))
        
        self.convs = nn.ModuleList(_convs)

    def forward(self, inpt):
        for conv in self.convs:
            inpt = conv(inpt)
            # print(inpt.size(), len(self.convs))
        # raise
        return super().forward(inpt)

class convEncoder_Unet(convEncoder):

    def __init__(self, in_channels, last_size,
                 hidden_dims: List = [32, 64, 128],
                 kernel_sizes: List = None,
                 strides: List = None,
                 paddings: List = None,
                 pool_kernels: List = None,
                 pool_strides: List = None
                 ) -> None:
        
        super().__init__(in_channels, last_size, hidden_dims, kernel_sizes, strides, paddings, pool_kernels, pool_strides)
        
        # for U-net
        self.is_unet = True
        self.feature_maps = []

    def forward(self, inpt):
        self.feature_maps = []
        if self.is_unet: self.feature_maps.append(inpt)
        for conv in self.convs:
            inpt = conv(inpt)
            # print(inpt.size(), len(self.convs))
            self.feature_maps.append(inpt)
        # raise
        return super().forward(inpt)

'''
class conv2dEncoder(Encoder):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List,
                 kernel_sizes: List,
                 strides: List,
                 paddings: List,
                 max_pools: List,
                 **kwargs) -> None:
        
        super().__init__(in_channels)
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
    
    def forward(self, inpt):

        result = self.encoder(inpt)

        return super().forward(result)
'''
           
class Resnet18Encoder(Encoder):

    def __init__(self,
                 in_channels: int,
                 last_size: List[int, int],
                 hidden_dims: List, #  = [16, 32, 64, 128, 256]
                 num_blocks: List = None,
                 strides: List = None,
                 preactive: bool = False, 
                 extra_first_conv: Tuple[int, int, int, int] = None,
                 force_last_size: bool = False) -> None:
        
        super().__init__(in_channels, last_size, hidden_dims)

        self.preactive = preactive

        if num_blocks is None: num_blocks = [2 for _ in hidden_dims]
        if strides is None:    strides    = [2 for _ in hidden_dims]
        
        h0 = self.in_channels

        if extra_first_conv is not None:
            h, k, s, p = extra_first_conv
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=h0, out_channels=h, kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(h),
                nn.LeakyReLU()) 
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

        for st in strides:
            layers.append(BasicEncodeBlock(in_channels, out_channels, st, self.preactive))

        return nn.Sequential(*layers)

    def forward(self, inpt):
        result = self.conv1(inpt)
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
    
    def forward(self, inpt):
        result = self.main(inpt) + self.shortcut(inpt)

        if not self.preactive:
            result = torch.relu(result)

        return result

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
        self.padding = (kernel_size - 1) / 2

    def forward(self, inpt):
        result = F.interpolate(inpt, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        result = self.conv(result)
        return result

class IntpConv1d(IntpConv):

    def __init__(self, in_channels, out_channels, kernel_size, size=None, scale_factor=None, mode=None):
        super().__init__(in_channels, out_channels, kernel_size, size, scale_factor, mode)
        if mode is None:    self.mode = 'linear'
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=self.padding)

class IntpConv2d(IntpConv):

    def __init__(self, in_channels, out_channels, kernel_size, size=None, scale_factor=None, mode=None):
        super().__init__(in_channels, out_channels, kernel_size, size, scale_factor, mode)
        if mode is None:    self.mode = 'bilinear'
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=self.padding)

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

    def __init__(self, in_channels, out_sizes,
                 hidden_dims: List = [128, 512, 1024]) -> None:
        
        super().__init__()
        
        layers = []
        self.last_flat_size = hidden_dims[0]
        self.out_sizes = out_sizes

        h0 = in_channels

        for h in hidden_dims:
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
                 basic_layers: Dict[nn.Module] = {}
                 ) -> None:
        
        super().__init__(out_channels)
        self.inpt_shape = tuple([-1] + [hidden_dims[0]] + last_size)
        self.last_flat_size = abs(reduce(lambda x, y: x*y, self.inpt_shape))
        
        if kernel_sizes is None: kernel_sizes = [3 for _ in hidden_dims]

        if dimension == 1:
            if 'conv' not in basic_layers.keys():   basic_layers['conv'] = nn.Conv1d
            if 'actv' not in basic_layers.keys():   basic_layers['actv'] = nn.LeakyReLU
            if 'deconv' not in basic_layers.keys(): basic_layers['deconv'] = IntpConv1d
            #   Another options include `ConvTranspose1d`
            if 'bn'   not in basic_layers.keys():   basic_layers['bn']   = None
        elif dimension == 2:
            if 'conv' not in basic_layers.keys():   basic_layers['conv'] = nn.Conv2d
            if 'actv' not in basic_layers.keys():   basic_layers['actv'] = nn.LeakyReLU
            if 'deconv' not in basic_layers.keys(): basic_layers['deconv'] = IntpConv2d
            #   Another options include `ConvTranspose2d`
            if 'bn'   not in basic_layers.keys():   basic_layers['bn']   = None
        
        if 'bn'   not in basic_layers.keys():   basic_layers['bn']   = None
        
        _convs = []
        h0 = hidden_dims[0]

        for h, s, k in zip(hidden_dims[1:], sizes, kernel_sizes[:-1]):
            layers = []
            layers.append(basic_layers['deconv'](in_channels=h0, out_channels=h, kernel_size=k, size=s))
            if basic_layers['bn'] is not None:  layers.append(basic_layers['bn'](h))
            layers.append(basic_layers['actv']())
            h0 = h
            _convs.append(nn.Sequential(*layers))
            
        self.convs = nn.ModuleList(_convs)
        self.fl = basic_layers['conv'](hidden_dims[-1], out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, inpt: torch.Tensor):
        inpt = inpt.view(self.inpt_shape)
        for conv in self.convs:
            inpt = conv(inpt)
        return self.fl(inpt)

class convDecoder_Unet(convDecoder):

    def __init__(self, out_channels,
                 last_size: List[int], 
                 hidden_dims: List[int], 
                 sizes: List[int], 
                 encoder_hidden_dims: List[int]) -> None:
        
        unet_hidden_dims = [hidden_dims[i] + encoder_hidden_dims[i] for i in range(len(hidden_dims))]
        super().__init__(out_channels, last_size, unet_hidden_dims, sizes)

        # The input shape and last flatten size should use non-unet hidden dimension
        # Here cover the value from the initialization of the super class
        self.inpt_shape = tuple([-1] + [hidden_dims[0]] + last_size)
        self.last_flat_size = abs(reduce(lambda x, y: x*y, self.inpt_shape))

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
                 last_size: List[int, int],
                 hidden_dims: List, #  = [16, 16, 32, 64, 128, 256], # The order is reversed!
                 num_blocks: List = None, # = [2, 2, 2, 2, 2],
                 scales: List = None, # = [2, 2, 2, 2, 2, 2],
                 output_size: List[int, int] = None, # if is not None, addition layer to interpolate
                 basic_layers: Dict = {}):
        
        super().__init__(out_channels)
        
        self.inpt_shape = tuple([-1] + [hidden_dims[0]] + last_size)
        self.last_flat_size = abs(reduce(lambda x, y: x*y, self.inpt_shape))
        self.basic_layers = basic_layers

        if 'preactive' not in basic_layers.keys():   basic_layers['actv'] = False
        if 'actv' not in basic_layers.keys():        basic_layers['actv'] = nn.LeakyReLU
        if 'last_actv' not in basic_layers.keys():   basic_layers['last_actv'] = nn.Identity

        if num_blocks is None:  num_blocks  = [2 for _ in hidden_dims[1:]]
        if scales is None:     scales     = [2 for _ in hidden_dims[1:]]

        h0 = hidden_dims[0]

        block_layers = []
        for h, nb, s in zip(hidden_dims[1:], num_blocks, scales):
            block_layers.append(self._make_layer(in_channels=h0, out_channels=h, num_blocks=nb, scale=s))
            h0 = h
        
        self.blocks = nn.Sequential(*block_layers)

        if output_size is not None:
            self.conv1 = nn.Sequential(
                IntpConv2d(h0, h0, kernel_size=3, size=output_size),
                nn.BatchNorm2d(h0),
                nn.LeakyReLU())
        else:
            self.conv1 = nn.Identity()

        # Since the output not strictly inside [0,1], the activation layer 
        # after the last conv (self.fl) is removed
        self.fl = nn.Sequential(
                    nn.Conv2d(h0, out_channels=out_channels, kernel_size=3, padding=1),
                    basic_layers['last_actv']())

    def _make_layer(self, in_channels, out_channels, num_blocks, scale):
        scales = [scale] + [1] * (num_blocks - 1)
        layers = []

        for st in scales:
            layers.append(BasicDecodeBlock(in_channels, out_channels, st, self.basic_layers))
            self.layer_in_channels = out_channels
        
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

        # out_channels = int(in_channels / scale)

        if scale == 1:

            if self.preactive:
                self.main = nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    self.actv(),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, scale=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    self.actv(),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, scale=1, padding=1, bias=False)
                )
            else:
                self.main = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, scale=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    self.actv(),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, scale=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            
            self.shortcut = nn.Sequential()

        else:
            if self.preactive:
                self.main = nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    self.actv(),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, scale=1, padding=1, bias=False),
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
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, scale=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    self.actv(),
                    IntpConv2d(in_channels, out_channels, kernel_size=3, scale_factor=scale),
                    nn.BatchNorm2d(out_channels)
                )
                self.shortcut = nn.Sequential(
                    IntpConv2d(in_channels, out_channels, kernel_size=3, scale_factor=scale),
                    nn.BatchNorm2d(out_channels)
                )
    
    def forward(self, inpt):
        result = self.main(inpt) + self.shortcut(inpt)

        if not self.preactive:
            result = self.actv()(result)

        return result

'''
class conv2dDecoder(Decoder):

    def __init__(self,
				 out_channels: int,
                 hidden_dims: List,
                 kernel_sizes: List,
                 scales: List,
                 paddings: List,
                 max_pools: List,
                 **kwargs) -> None:

        super().__init__(out_channels=out_channels)
        
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

    def forward(self, inpt):
        inpt = inpt.view(tuple([-1] + self.inpt_shape)) # 编码器最后全连接层之前的最后的大小，第一个数字是batch的大小
        result = self.decoder(inpt)
        return super().forward(result)
'''

def _decoder_input(typ: float, ld: int, lfd: int) -> nn.Module:

    if typ == 0:
        return nn.Identity()
        
    elif typ == 1:
        return nn.Linear(ld, lfd)
    
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

