'''
This file is a libaray of models used in training a wing predictin model.

It is a combination of `run.py` (airfoil-to-wing models), `run_2d.py` (airfoil prediction models), and `run_cldis` (SLD estimation models)


'''

from torch import nn
import torch
from flowvae.vae import DecoderModel, EncoderDecoderLSTM, EncoderDecoder, CondAutoEncoder, Unet, BranchUnet, BranchDecoderModel
from flowvae.base_model.rnn import LSTM
from flowvae.base_model.utils import Decoder, Encoder
from flowvae.base_model.conv import convEncoder, convDecoder, convEncoder_Unet, convDecoder_Unet
from flowvae.base_model.mlp import mlp
from flowvae.base_model.resnet import Resnet18Decoder, Resnet18Encoder, ResnetDecoder_Unet, ResnetEncoder_Unet
from flowvae.base_model.trans import Transolver, UTransolver, EncoderDecoderTransolver, ViT, TokenSpaceTransolver
from flowvae.base_model.pdet.pde_transformer import PDEImpl

'''
This part is for airfoil-to-wing models

input dimensions
        0.   1.    2.           3               4             5             6                7
index: id, AoA, Mach, swept_angle, dihedral_angle, aspect_ratio, tapper_ratio, tip_twist_angle, tip2root_thickness_ratio, ref_area, root_thickness, 
    cstu, cstl, real_cl, real_cd, recon_cl, recon_cd

delta: xLE, yLE, zLE, alpha, chord, thick, cstu, cstl, Ma
       0    1    2    3      4      5      6-15  16-25 26
'''

def convLSTMmodel(h_e1, h_e2, h_lstm, h_d):

    encodercell = convDecoder(out_channels=h_e2[-1], last_size=[5], hidden_dims=h_e2[:-1], 
                            sizes=[100, 321], last_conv='bottleneck')
    
    encoder = DecoderModel(input_channels=6, decoder=encodercell, decoder_input_layer=h_e1)

    lstm = LSTM(input_dim=h_lstm[0], hidden_dims=h_lstm[1:], cell_type='ConvGRU', bi_direction=True, kernel_sizes=[3 for _ in h_lstm[1:]])

    decoder = nn.Conv1d(in_channels=h_d[0], out_channels=1, kernel_size=3, stride=1, padding=1)
    
    vae_model = EncoderDecoderLSTM(lstm=lstm, encoder=encoder, decoder=decoder, nt=61)

    return vae_model

def latentLSTMmodel(h_e, h_lstm, h_d1, h_d2, in_features=6):
        
    encoder = mlp(in_features=in_features, out_features=h_e[-1], hidden_dims=h_e[:-1])

    lstm = LSTM(input_dim=h_lstm[0], hidden_dims=h_lstm[1:], cell_type='GRU', bi_direction=True)

    decodercell = convDecoder(out_channels=1, last_size=[5], hidden_dims=h_d2, 
                            sizes=[24, 100, 321], last_conv='bottleneck')
    
    decoder = DecoderModel(input_channels=h_d1[0], decoder=decodercell, decoder_input_layer=h_d1[1:])
    
    vae_model = EncoderDecoderLSTM(lstm=lstm, encoder=encoder, decoder=decoder, nt=61)

    return vae_model

def latentoutLSTMmodel(h_e, h_lstm, h_d, in_features=6):
        
    encoder = mlp(in_features=in_features, out_features=h_e[-1], hidden_dims=h_e[:-1])

    lstm = LSTM(input_dim=h_lstm[0], hidden_dims=h_lstm[1:], cell_type='GRU', bi_direction=True)

    if h_d is not None:
        decoder = mlp(in_features=h_d[0], out_features=32, hidden_dims=h_d[1:])
    
    vae_model = EncoderDecoderLSTM(lstm=lstm, encoder=encoder, decoder=decoder, nt=61)

    return vae_model

def conv2dmodel(h_d1, h_d2):
      
    decodercell = convDecoder(out_channels=1, last_size=[3, 5], hidden_dims=h_d2, 
                                sizes=[[6, 24],  [21, 100], [61, 321]], dimension=2, last_conv='bottleneck')
        
    decoder = DecoderModel(input_channels=6, decoder=decodercell, decoder_input_layer=h_d1)

    return decoder

def resnet2dmodel(h_d1, h_d2, h_out=1, batchnorm=True, nt=61, last_size=[3, 5]):
      
    decodercell = Resnet18Decoder(out_channels=h_out, last_size=last_size, hidden_dims=h_d2, output_size=[nt, 321], batchnorm=batchnorm)
    decoder = DecoderModel(input_channels=29, decoder=decodercell, decoder_input_layer=h_d1)

    return decoder

def resnet2dsmodel(h_d1, h_d2, h_in=29, h_out=1, nt=61, last_size=1):
      
    decodercell = Resnet18Decoder(out_channels=h_out, last_size=[nt, last_size], hidden_dims=h_d2, scales=(1, 2), output_size=[nt, 321])
    decoder = DecoderModel(input_channels=h_in, decoder=decodercell, decoder_input_layer=h_d1)

    return decoder

class multiinput_resnet2dsmodel(nn.Module):

    def __init__(self, h_e1, h_e2, h_d1, h_d2, h_out=1, nt=61, last_size=8):
        super().__init__()
        
        self.foil_encoder = mlp(in_features=21, out_features=h_e1[-1], hidden_dims=h_e1[:-1])
        self.wing_encoder = mlp(in_features=8,  out_features=h_e2[-1], hidden_dims=h_e2[:-1])

        self.decoder = resnet2dsmodel(h_d1, h_d2, h_in=h_e1[-1]+h_e2[-1], h_out=h_out, nt=nt, last_size=last_size)
        
    def forward(self, inputs):
        foil_encoded = self.foil_encoder(inputs[:, 8:])
        wing_encoded = self.wing_encoder(inputs[:, :8])
        encoded = torch.concatenate((foil_encoded, wing_encoded), dim=1)
        return self.decoder(encoded)

class triinput_resnet2dsmodel(nn.Module):

    def __init__(self, h_e1, h_e2, h_e3, h_d1, h_d2, h_out=1, nt=61, last_size=8, dropout=0.):
        super().__init__()
        
        self.foil_encoder = mlp(in_features=21, out_features=h_e1[-1], hidden_dims=h_e1[:-1], basic_layers={'dropout': dropout})
        self.wing_encoder = mlp(in_features=6,  out_features=h_e2[-1], hidden_dims=h_e2[:-1], basic_layers={'dropout': dropout})
        self.cond_encoder = mlp(in_features=2,  out_features=h_e3[-1], hidden_dims=h_e3[:-1], basic_layers={'dropout': dropout})

        self.decoder = resnet2dsmodel(h_d1, h_d2, h_in=h_e1[-1]+h_e2[-1]+h_e3[-1], h_out=h_out, nt=nt, last_size=last_size)
        
    def forward(self, inputs):
        foil_encoded = self.foil_encoder(inputs[:, 8:])
        wing_encoded = self.wing_encoder(inputs[:, 2:8])
        cond_encoded = self.cond_encoder(inputs[:, :2])
        encoded = torch.concatenate((cond_encoded, foil_encoded, wing_encoded), dim=1)
        return self.decoder(encoded)

class onet_resnet2dsmodel(nn.Module):

    def __init__(self, h_e1, h_e2, h_d1, h_d2, h_out=1, nt=61, last_size=8):
        super().__init__()
        
        self.foil_encoder = mlp(in_features=21, out_features=h_e1[-1], hidden_dims=h_e1[:-1])
        self.wing_encoder = mlp(in_features=8,  out_features=h_e2[-1], hidden_dims=h_e2[:-1])

        if h_e1[-1] != h_e2[-1]: raise AttributeError('h_e1[-1] (%d) != h_e2[-1] (%d)' % (h_e1[-1], h_e2[-1]))

        self.decoder = resnet2dsmodel(h_d1, h_d2, h_in=h_e1[-1], h_out=h_out, nt=nt, last_size=last_size)
        
    def forward(self, inputs):
        foil_encoded = self.foil_encoder(inputs[:, 8:])
        wing_encoded = self.wing_encoder(inputs[:, :8])
        encoded = foil_encoded * wing_encoded
        return self.decoder(encoded)

def latentLSTMresnet(h_e, h_lstm, h_d1, h_d2, in_features=6, h_out=1, nt=61):
        
    encoder = mlp(in_features=in_features, out_features=h_e[-1], hidden_dims=h_e[:-1])

    lstm = LSTM(input_dim=h_lstm[0], hidden_dims=h_lstm[1:], cell_type='GRU', bi_direction=True)

    decodercell = Resnet18Decoder(out_channels=h_out, last_size=[5], hidden_dims=h_d2, output_size=[321], dimension=1)
    
    decoder = DecoderModel(input_channels=h_d1[0], decoder=decodercell, decoder_input_layer=h_d1[1:])
    
    vae_model = EncoderDecoderLSTM(lstm=lstm, encoder=encoder, decoder=decoder, nt=nt)

    return vae_model

def latentLSTMresnet2d(h_e, h_lstm, h_d, in_features=27, h_out=1, nt=61, last_size=1):
    
    if h_e is None:
        encoder = nn.Identity()
    else:
        encoder = mlp(in_features=in_features, out_features=h_e[-1], hidden_dims=h_e[:-1])

    lstm = LSTM(input_dim=h_lstm[0], hidden_dims=h_lstm[1:], cell_type='GRU', bi_direction=True)

    decodercell = Resnet18Decoder(out_channels=h_out, last_size=[nt, last_size], hidden_dims=[int(h_lstm[-1]/last_size)] + h_d, scales=(1, 2), output_size=[nt, 321])
    
    decoder = DecoderModel(input_channels=h_lstm[-1], decoder=decodercell, decoder_input_layer=0)
    
    vae_model = EncoderDecoderLSTM(lstm=lstm, encoder=encoder, decoder=decoder, nt=nt, decoder_input_mode='2D')

    return vae_model

def embedLSTMresnet2d(h_e, h_lstm, h_d, in_features=6, h_out=1, nt=61):
        
    encoder = mlp(in_features=in_features, out_features=h_e[-1], hidden_dims=h_e[:-1])

    lstm = LSTM(input_dim=h_lstm[0], hidden_dims=h_lstm[1:], cell_type='GRU', bi_direction=True)

    decodercell = Resnet18Decoder(out_channels=h_out, last_size=[nt, 1], hidden_dims=[h_lstm[-1]] + h_d, scales=(1, 2), output_size=[nt, 321], batchnorm=True)
    
    decoder = DecoderModel(input_channels=h_lstm[-1]+in_features, decoder=decodercell, decoder_input_layer=0)
    
    vae_model = EncoderDecoderLSTM(lstm=lstm, encoder=encoder, decoder=decoder, nt=nt, decoder_input_mode='2d')
    raise NotImplementedError()
    return vae_model

def resnetedmodel(h_e, h_d1, h_d2, h_out=3, batchnorm=True):

    encodercell = Resnet18Encoder(in_channels=2, last_size=[3, 5], hidden_dims=h_e, batchnorm=batchnorm, force_last_size=True)
    decodercell = Resnet18Decoder(out_channels=h_out, last_size=[3, 5], hidden_dims=h_d2, output_size=[61, 321], batchnorm=batchnorm)
    ae_model = EncoderDecoder(latent_dim=32, encoder=encodercell, decoder=decodercell, code_mode='ed', code_dim=6, decoder_input_layer=h_d1)

    return ae_model

def conv2dedmodel(h_e, h_d1, h_d2, h_out=3):

    encodercell = convEncoder(in_channels=2, last_size=[4, 4], hidden_dims=h_e, pool_kernels=[3, (1, 3), (1, 3)], pool_strides=[2, (1, 2), (1, 2)], dimension=2)
    decodercell = convDecoder(out_channels=h_out, last_size=[3, 5], hidden_dims=h_d2, sizes=[[6, 24],  [21, 100], [61, 321]], dimension=2, last_conv='bottleneck')
    ae_model = EncoderDecoder(latent_dim=32, encoder=encodercell, decoder=decodercell, code_mode='ed', code_dim=6, decoder_input_layer=h_d1)

    return ae_model

def resnetdeltadecode(h_d, h_out=1, batchnorm=True):
      
    decodercell = Resnet18Decoder(out_channels=h_out, last_size=[61, 5], hidden_dims=h_d, scales=(1, 2), output_size=[61, 321], batchnorm=batchnorm)
    decoder = DecoderModel(input_channels=1, decoder=decodercell, decoder_input_layer=0)

    return decoder

def resnetdeltachanneldecode(h_d, h_e=None, h_out=1, nt=61, in_channels=27, last_size=1, dropout=0.):
    
    if h_e is not None:
        encodercell = h_e 
        decodercell = Resnet18Decoder(out_channels=h_out, last_size=[nt, last_size], hidden_dims=[int(h_e.h_e_last/last_size)] + h_d, scales=(1, 2), output_size=[nt, 321], basic_layers={'dropout': dropout})
        decoder = DecoderModel(decoder=decodercell, decoder_input_layer=encodercell)
    else:
        decodercell = Resnet18Decoder(out_channels=h_out, last_size=[nt, 1], hidden_dims=[in_channels] + h_d, scales=(1, 2), output_size=[nt, 321], basic_layers={'dropout': dropout})
        decoder = DecoderModel(decoder=decodercell, decoder_input_layer=0)

    return decoder

class singleinput_encoder(nn.Module):

    def __init__(self, h_e, in_channels=27, kernel=1, nt=61, h_e_type='resnet'):
        if h_e_type == 'resnet':
            self.model = Resnet18Encoder(in_channels=in_channels, last_size=[nt], hidden_dims=h_e, strides=1, dimension=1)
        elif h_e_type == 'conv':
            self.model = convEncoder(in_channels=in_channels, last_size=[nt], hidden_dims=h_e, kernel_sizes=kernel, strides=1, paddings=int(kernel/2), pool_kernels=0, dimension=1)
        
        self.h_e_last = h_e[-1]

    def forward(self, inputs):
        return self.model(inputs)

class onet_mlpconv_encoder(nn.Module):

    def __init__(self, h_e1, h_e2, kernel=1, nt=61, dropout=0.):
        super().__init__()
        
        self.foil_encoder = mlp(in_features=21, out_features=h_e1[-1], hidden_dims=h_e1[:-1], basic_layers={'dropout': dropout})
        self.wing_encoder = convEncoder(in_channels=6, last_size=[nt], hidden_dims=h_e2, kernel_sizes=kernel, strides=1, paddings=int(kernel/2), pool_kernels=0, dimension=1, basic_layers={'dropout': dropout})

        self.nt = nt

        if h_e1[-1] != h_e2[-1]: raise AttributeError('h_e1[-1] (%d) != h_e2[-1] (%d)' % (h_e1[-1], h_e2[-1]))
        self.h_e_last = h_e1[-1]
        
    def forward(self, inputs):
        foil_encoded = self.foil_encoder(inputs[:, 6:, 0])
        wing_encoded = self.wing_encoder(inputs[:, :6])#.reshape(-1, self.h_e_last, self.nt)
        encoded = foil_encoded.unsqueeze(2).repeat(1, 1, self.nt) * wing_encoded
        return encoded

class cat_mlpconv_encoder(nn.Module):
    
    def __init__(self, h_e1, h_e2, h_e3=None, kernel=1, nt=61, dropout=0.):
        super().__init__()
        
        self.foil_encoder = mlp(in_features=21, out_features=h_e1[-1], hidden_dims=h_e1[:-1], basic_layers={'dropout': dropout})
        self.wing_encoder = convEncoder(in_channels=6, last_size=[nt], hidden_dims=h_e2, 
                                        kernel_sizes=kernel, strides=1, paddings=int(kernel/2), pool_kernels=0, dimension=1, basic_layers={'dropout': dropout})
        if h_e3 is not None:
            self.cat_encoder  = convEncoder(in_channels=h_e1[-1]+h_e2[-1], last_size=[nt], hidden_dims=h_e3, 
                                        kernel_sizes=kernel, strides=1, paddings=int(kernel/2), pool_kernels=0, dimension=1, basic_layers={'dropout': dropout})
            self.h_e_last = h_e3[-1]
        else:
            self.cat_encoder = nn.Identity()
            self.h_e_last = h_e1[-1] + h_e2[-1]
            
        self.nt = nt
        self.h_e2_last = h_e2[-1]
        
    def forward(self, inputs):
        foil_encoded = self.foil_encoder(inputs[:, 6:, 0]).unsqueeze(2).repeat(1, 1, self.nt)
        wing_encoded = self.wing_encoder(inputs[:, :6]).reshape(-1, self.h_e2_last, self.nt)
#         print(foil_encoded.size(), wing_encoded.size())
        encoded = torch.concatenate((foil_encoded, wing_encoded), dim=1)
        return self.cat_encoder(encoded)
    
###############
#    500
###############

class basiconetmodel(nn.Module):
    
    def set_basic(self, h_e1, h_e2, h_lstm, nt, coder_type, coder_kernel, last_size):

        if coder_type == 'onet':
            self.coder = onet_mlpconv_encoder(h_e1=h_e1, h_e2=h_e2, nt=nt, kernel=coder_kernel)
            self.last_size = [-1, nt, last_size, int(h_e2[-1] / last_size)]
        elif coder_type == 'onlywing':
            dropout = 0.
            self.coder = convEncoder(in_channels=6, last_size=[nt], hidden_dims=h_e2, kernel_sizes=coder_kernel, strides=1, paddings=int(coder_kernel/2), pool_kernels=0, dimension=1, basic_layers={'dropout': dropout})
            self.last_size = [-1, nt, last_size, int(h_e2[-1] / last_size)]
        elif coder_type == 'onlycond':
            self.coder = None
            self.last_size = [1, 1, nt, last_size]
            
        else:
            raise Exception()

        
        if h_lstm is not None:
            self.lstm = LSTM(input_dim=h_lstm[0], hidden_dims=h_lstm[1:], cell_type='GRU', bi_direction=True)
            self.decoder_input_size = [-1, nt, last_size, int(h_lstm[-1] / last_size)]
        else:
            self.lstm = None
            
    def forward(self, inputs, code):
#         print(inputs.size(), code.size())
        mu = self.encode(inputs)
        if self.coder_type == 'onlywing':
            mach = code[:, :, -1]
            code = code[:, :, :6]
        elif self.coder_type == 'onlycond':
            # c = code[:, 1:3].reshape(-1, 2, 1, 1).repeat(*self.last_size)
            c = code[:, :2].reshape(-1, 2, 1, 1).repeat(*self.last_size)    #! critical change on Aug. 30 2024; old training should modify to extract oc from dataset
        
        if self.coder_type in ['onlywing', 'onet']:
            c  = self.coder(code.transpose(1,2)).reshape(self.last_size).permute((0, 3, 1, 2))

        decoder_input = self.code_func(mu, c)
#         print(decoder_input.size(), mu.size(), c.size())

        if self.coder_type == 'onlywing':
            # product with mach number of wing
            if self.de_type == 'prod':
                decoder_input = decoder_input * mach.reshape(-1, 1, decoder_input.size(2), 1).repeat(1, decoder_input.size(1), 1, decoder_input.size(3))
            else:
                decoder_input = torch.cat((decoder_input, mach.reshape(-1, 1, decoder_input.size(2), 1).repeat(1, 1, 1, decoder_input.size(3))), dim=1)
#         print(decoder_input.size())
                
        if self.lstm is not None:
            decoder_input = torch.flatten(decoder_input.permute((0, 2, 1, 3)), start_dim=2)
            decoder_input = self.lstm(decoder_input)[0].reshape(self.decoder_input_size).permute((0, 3, 1, 2))
        return [self.decode(decoder_input), mu, c]
    
class onetedmodel(basiconetmodel, CondAutoEncoder):

    def __init__(self, h_e, h_e1, h_e2, h_d, de_type='prod', coder_type='onet', nt=101, h_out=1, h_in=1, last_size=6):
        CondAutoEncoder.__init__(self, latent_dim=0, encoder=nn.Identity(), decoder=nn.Identity(), code_mode=de_type, decoder_input_layer=nn.Identity())

        self.encoder = Resnet18Encoder(in_channels=h_in, last_size=[nt, last_size], hidden_dims=h_e, strides=(1,2))
        self.decoder = Resnet18Decoder(out_channels=h_out, last_size=[nt, last_size], hidden_dims=h_d, scales=(1, 2), output_size=[nt, 321])

        self.de_type = de_type
        self.coder_type = coder_type
        self.set_basic(h_e1, h_e2, h_lstm, nt, coder_type, coder_kernel, last_size)

class ounetedmodel(basiconetmodel, Unet):

    def __init__(self, h_e, h_e1, h_e2, h_d, h_lstm=None, de_type='prod', coder_type='onet', coder_kernel=1, nt=101, h_out=1, h_in=1, 
                 last_size=6, decoder_input_size=None):

        encoder = ResnetEncoder_Unet(in_channels=h_in, last_size=[nt, last_size], hidden_dims=h_e, strides=(1,2))
        # decoder = ResnetDecoder_Unet(out_channels=h_out, last_size=[nt, last_size], hidden_dims=h_d, scales=(1, 2), output_size=[nt, 321], encoder_hidden_dims=[h_e[-i] for i in range(1, len(h_e)+1)]+[h_in])
        decoder = ResnetDecoder_Unet(out_channels=h_out, last_size=[nt, last_size], hidden_dims=h_d, 
                                    sizes=[[101,11],[101,21],[101,41],[101,81],[101,161],[101,321]],
                                    output_size=[nt, 321], encoder_hidden_dims=[h_e[-i] for i in range(1, len(h_e)+1)]+[h_in])
        Unet.__init__(self, latent_dim=0, encoder=encoder, decoder=decoder, code_mode=de_type, decoder_input_layer=nn.Identity())

        self.de_type = de_type
        self.coder_type = coder_type
        self.set_basic(h_e1, h_e2, h_lstm, nt, coder_type, coder_kernel, last_size)
        
class ounetbedmodel(basiconetmodel, BranchUnet):

    def __init__(self, h_e, h_e1, h_e2, h_d, h_lstm=None, de_type='prod', coder_type='onet', coder_kernel=1, nt=101, h_out=1, h_in=1, 
                 last_size=6, decoder_layer_sizes=[11, 21, 41, 81, 161, 321], nn_out=321):
        
        encoder = ResnetEncoder_Unet(in_channels=h_in, last_size=[nt, last_size], hidden_dims=h_e, strides=(1,2))
        decoders = []
        for i in range(h_out):
            decoders.append(ResnetDecoder_Unet(out_channels=1, last_size=[nt, last_size], hidden_dims=h_d, 
                                          sizes=[[nt,nn] for nn in decoder_layer_sizes],
                                          output_size=[nt, nn_out], encoder_hidden_dims=[h_e[-i] for i in range(1, len(h_e)+1)]+[h_in]))
        
        BranchUnet.__init__(self, latent_dim=0, encoder=encoder, decoder=decoders, code_mode=de_type, decoder_input_layer=nn.Identity())

        # self.decoder = ResnetDecoder_Unet(out_channels=h_out, last_size=[nt, last_size], hidden_dims=h_d, scales=(1, 2), output_size=[nt, 321], encoder_hidden_dims=[h_e[-i] for i in range(1, len(h_e)+1)]+[h_in])

        self.de_type = de_type
        self.coder_type = coder_type
        self.set_basic(h_e1, h_e2, h_lstm, nt, coder_type, coder_kernel, last_size)

class WingViT(ViT):
    
    def __init__(self, image_size, patch_size, fun_dim = 3, out_dim = 1, n_layers = 5, n_hidden = 256, n_head = 8, mlp_ratio = 4, add_mesh = 0, dropout=0, pos_embedding='sincos', act='gelu'):
        super().__init__(image_size, patch_size, fun_dim + 2, out_dim, n_layers, n_hidden, n_head, mlp_ratio, add_mesh, dropout, pos_embedding, act)
        
    def forward(self, inputs, code: torch.Tensor) -> torch.Tensor:
        B_, C_, H_, W_ = inputs.shape
        cat_inputs = torch.concatenate((inputs, code[:, :2, None, None].repeat((1, 1, H_, W_))), axis=1)
        return [super().forward(cat_inputs)]

class WingTransformer(Transolver):
    
    def __init__(self, n_layers=5, n_hidden=256, n_head=8, slice_num=32, mlp_ratio=4, h_in=5, h_out=3, is_flatten=False, dual_slices=False) -> None:
        
        super().__init__(3, h_in-1, h_out, n_layers, n_hidden, n_head, slice_num=slice_num, mlp_ratio=mlp_ratio, mesh_type=['2d', 'point'][int(is_flatten)], dual_slices=dual_slices)
        
        self.is_flatten = is_flatten
        
    def _process(self, inputs, code: torch.Tensor) -> torch.Tensor:
        
        _, C_, H_, W_ = inputs.shape
        x  = inputs.permute(0, 2, 3, 1)
        fx = code[:, None, None, :2].repeat((1, H_, W_, 1))
        return super()._process(x, fx)
    
    def forward(self, inputs, code: torch.Tensor) -> torch.Tensor:
        B_, C_, H_, W_ = inputs.shape
        return [super().forward(inputs, code).permute(0, 3, 1, 2)]

class WingTokenTransformer(TokenSpaceTransolver):
    
    def __init__(self, n_layers=5, n_hidden=256, n_head=8, slice_num=32, mlp_ratio=4, h_in=5, h_out=3, is_flatten=False) -> None:
        
        super().__init__(3, h_in-1, h_out, n_layers, n_hidden, n_head, slice_num=slice_num, mlp_ratio=mlp_ratio, mesh_type=['2d', 'point'][int(is_flatten)])
        
        self.is_flatten = is_flatten
        
    def _process(self, inputs, code: torch.Tensor) -> torch.Tensor:
        
        _, C_, H_, W_ = inputs.shape
        x  = inputs.permute(0, 2, 3, 1)
        fx = code[:, None, None, :2].repeat((1, H_, W_, 1))
        return super()._process(x, fx)
    
    def forward(self, inputs, code: torch.Tensor) -> torch.Tensor:
        B_, C_, H_, W_ = inputs.shape
        return [super().forward(inputs, code).permute(0, 3, 1, 2)]
    
class WingUTransolver(UTransolver):
    
    def __init__(self, depths=[2, 5, 8, 5, 2], n_hidden=256, n_head=8, slice_num=32, mlp_ratio=4, h_in=5, h_out=3, is_flatten=False, dual_slices=False, u_shape=1) -> None:
        
        super().__init__(3, h_in-1, h_out, depths, n_hidden, n_head, slice_num=slice_num, mlp_ratio=mlp_ratio, mesh_type=['2d', 'point'][int(is_flatten)], dual_slices=dual_slices, u_shape=u_shape)
        
        self.is_flatten = is_flatten
        
    def _process(self, inputs, code: torch.Tensor) -> torch.Tensor:
        
        _, C_, H_, W_ = inputs.shape
        x  = inputs.permute(0, 2, 3, 1)
        fx = code[:, None, None, :2].repeat((1, H_, W_, 1))
        return super()._process(x, fx).permute(0, 3, 1, 2)
    
    def forward(self, inputs, code: torch.Tensor) -> torch.Tensor:
        B_, C_, H_, W_ = inputs.shape
        return [super().forward(inputs, code).permute(0, 3, 1, 2)]
    
class WingPlainTransformer(Transolver):
    '''
    concatenate the code to the input before processing, fx to be sinusoidal position encoding
    '''
    
    def __init__(self, n_layers=5, n_hidden=256, n_head=8, slice_num=32, mlp_ratio=4, h_in=3, h_out=3, is_flatten=False, u_shape=False, placeholder={'type': 'random'}) -> None:
        
        super().__init__(h_in+2, 0, h_out, n_layers, n_hidden, n_head, slice_num, mlp_ratio, ['2d', 'point'][int(is_flatten)], u_shape, placeholder=placeholder)
        
        self.is_flatten = is_flatten
        
    def _process(self, inputs, code: torch.Tensor) -> torch.Tensor:
        
        _, C_, H_, W_ = inputs.shape
        x  = torch.cat((inputs.permute(0, 2, 3, 1), code[:, None, None, :2].repeat((1, H_, W_, 1))), dim=-1)
        return super()._process(x, fx=None)
    
    def forward(self, inputs, code: torch.Tensor) -> torch.Tensor:
        B_, C_, H_, W_ = inputs.shape
        return [super().forward(inputs, code).permute(0, 3, 1, 2)]
 
class WingEDTransformer(EncoderDecoderTransolver):
    
    def __init__(self, n_layers_enc, n_layers_dec, n_hidden=256, n_head=8, slice_num=32, mlp_ratio=4, h_in=5, h_out=3, is_flatten=False) -> None:
        
        assert h_in >= 5, 'must input reference'
        super().__init__(3, 2, 5, h_out, n_layers_enc, n_layers_dec, n_hidden, n_head, slice_num, mlp_ratio, ['2d', 'point'][int(is_flatten)], add_mesh=0)
        
        self.is_flatten = is_flatten
        
    def _process(self, inputs, code: torch.Tensor, **kwargs) -> torch.Tensor:
        
        _, C_, H_, W_ = inputs.shape
        x  = inputs.permute(0, 2, 3, 1)
        fx = code[:, None, None, :2].repeat((1, H_, W_, 1))
        return super()._process(x[:, :, :, :3], fx, x)
    
    def forward(self, inputs, code: torch.Tensor) -> torch.Tensor:
        B_, C_, H_, W_ = inputs.shape
        return [super().forward(inputs, code).permute(0, 3, 1, 2)]

class WingEDTransformer_Mesh(EncoderDecoderTransolver):
    
    def __init__(self, n_layers_enc, n_layers_dec, n_hidden=256, n_head=8, slice_num=32, mlp_ratio=4, h_in=5, h_out=3, is_flatten=False) -> None:
        
        assert h_in >= 5, 'must input reference'
        super().__init__(3, 2, 5, h_out, n_layers_enc, n_layers_dec, n_hidden, n_head, slice_num, mlp_ratio, ['2d', 'point'][int(is_flatten)], add_mesh=3)
        
        self.is_flatten = is_flatten
        
    def _process(self, inputs, code: torch.Tensor, **kwargs) -> torch.Tensor:
        
        _, C_, H_, W_ = inputs.shape
        x  = inputs.permute(0, 2, 3, 1)
        fx = code[:, None, None, :2].repeat((1, H_, W_, 1))
        return super()._process(x[:, :, :, :3], fx, x)
     
    def forward(self, inputs, code: torch.Tensor) -> torch.Tensor:
        B_, C_, H_, W_ = inputs.shape
        mesh = inputs[:, :3].permute(0, 2, 3, 1)
        return [super().forward(inputs, code, mesh=mesh, ref_mesh=mesh).permute(0, 3, 1, 2)]
    
class WingPDETransformer(PDEImpl):

    def __init__(self, patch_size, window_size=(8,8), fun_dim = 3, out_dim = 1, n_layers = 5, n_hidden = 256, n_head = 8, mlp_ratio = 4, depth=[2, 5, 8, 5, 2],
                 type_cond: str = 'cat', ct_active: bool = False, dropout=0, output_type=None, is_final_adp=False, sampler = 1):
        # n_layers=5, n_hidden=256, n_head=8, slice_num=32, mlp_ratio=4, h_in=5, h_out=3
        kwargs = {
            'in_channels':      fun_dim + (2 if type_cond in ['cat'] else 0),
            'out_channels':     out_dim,
            'window_size':      int(0.5 * (window_size[0] + window_size[1])),    # in PDE stage, the window
            'patch_size':       int(0.5 * (patch_size[0] + patch_size[1])),    # in patch embedding step, patch
            'hidden_size':      n_hidden,       # multiple by 2 for each block
            'depth':            depth,
            'num_heads':        n_head,
            'mlp_ratio':        mlp_ratio,
            'num_classes':      2,              # classification (condition) embedding
            'class_dropout_prob': 0,            # also for classification (condition) embedding
            'periodic':         False,
            'carrier_token_active': ct_active,
            'dit_active':       False,
            'inj_active':       type_cond not in ['cat'],
            'output_type':      output_type,
            'is_final_adp':     is_final_adp,
            'sampler':          sampler
        }
        
        super().__init__(**kwargs)
        self.type_cond = type_cond

    def forward(self, inputs, code: torch.Tensor) -> torch.Tensor:
        if self.type_cond in ['cat']:
            B_, C_, H_, W_ = inputs.shape
            inputs = torch.concatenate((inputs, code[:, :2, None, None].repeat((1, 1, H_, W_))), axis=1)
        return [super().forward(inputs, None, None, c = (None if self.type_cond in ['cat'] else code))]

'''

This part is for airfoil prediction models

    0      1     2  3   4  5     6-15  16-25 26 27 28   
    igroup icond ma aoa re t_max cst_u cst_l cl cd cm

'''  

class multiinput_resnet2dsmodel(nn.Module):

    def __init__(self, h_e1, h_e2, h_d1, h_d2, sizes=[24, 100, 321], last_size=5, h_out=2):
        super().__init__()
        
        self.foil_encoder = mlp(in_features=21, out_features=h_e1[-1], hidden_dims=h_e1[:-1])
        self.wing_encoder = mlp(in_features=3,  out_features=h_e2[-1], hidden_dims=h_e2[:-1])
        
        if h_out == 2:
            decoders = [convDecoder(out_channels=1, last_size=[last_size], hidden_dims=h_d2, 
                                    sizes=sizes, last_conv='bottleneck'),
                        convDecoder(out_channels=1, last_size=[last_size], hidden_dims=h_d2, 
                                        sizes=sizes, last_conv='bottleneck')]

            self.decoder = BranchDecoderModel(input_channels=h_e1[-1]+h_e2[-1], decoders=decoders, decoder_input_layer=h_d1)
        elif h_out == 1:
            decoder = convDecoder(out_channels=1, last_size=[last_size], hidden_dims=h_d2, 
                                    sizes=sizes, last_conv='bottleneck')

            self.decoder = DecoderModel(input_channels=h_e1[-1]+h_e2[-1], decoder=decoder, decoder_input_layer=h_d1)    
        
    def forward(self, inputs):
        foil_encoded = self.foil_encoder(inputs[:, 3:])
        wing_encoded = self.wing_encoder(inputs[:, :3])
        encoded = torch.concatenate((foil_encoded, wing_encoded), dim=1)
        return self.decoder(encoded)

def resnetedmodel(h_e, h_d1, h_d2, sizes=[24, 100, 321], h_out=1, h_in=1):

    encodercell = convEncoder(in_channels=h_in, last_size=[4], hidden_dims=h_e)
    decodercell = convDecoder(out_channels=1, last_size=[4], hidden_dims=h_d2, sizes=sizes, last_conv='bottleneck')
    ae_model = EncoderDecoder(latent_dim=32, encoder=encodercell, decoder=decodercell, code_mode='ed', code_dim=3, decoder_input_layer=h_d1)

    return ae_model

def resnetunetmodel(h_e, h_d1, h_d2, sizes=[19, 80, 321], h_out=1, h_in=1):

    encodercell = convEncoder_Unet(in_channels=h_in, last_size=[4], hidden_dims=h_e)
    decodercell = convDecoder_Unet(out_channels=h_out, last_size=[4], hidden_dims=h_d2, sizes=sizes, last_conv='bottleneck', 
                                   encoder_hidden_dims=[h_e[-i] for i in range(1, len(h_e)+1)]+[h_in])
    ae_model = Unet(latent_dim=32, encoder=encodercell, decoder=decodercell, code_mode='ed', code_dim=3, decoder_input_layer=h_d1)

    return ae_model

def bresnetunetmodel(h_e, h_d1, h_d2, sizes=[19, 80, 321], h_out=1, h_in=1):

    encodercell = convEncoder_Unet(in_channels=h_in, last_size=[4], hidden_dims=h_e)
    decodercells = []
    for i in range(h_out):
        decodercells.append(convDecoder_Unet(out_channels=1, last_size=[4], hidden_dims=h_d2, sizes=sizes, last_conv='bottleneck', 
                                   encoder_hidden_dims=[h_e[-i] for i in range(1, len(h_e)+1)]+[h_in]))
    ae_model = BranchUnet(latent_dim=32, encoder=encodercell, decoder=decodercells, code_mode='ed', code_dim=3, decoder_input_layer=h_d1)

    return ae_model

class bresnetunetmodel1(BranchUnet):

    def __init__(self, h_e, h_d1, h_d2, sizes=[19, 80, 321], h_out=1, h_in=1):

        encodercell = convEncoder_Unet(in_channels=h_in, last_size=[4], hidden_dims=h_e)
        decodercells = []
        for _ in range(h_out):
            decodercells.append(convDecoder_Unet(out_channels=1, last_size=[4], hidden_dims=h_d2, sizes=sizes, last_conv='bottleneck', 
                                    encoder_hidden_dims=[h_e[-i] for i in range(1, len(h_e)+1)]+[h_in]))
        super().__init__(latent_dim=32, encoder=encodercell, decoder=decodercells, code_mode='ed', code_dim=3, decoder_input_layer=h_d1)


'''

This part is for SLD prediction models

input dimensions
        0.   1.    2.           3               4             5             6                7
index: id, AoA, Mach, swept_angle, dihedral_angle, aspect_ratio, tapper_ratio, tip_twist_angle, tip2root_thickness_ratio, ref_area, root_thickness, 
    cstu, cstl, real_cl, real_cd, recon_cl, recon_cd

delta: xLE, yLE, zLE, alpha, chord, thick, cstu, cstl, Ma
       0    1    2    3      4      5      6-15  16-25 26
'''


def simpledecode(h_d, h_e, nt=61, kernel=3, dropout=0.):
    
    encodercell = h_e 
    decodercell = convEncoder(in_channels=h_d[0], last_size=[nt], hidden_dims=h_d[1:], kernel_sizes=kernel, strides=1, paddings=int(kernel/2), pool_kernels=0, dimension=1, basic_layers={'dropout': dropout})

    decoder = DecoderModel(decoder=decodercell, decoder_input_layer=encodercell)

    return decoder

class triinput_simplemodel(nn.Module):

    def __init__(self, h_e1, h_e2, h_e3, h_d1, nt, dropout=0.):
        super().__init__()
        
        self.foil_encoder = mlp(in_features=21, out_features=h_e1[-1], hidden_dims=h_e1[:-1], basic_layers={'dropout': dropout})
        self.wing_encoder = mlp(in_features=6,  out_features=h_e2[-1], hidden_dims=h_e2[:-1], basic_layers={'dropout': dropout})
        self.cond_encoder = mlp(in_features=2,  out_features=h_e3[-1], hidden_dims=h_e3[:-1], basic_layers={'dropout': dropout})

        self.mlp = mlp(in_features=h_e1[-1]+h_e2[-1]+h_e3[-1], out_features=h_d1[-1], hidden_dims=h_d1[:-1], basic_layers={'dropout': dropout})
        if nt > 0:
            self.last_size = [-1, int(h_d1[-1] / nt), nt]
        else:
            self.last_size = [-1, h_d1[-1]]
        
    def forward(self, inputs):
        foil_encoded = self.foil_encoder(inputs[:, 8:])
        wing_encoded = self.wing_encoder(inputs[:, 2:8])
        cond_encoded = self.cond_encoder(inputs[:, :2])
        encoded = torch.concatenate((cond_encoded, foil_encoded, wing_encoded), dim=1)
        return self.mlp(encoded).reshape(self.last_size)

class triinput_simplemodel1(nn.Module):

    def __init__(self, h_e1, h_e2, h_e3, h_d1, nt, dropout=0.):
        super().__init__()
        
        self.foil_encoder = mlp(in_features=21, out_features=h_e1[-1], hidden_dims=h_e1[:-1], basic_layers={'dropout': dropout})
        self.wing_encoder = mlp(in_features=6,  out_features=h_e2[-1], hidden_dims=h_e2[:-1], basic_layers={'dropout': dropout})
        self.cond_encoder = mlp(in_features=2,  out_features=h_e3[-1], hidden_dims=h_e3[:-1], basic_layers={'dropout': dropout})

        self.mlp = mlp(in_features=h_e1[-1]+h_e2[-1]+h_e3[-1], out_features=h_d1[-2], hidden_dims=h_d1[:-2], basic_layers={'dropout': dropout})
        self.last_mlp = nn.Linear(in_features=h_d1[-2], out_features=h_d1[-1])
        if nt > 0:
            self.last_size = [-1, int(h_d1[-1] / nt), nt]
        else:
            self.last_size = [-1, h_d1[-1]]
        
    def forward(self, inputs):
        foil_encoded = self.foil_encoder(inputs[:, 8:])
        wing_encoded = self.wing_encoder(inputs[:, 2:8])
        cond_encoded = self.cond_encoder(inputs[:, :2])
        encoded = torch.concatenate((cond_encoded, foil_encoded, wing_encoded), dim=1)
        return self.last_mlp(self.mlp(encoded)).reshape(self.last_size)