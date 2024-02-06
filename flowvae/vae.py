'''

the vae model classes(vanilla, conditional, etc.)
adapted from AntixK, PyTorch-VAE. https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

同一个encoder，接不同decoder，1.加上工况，直接预测cl，cd 2.不加工况，直接预测抖振攻角和升阻力

'''



import torch
from torch import nn
from torch.nn import functional as F

from flowvae.dataset import ConditionDataset
from flowvae.base_model import Encoder, Decoder, _decoder_input

import numpy as np
import copy
from typing import List, Callable, NewType, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor

Tensor = NewType('Tensor', torch.tensor)

# from utils import get_force

from flowvae.post import get_aoa, get_vector, get_force_cl, WORKCOD, get_force_1dc

class AutoEncoder(nn.Module):

    def __init__(self,         
                 latent_dim: int,
                 encoder: Encoder = None,
                 decoder: Decoder = None,
                 decoder_input_layer: int = 0,
                 decoder_input_dropout: float = 0.,
                 device = 'cuda:0',
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim        # the total dimension of latent variable (include code dimension)
        # self.in_channels = in_channels
        self.paras = kwargs
        self.paras['decoder_input_layer'] = decoder_input_layer
        self.paras['decoder_input_dropout'] = decoder_input_dropout
        self.device = device

        if encoder is None:
            pass
        else:
            self.encoder = encoder
            self.fc_mu = nn.Linear(self.encoder.last_flat_size, latent_dim)

        if decoder is None:
            pass
        elif isinstance(decoder, list):
            _decoder_inputs = []
            self.decoders = nn.ModuleList(decoder)
            for decoder in self.decoders:
                decoder_input = _decoder_input(typ=decoder_input_layer, 
                                               ld=self.latent_dim, 
                                               lfd=decoder.last_flat_size,
                                               drop_out=decoder_input_dropout)
                _decoder_inputs.append(decoder_input)
            self.decoder_inputs = nn.ModuleList(_decoder_inputs)
        else:
            self.decoder = decoder
            self.decoder_input = _decoder_input(typ=decoder_input_layer, 
                                                ld=self.latent_dim, 
                                                lfd=self.decoder.last_flat_size, 
                                                drop_out=decoder_input_dropout)

    def encode(self, input: Tensor) -> Tensor:
        # print(input.size())
        results = self.encoder(input)
        # print(results.size())
        return self.fc_mu(results)

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # output_padding 可以用来补齐在卷积编码器中向下取整的损失
        result = self.decoder(result)
        # print(result.size())
        
        return result

    def forward(self, inputs: Tensor) -> List[Tensor]:

        mu = self.encode(inputs)
        return [self.decode(mu), mu]

class EncoderDecoder(AutoEncoder):
    '''
    the model of VAE

    paras:
    ===

    - `latent_dim`| `int` |
        - the total dimension of latent variable (include code dimension)
    - `encoder`| | the encoder
    - `decoder`| | the decoder
    - `code_mode`| `str` | the mode to introduce condition codes in the concatenator. See the table below.
    - `code_dim`| `int` | 
        - **Default**  `1` 
        - the condition code dimension (same with that in the dataset)
    - `device`|| 
        - **Default**  `cuda:0`
        - training device
    - `dataset_size`| `Tuple` | 
        - **Default** `None`
        - (`code_mode` = `'im'`, `'semi'`, `'ex'`, `'ae'`)
        - the size to initial the prior latent variable storage. The size should be ($N_f$, $N_c$)
    - `decoder_input_layer`| `float` | 
        - **Default**  `0`
        - the amount of dense connect layers between latent variable `z` and the decoder input, see [Decoder input layers](#decoder-input-layers)|
    - `code_layer`|`List`|
        - **Default**  `[]`
        - (`code_mode` = `'semi'`, `'ex'`, `'ae'`, `'ed'`, `ved`, `ved1`)
        - The layers between the real condition code input and where it is concatenate to the latent variables, see [Coder input layers](#code-input-layers)
    
    '''

    def __init__(self,
                 latent_dim: int,
                 encoder: Encoder,
                 decoder: Decoder,
                 code_mode: str,
                 dataset_size: Tuple[int, int] = None,
                 decoder_input_layer: int = 0,
                 code_dim: int = 1,
                 code_layer = [],
                 device = 'cuda:0',
                 **kwargs) -> None:


        super().__init__(latent_dim, encoder, decoder, decoder_input_layer, 0., device, **kwargs)
        
        self.code_dim = code_dim
        self.cm = code_mode
        self.series_data = {}

        if code_mode in ['semi', 'im', 'ex', 'ae']:
            self.set_aux_data(dataset_size)

        ld = self.latent_dim
        cd = self.code_dim
        fd = ld - cd
        lfe = self.encoder.last_flat_size

        if code_mode in ['ex']:
            self.fc_mu = nn.Sequential(
                nn.Linear(lfe + cd, fd), nn.BatchNorm1d(fd), nn.LeakyReLU(),
                nn.Linear(fd, fd))
            self.fc_var = nn.Sequential(
                nn.Linear(lfe + cd, fd), nn.BatchNorm1d(fd), nn.LeakyReLU(),
                nn.Linear(fd, fd))
        elif code_mode in ['ae', 'ed']:
            self.fc_mu = nn.Linear(lfe, fd)
        elif code_mode in ['ved', 'ved1']:
            self.fc_mu = nn.Linear(lfe, fd)
            self.fc_var = nn.Linear(lfe, fd)    
        elif code_mode in ['semi', 'im']:
            self.fc_mu = nn.Linear(lfe, ld)
            self.fc_var = nn.Linear(lfe, ld)

        if code_layer != []:
            ld = ld - cd + code_layer[-1]
            # if another layers are inserted betweem code input and first layer of the decoder
            code_layers = []
            in_dim = cd
            for code_layer_dim in code_layer:
                code_layers.append(nn.Linear(in_dim, code_layer_dim))
                code_layers.append(nn.LeakyReLU())
                in_dim = code_layer_dim
            self.fc_code = nn.Sequential(*code_layers)
        else:
            self.fc_code = nn.Identity()
        
    def set_aux_data(self, dataset_size: Tuple[int, int]):
        n_airfoil = dataset_size[0]
        n_condi   = dataset_size[1]
        n_z       = self.latent_dim - self.code_dim
        self.series_data = {
            'series_latent': torch.cat(
                [torch.zeros((n_airfoil, n_condi, 1, n_z)), 
                 torch.ones((n_airfoil, n_condi, 1, n_z))], dim=2).to(self.device),
            'series_avg_latent': torch.cat(
                [torch.zeros((n_airfoil, 1, n_z)),
                 torch.ones((n_airfoil, 1, n_z))], dim=1).to(self.device)}


    def preprocess_data(self, fldata: ConditionDataset=None):
        
        self.geom_data = {}
        # print(' === Warning:  preprocess_data is not implemented. If auxilary geometry data is needed, please rewrite vae.preprocess_data')
        # return
        '''
        produce geometry data
        - x01, x0p: the x coordinate data for smooth calculation
        '''
        from flowvae.post import clustcos
        
        nn = 201
        xx = [clustcos(i, nn) for i in range(nn)]
        all_x = torch.from_numpy(np.concatenate((xx[::-1], xx[1:]), axis=0)).to(self.device).float()
        self.geom_data['xx']  = all_x
        self.geom_data['x01'] = all_x[2:] - all_x[1: -1]
        self.geom_data['x0p'] = all_x[1: -1] - all_x[:-2]
        self.geom_data['dx'] = all_x[1:] - all_x[:-1]
        self.geom_data['avgdx'] = torch.mean(self.geom_data['dx']**2)

        # if fldata is not None:

        #     geom = torch.cat((all_x.repeat(fldata.data.size(0), 1).unsqueeze(1), fldata.data[:, 0].unsqueeze(1)), dim=1)
        #     profile = fldata.data[:, 1]
        #     self.geom_data['all_clcd'] = get_force_1dc(geom, profile, fldata.cond.squeeze())
        #     print('all_clcd size:',  self.geom_data['all_clcd'].size())

        #     geom = torch.cat((all_x.repeat(fldata.refr.size(0), 1).unsqueeze(1), fldata.refr[:, 0].unsqueeze(1)), dim=1)
        #     profile = fldata.refr[:, 1]
        #     self.geom_data['ref_clcd'] = get_force_1dc(geom, profile, fldata.ref_condis.squeeze())
        #     print('ref_clcd size:',  self.geom_data['ref_clcd'].size())

    def encode(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :kwargs
         - `code`: for `code_mode` = `ex` or `ae`
        :return: (Tensor) List of latent codes

        :self.code_mode:
            - `ex`:     explicitly introduce the code
                        (the code is cat with the result from encoder, then be put into fc_mu/fc_var)
                        (the output dimension of fc_mc/fc_var is one below latent_dim, and is cat with the code at first digit)
            - `ae`:     auto encoder
                        (only fc_mc is used to generate mu from the result from encoder)
                        (the result of fc_mc is cat with the code)
            - others:   implicit/ semi-implicit
                        (the first digit of latent dim is recognized as code)
        """
        # input channel control is moved to ml_operate, ori: input[:, :self.in_channel]
        result = self.encoder(input)
        if self.cm in ['ex']:
            result = torch.cat([result, kwargs['code']], dim=1)
            mu = self.fc_mu(result)
            log_var = self.fc_var(result)
        elif self.cm in ['ae', 'ed']:
            mu = self.fc_mu(result)
            log_var = torch.zeros_like(mu)
        # Split the result into mu and var components of the latent Gaussian distribution
        elif self.cm in ['im', 'semi', 'ved', 'ved1']:
            mu = self.fc_mu(result)
            # TODO here log_var's first element should be 0 for im and semi
            log_var = self.fc_var(result)

        # thr output of ae and ex is in dimension 11
        return [mu, log_var]

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        '''
        Go through the whole network and get the result
        :param input: (Tensor) the input tensor N x B x H x W
        :kwargs
         - `code`: for `self.code_mode`=`ae`, `ex`, `semi`
        '''

        mu, log_var = self.encode(input, **kwargs)
        
        if self.cm in ['ae', 'ed']:
            zf = mu
            zc = self.fc_code(kwargs['code'])
            z = torch.cat([zc, zf], dim=1)
        
        elif self.cm in ['ved']:
            #! this is log_var, so std_var=1 means log_var=0
            zc = self.reparameterize(kwargs['code'], torch.zeros_like(kwargs['code'], device=self.device))
            zf = self.reparameterize(mu, log_var)
            zc = self.fc_code(zc)
            z = torch.cat([zc, zf], dim=1)
        
        elif self.cm in ['ved1']:
            zf = self.reparameterize(mu, log_var)
            zc = self.fc_code(kwargs['code'])
            z = torch.cat([zc, zf], dim=1)

        elif self.cm in ['semi']:
            z = self.reparameterize(mu, log_var)
            zc = self.fc_code(kwargs['code'])
            # print("new, old:", kwargs['code'], z[:, 0])
            z[:, :zc.size()[1]] = zc

        elif self.cm in ['ex']:
            zf = self.reparameterize(mu, log_var)
            zc = self.fc_code(kwargs['code'])
            z = torch.cat([zc, zf], dim=1)

        elif self.cm in ['im']:
            z = self.reparameterize(mu, log_var)
            # TODO: the mode `im` is not consist with code layers!

        return  [self.decode(z), mu, log_var]

    def loss_function(self, real,
                      *args,
                      **kwargs) -> dict:
        '''
        Computes the reconstruction and smooth loss function.
        
        :param args:
         - `recons`:        the reconstructed flowfield, shape: B x C x (SIZE) 
        :param kwargs:
         - `sm_weight`:     the weight of the smooth loss
         - `sm_mode`:       the mode to calculate smooth loss
                - `NS`:     use navier-stokes equation's conservation of mass and moment to calculate smooth
                - `NS_r`    use the residual between mass/moment flux of reconstructed and ground truth as loss
                - `offset`  offset diag. the field for several distance and sum the difference between field before and after move
                - `1d`      one-dimensional data (adapted from Runze)

        :return:
        '''

        recons = args[0]

        ref = kwargs['ref']
        sm_weight = kwargs['sm_weight']
        ge_weight = kwargs['ge_weight']

        loss = torch.FloatTensor([0.0]).to(self.device)
        sm_loss = torch.FloatTensor([0.0]).to(self.device)
        ge_loss = torch.FloatTensor([0.0]).to(self.device)

        # print(recons.size(), ref.size(), real.size())
        # input()
        recons_loss = F.mse_loss(recons + ref, real)
        # the real reconstruction is done by setting ref = 0
        loss = loss + recons_loss

        if sm_weight > 0.0:

            if kwargs['sm_mode'] in ['NS', 'NS_r']:
                
                sm_loss_recon = NSres(recons, mesh=real[:, :2], w_mom=kwargs['moment_weight'], dev=self.device)

                if kwargs['sm_mode'] == 'NS':

                    sm_loss = torch.mean(sm_loss_recon)
                else:
                    real_code_indx = kwargs['real_code_indx']
                    real_indx = kwargs['real_indx']
                    for ii, (isample, icode) in enumerate(zip(real_indx, real_code_indx)):
                        sm_loss_recon[ii] -= self.new_data['ns_mass'][isample, icode]
                    sm_loss = torch.mean(torch.relu(sm_loss_recon))

            elif kwargs['sm_mode'] == 'offset':
                sm_loss = smoothness(recons, mesh=real[:, :2], offset=kwargs['sm_offset'])
            
            elif kwargs['sm_mode'] == '1d':
                x01 = self.geom_data['x01']
                x0p = self.geom_data['x0p']
                sm_loss = roughness_penalty(x01, x0p, recons.squeeze(1), dev=self.device)

            else:
                raise AttributeError()
            
            loss += sm_weight * sm_loss
        
        if ge_weight > 0.0:
            ge_loss = gradient_enhance_1d(self.geom_data['dx'], self.geom_data['avgdx'], recons + ref, real)
            loss += ge_weight * ge_loss
        
        origin_loss = {'loss': loss, 'recons':recons_loss, 'smooth':sm_loss, 'grad': ge_loss}    

        if self.cm in ['ved', 'ved1']:
            #! mu and var returned not include code dimensions
            mu = args[1]
            log_var = args[2]
            indx_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            # print(origin_loss['loss'], indx_loss, kwargs['indx_weight'])
            origin_loss['loss'] += indx_loss * kwargs['indx_weight']
            origin_loss['indx'] = indx_loss
            # print(origin_loss['loss'], indx_loss)
            # input()

        return origin_loss
    
    def series_loss_function(self, real, *args, **kwargs):
        '''
         - `real`:          Tensor, shape: [B x C0 x (SIZE)],   the input flowfield
                                C0 stands for the channel number with geometry
        param args
        ---
         - `recons`:        Tensor, shape: [B x C x (SIZE)],    the reconstructed flowfield 
         - `mu`:            Tensor, shape: [B x D],   mu of the latent
         - `log_var`:       Tensor, shape: [B x D],   log_var of the latent 
        
        param kwargs
        ===
        == reconstruction loss ==
         - `ref`            Tensor, shape: [B x C x (SIZE)],    the reference flowfield
         - `recons_mesh`
        
        == smooth loss ==
         - `sm_weight`:     float,  the weight of the smooth loss
         - `sm_mode`:       str,    the mode to calculate smooth loss
                - `NS`:     use navier-stokes equation's conservation of mass and moment to calculate smooth
                - `NS_r`    use the residual between mass/moment flux of reconstructed and ground truth as loss
                - `offset`  offset diag. the field for several distance and sum the difference between field before and after move
                - `1d`      one-dimensional data (adapted from Runze)
         - `moment_weight`  float,  (need for `NS`, `NS_r`) the weight of momentum flux residual
         - `sm_offset`      int,    (need for `offset`) the diag. direction move offset
        
        == code loss ==
         - `real_code`      Tensor[float], shape: [B], the value of the code (currently the AoAs), Tensor
         - `code_weight`    float,  the weight of code loss, better for 0.1
        
        == index KLD loss ==
         - `indx_weight`    float,  the weight of index KLD loss
        
        == aerodynamic loss ==
         - `aero_weight`    float,  the weight of aerodynamic loss
        
        == index information for find data in aux_data ==
         (need for `NS_r` mode for smooth, aerodynamic loss, index KLD loss)
         - `real_code_indx` Tensor[int], shape: [B],    the index of code (0 ~ n_c-1)
         - `real_indx`      Tensor[int], shape: [B],    the index of airfoil (0 ~ n_airfoil-1)

        '''

        origin_loss = self.loss_function(real, *args, **kwargs)
        
        recons = args[0]
        mu = args[1]
        log_var = args[2]

        real_code = kwargs['real_code']
        code_weight = kwargs['code_weight']

        if self.cm in ['semi', 'im']:
            
            code_loss = F.mse_loss(mu[:,:self.code_dim], real_code)

            origin_loss['loss'] += code_loss * code_weight
            origin_loss['code'] = code_loss

            # only left mu & log_var for foil
            mu = mu[:, self.code_dim:]
            log_var = log_var[:, self.code_dim:]

        #TODO very slow!

        real_code_indx = kwargs['real_code_indx']
        real_indx = kwargs['real_indx']
        indx_weight = kwargs['indx_weight']
        aero_weight = kwargs['aero_weight']

        # print(origin_loss)

        indx_loss = 0.0
        aero_loss = 0.0

        for isample, icode, imu, ivar, irec, iaoa in zip(real_indx, real_code_indx, mu, log_var, recons, real_code):

            # print(isample, icode, self.series_data['series_avg_latent'][isample, 0], self.series_data['series_avg_latent'][isample, 1])
            if indx_weight > 0.0:

                self.series_data['series_latent'][isample, icode, 0] = imu.detach()
                d_mu = imu - self.series_data['series_avg_latent'][isample, 0]

                if self.cm in ['ae']:
                    # only MSE loss of mu
                    indx_loss += torch.mean(d_mu**2)
                else:
                    # index KL loss
                    self.series_data['series_latent'][isample, icode, 1] = ivar.detach()
                    d_log_var = ivar - self.series_data['series_avg_latent'][isample, 1]
                    indx_loss += -0.5 * torch.sum(1 + d_log_var - (d_mu**2) / self.series_data['series_avg_latent'][isample, 1].exp() - d_log_var.exp())
                # print(indx_loss)

            if aero_weight > 0.0:
                WORKCOD['AoA'] = iaoa
                n_coef = get_force_cl(iaoa, _vec_sl=self.geom_data['vec_sl'][isample], 
                                    _veclen=self.geom_data['veclen'][isample],
                                    _area=self.geom_data['area'][isample],
                                    vel=irec[2:4, :, :], T=irec[1, :, :], P=irec[0, :, :], j0=15, j1=316, paras=WORKCOD)
                #! HERE we use relative error of lift and drag coefficients
                #! furture is need modified to absolute error
                aero_loss += torch.sum(abs(n_coef - self.new_data['coef'][isample][icode]) / self.new_data['coef'][isample][icode])
        
        # AoAs = get_aoa(recons[:, 2:4, :, :])
        # print(AoAs)
        
        indx_loss /= mu.size()[0]
        aero_loss /= mu.size()[0]

        # print(indx_loss, aero_loss, origin_loss)

        origin_loss['loss'] += indx_loss * indx_weight + aero_loss * aero_weight
        origin_loss['indx'] = indx_loss
        origin_loss['aero'] = aero_loss

        # print(origin_loss)

        return origin_loss

    def cal_avg_latent(self, pivot=-1):

        if pivot < 0:
            self.series_data['series_avg_latent'] = torch.mean(self.series_data['series_latent'], dim=1)
        
        else:
            self.series_data['series_avg_latent'] = self.series_data['series_latent'][:, pivot, :]

    def sample(self, num_samples: int, fmode: str = '', cmode: str = '', 
               indx: int = None, code: np.array = None, mu: Tensor = None, log_var: Tensor = None) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.

        param:
        ===
        - `num_samples`:    number of samples to generate
        - `fmode`:          how to generate latent for foil `zf`
            - `db`:         read from database `indx` (only for Frame A)
            - `dbr`:        read mu and log_var from database `indx` and sample (only for Frame A)
            - `i`:          input in keyword `mu` 
            - `id`:         input in keyword `mu` and `log_var` and sample from this distribution
            - `imu`:        input in keyword `mu` and sample with a Gauss (mu, 1) (for `ved`)
            - `sn`:         sample from a standard normal distribution
        - `cmode`:          how to generate latent for code `zc`
            - `no`:         latent for code is already generated in fmode (for `im`)
            - `i`:          input in keyword `code`
            - `ir`:         input in keyword `code` and sample with a Gauss (code, 1)

        return: 
        ===
        (Tensor)
        """
        if self.cm in ['ed']:
            fmode = 'i'
            cmode = 'i'
        elif self.cm in ['ved', 'ved1']:
            fmode = 'id'
            cmode = 'i'

        if fmode == 'db':
            zf = self.series_data['series_avg_latent'][indx][0].unsqueeze(0).repeat(num_samples, 1)
        elif fmode == 'i':
            zf = mu.repeat(num_samples, 1)
        
        else:
            if fmode == 'dbr':
                fmu = self.series_data['series_avg_latent'][indx][0].unsqueeze(0).repeat(num_samples, 1)
                flog_var = self.series_data['series_avg_latent'][indx][1].unsqueeze(0).repeat(num_samples, 1)
            elif fmode == 'id':
                fmu = mu.repeat(num_samples, 1)
                flog_var = log_var.repeat(num_samples, 1)
            elif fmode == 'sn':
                #! should be log_var, not sigma; so default (normal distribution) should be 0
                fmu = torch.zeros((num_samples, self.latent_dim - self.code_dim))
                flog_var = torch.zeros((num_samples, self.latent_dim - self.code_dim))
            
            zf = self.reparameterize(fmu, flog_var)

        if cmode == 'no':
            samples = self.decode(zf)
        else:
            tcode = torch.Tensor(code).to(self.device).unsqueeze(0).repeat(num_samples, 1)
            if cmode == 'i':
                zc = tcode
            elif cmode == 'ir':
                zc = self.reparameterize(tcode, torch.zeros_like(tcode, device=self.device))
            z = torch.cat([self.fc_code(zc), zf], dim=1)
            samples = self.decode(z)

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.forward(x, **kwargs)

        return result

    def get_test_latent(self, data, nob, pivot=-1, verbose=False):
        # data: fldata[nob]['flowfields']
        if verbose:
            print("{}  Old: {}".format(nob, self.new_data['series_avg_latent'][nob]))

        if pivot > 0:
            result0 = self.generate(data[pivot].unsqueeze(0).to(self.device), code=self.new_data['aoa'][nob][pivot].unsqueeze(0))
            self.new_data['series_avg_latent'][nob, 0] = result0[2][0][1:].detach()
            self.new_data['series_avg_latent'][nob, 1] = result0[3][0][1:].detach()
        
        else:
            for idx in range(len(self.new_data['aoa'][nob])):

                result0 = self.generate(data[idx].unsqueeze(0).to(self.device), code=self.new_data['aoa'][nob][idx].unsqueeze(0))
                self.new_data['series_latent'][nob, idx, 0] = result0[2][0][1:].detach()
                self.new_data['series_latent'][nob, idx, 1] = result0[3][0][1:].detach()
            
            self.new_data['series_avg_latent'][nob] = torch.mean(self.new_data['series_latent'][nob], dim=0)

    def load(self, folder, name=None, fro='cp'):
        if fro == 'd':
            saved_model = name
            self.load_state_dict(saved_model, strict=False)
        elif fro == 'c':
            path = folder + '/' + name
            saved_model = torch.load(path, map_location=self.device)['model_state_dict']
            self.load_state_dict(saved_model, strict=False)
        elif fro == 'cp':
            path = folder + '/checkpoint_epoch_' + str(299)
            save_dict = torch.load(path, map_location=self.device)
            self.load_state_dict(save_dict['model_state_dict'], strict=False)
            self.series_data = save_dict['series_data']
            self.geom_data = save_dict['geom_data']
        else:
            raise AttributeError

class frameVAE(EncoderDecoder):

    def __init__(self, latent_dim: int, encoder: Encoder, decoder: Decoder, code_mode: str, dataset_size: Tuple[int, int] = None, decoder_input_layer: int = 0, code_dim: int = 1, code_layer=[], device='cuda:0', **kwargs) -> None:
        super().__init__(latent_dim, encoder, decoder, code_mode, dataset_size, decoder_input_layer, code_dim, code_layer, device, **kwargs)
        print('Warning: the class frameVAE is renamed to EncoderDecoder')

class BranchEncoderDecoder(EncoderDecoder):
    
    def decode(self, z: Tensor, real_mesh: Tensor = None) -> Tensor:
        # if self.split_decoder:
        results = []
        for decoder, decoder_input in zip(self.decoders, self.decoder_inputs):
            result = decoder_input(z)
            result = decoder(result)
            results.append(result)
        # print(result.size())
        cat_results = torch.cat(results, dim=1)
        # print(cat_results.size())
        return cat_results

class Unet(EncoderDecoder):

    def __init__(self, latent_dim: int, encoder: Encoder, decoder: Decoder, code_mode: str, fldata: ConditionDataset = None, decoder_input_layer: int = 0, code_dim: int = 1, code_layer=[], device='cuda:0', **kwargs) -> None:
        super().__init__(latent_dim, encoder, decoder, code_mode, fldata, decoder_input_layer, code_dim, code_layer, device, **kwargs)
        if isinstance(decoder, Decoder):
            if not (self.encoder.is_unet and self.decoder.is_unet):
                raise Exception('Encoder or Decoder does not support U-Net')
        elif isinstance(decoder, list):
            for decoder in self.decoders:
                if not (self.encoder.is_unet and decoder.is_unet):
                    raise Exception('Encoder or Decoder does not support U-Net')

    def repeat_feature_maps(self, n):
        for idx in range(len(self.encoder.feature_maps)):
            self.encoder.feature_maps[idx] = self.encoder.feature_maps[idx].repeat(n, 1, 1)
        # print( self.encoder.feature_maps[0].size())

    def decode(self, z: Tensor, real_mesh: Tensor = None) -> Tensor:
        
        result = self.decoder_input(z)
        result = self.decoder(result, encoder_feature_map=self.encoder.feature_maps)

        return result

class BranchUnet(Unet):

    def decode(self, z: Tensor, real_mesh: Tensor = None) -> Tensor:
        # if self.split_decoder:
        results = []
        for decoder, decoder_input in zip(self.decoders, self.decoder_inputs):
            result = decoder_input(z)
            result = decoder(result, encoder_feature_map=self.encoder.feature_maps)
            results.append(result)
        # print(result.size())
        cat_results = torch.cat(results, dim=1)
        # print(cat_results.size())
        return cat_results

class DecoderModel(AutoEncoder):

    def __init__(self, input_channels: int, decoder: Decoder, device: str, decoder_input_layer: float = 1.5) -> None:
        super().__init__(latent_dim=input_channels, decoder=decoder, decoder_input_layer=decoder_input_layer, device=device)

    def forward(self, input: Tensor):

        return self.decode(input)

class BranchDecoderModel(BranchEncoderDecoder, nn.Module):

    def __init__(self, input_channels: int, decoders: List[Decoder], device: str, decoder_input_layer: float = 1.5) -> None:
        nn.Module.__init__(self)
        self.device = device
        _decoder_inputs = []
        for decoder in decoders:
            decoder_input = _decoder_input(typ=decoder_input_layer, ld=input_channels, lfd=decoder.last_flat_size)
            _decoder_inputs.append(decoder_input)
        self.decoders = nn.ModuleList(decoders)
        self.decoder_inputs = nn.ModuleList(_decoder_inputs)

    def forward(self, input):

        return self.decode(input)

class EncoderModel(AutoEncoder):
    
    def __init__(self, output_channels: int, encoder: Encoder, device: str) -> None:
        super().__init__(latent_dim=output_channels, encoder=encoder, device=device)

    def forward(self, input: Tensor) -> Tensor:

        return self.fc_mu(self.encoder(input))

class EncoderDecoderLSTM(nn.Module):
    
    def __init__(self, lstm, encoder, decoder, nt) -> None:
        super().__init__()

        self.lstm  = lstm
        self.encoder = encoder
        self.decoder = decoder
        self.device = 'cuda:0'
        self.nt = nt

    def forward(self, inputs):

        nb = inputs.size()[0]

        if inputs.ndim > 2:
            # if input has nt dimension, flatten nt and input to encoder
            inputs = torch.reshape(inputs, (-1, *inputs.size()[2:]))
            lstm_input = self.encoder(inputs)
            lstm_input = torch.reshape(lstm_input, (nb, self.nt, *lstm_input.size()[1:]))
        else:
            # if not, duplicate the only input to nt
            lstm_input = self.encoder(inputs)
            lstm_input = torch.unsqueeze(lstm_input, dim=1)
            lstm_input = torch.repeat_interleave(lstm_input, self.nt, dim=1)
        
        lstm_output = self.lstm(lstm_input)[0]

        decoder_input = torch.reshape(lstm_output, (-1, *lstm_output.size()[2:]))
        decoder_output = self.decoder(decoder_input)
        decoder_output = torch.reshape(decoder_output, (nb, self.nt, *decoder_output.size()[1:]))

        if decoder_output.ndim > 3:
            # if the output has a channel dimension, then swap the channel with i_t dimension
            decoder_output = torch.transpose(decoder_output, 2, 1)

        return decoder_output

def smoothness(field: Tensor, mesh: Tensor = None, offset: int = 2, field_size: Tuple = None) -> Tensor:
    # smooth = 0.0
    # if field_size is None:
    #     field_size = field.size()
    f_se = field[:, :, offset: , offset:]
    f_ne = field[:, :, : -offset, offset:]
    f_nw = field[:, :, : -offset, : -offset]
    f_sw = field[:, :, offset: , : -offset]

    m_se = mesh[:, :, offset: , offset:]
    m_ne = mesh[:, :, : -offset, offset:]
    m_nw = mesh[:, :, : -offset, : -offset]
    m_sw = mesh[:, :, offset: , : -offset]

    dis_se_nw = torch.unsqueeze(torch.sum((m_se - m_nw)**2, dim=1), 1)
    dis_ne_sw = torch.unsqueeze(torch.sum((m_ne - m_sw)**2, dim=1), 1)

    # stupid
    size = f_se.size(0) * f_se.size(1) * f_se.size(2) * f_se.size(3) 

    # smt = F.mse_loss(f_ne, f_sw) + F.mse_loss(f_se, f_nw)
    smt = torch.sum((f_se - f_nw)**2 / dis_se_nw) *2 / size + torch.sum((f_ne - f_sw)**2 / dis_ne_sw) *2 / size
    
    smt /= 5e4

    return smt

def NSres(field: Tensor, mesh: Tensor = None, w_mom: float = 0.1, dev=None) -> Tensor:
    hoz_distance = mesh[:, :, 1:, :] - mesh[:, :, :-1, :]  #   B * 2 * (H-1) * W  //   2 -> x, y
    ver_distance = mesh[:, :, :, 1:] - mesh[:, :, :, :-1]  #   B * 2 * H * (W-1)  //   2 -> x, y

    rotate = torch.Tensor([[0.0, -1.0], [1.0, 0.0]]).to(dev)
    eigenp = torch.Tensor([[1.0,  0.0], [0.0, 1.0]]).to(dev)

    hoz_vec = torch.einsum('aj,ijkl->iakl', rotate, hoz_distance)
    ver_vec = torch.einsum('aj,ijkl->iakl', rotate, ver_distance)

    # hoz_vec = hoz_distance
    # ver_vec = ver_distance

    # print(hoz_vec.size(), (-hoz_distance[:, 1, :, :]).size())

    hoz_ave_uv = 0.5 * (field[:, 2:, 1:, :] + field[:, 2:, :-1, :])  #   B * 2 * (H-1) * W  //   2 -> u, v
    ver_ave_uv = 0.5 * (field[:, 2:, :, 1:] + field[:, 2:, :, :-1])  #   B * 2 * H * (W-1)  //   2 -> u, v

    hoz_p_ave = 0.5 * (field[:, 0, 1:, :] + field[:, 0, :-1, :])
    hoz_t_ave = 0.5 * (field[:, 1, 1:, :] + field[:, 1, :-1, :])
    ver_p_ave = 0.5 * (field[:, 0, :, 1:] + field[:, 0, :, :-1])
    ver_t_ave = 0.5 * (field[:, 1, :, 1:] + field[:, 1, :, :-1])

    hoz_ave_rho = hoz_p_ave / hoz_t_ave  ## p / T
    ver_ave_rho = ver_p_ave / ver_t_ave  ## p / T

    hoz_face_flux = hoz_ave_rho * torch.einsum('ijkl,ijkl->ikl', hoz_vec, hoz_ave_uv)
    ver_face_flux = ver_ave_rho * torch.einsum('ijkl,ijkl->ikl', ver_vec, ver_ave_uv)

    hoz_flux =  hoz_face_flux[:, :, 1:] - hoz_face_flux[:, :, :-1]
    ver_flux = -ver_face_flux[:, 1:, :] + ver_face_flux[:, :-1, :]  #   B * (H-1) * (W-1)

    smt_m = torch.abs(torch.einsum('ikl->i', (hoz_flux + ver_flux)))
    smt = smt_m

    if w_mom > 0:

        # ((ux2 + p, uxuy), (uxuy, uy2 + p))
        hoz_ave_mom = torch.einsum('aibc,ajbc->aijbc', hoz_ave_uv, hoz_ave_uv) + torch.einsum('ij,abc->aijbc', eigenp, hoz_p_ave)
        ver_ave_mom = torch.einsum('aibc,ajbc->aijbc', ver_ave_uv, ver_ave_uv) + torch.einsum('ij,abc->aijbc', eigenp, ver_p_ave)

        hoz_face_mom_flux = torch.einsum('abc,ajbc,aijbc->aibc', hoz_ave_rho, hoz_vec, hoz_ave_mom)
        ver_face_mom_flux = torch.einsum('abc,ajbc,aijbc->aibc', ver_ave_rho, ver_vec, ver_ave_mom)

        hoz_mom_flux =  hoz_face_mom_flux[:, :, :, 1:] - hoz_face_mom_flux[:, :, :, :-1]
        ver_mom_flux = -ver_face_mom_flux[:, :, 1:, :] + ver_face_mom_flux[:, :, :-1, :]

        smt_mom = torch.abs(torch.einsum('abcd->ab', (hoz_mom_flux + ver_mom_flux)))
        smt += w_mom * torch.einsum('ab->a', smt_mom)

    return smt

def roughness_penalty(x01: Tensor, x0p: Tensor, ys: Tensor, dev=None) -> Tensor:
    '''
    The roughness penalty of a curve based on area
    
    >>> rs = roughness_penalty(x, ys)
    
    ### Inputs:
    ```text
    x01: Tensor  [n_grid-2], x01 = x[2:]-x[:n_grid-2]
    x0p: Tensor  [n_grid-2], x0p = x[1:]-x[:n_grid-1]
    ys:  Tensor  [n_sample, n_grid]
    ```
    ### Output:
    ```text
    rs: Tensor [n_sample]
    ```
    '''
    n_grid = ys.size(1)
    ss = torch.FloatTensor([(-1)**i for i in range(n_grid-2)]).to(dev)    # [n_grid-2]

    y01 = ys[:,2:]        -ys[:,1:n_grid-1]  # [n_sample, n_grid-2]
    y0p = ys[:,1:n_grid-1]-ys[:, :n_grid-2]  # [n_sample, n_grid-2]

    tmp = (y0p*x01 - y01*x0p)*0.5   # [n_sample, n_grid-2]
    rs  = tmp*ss                    # [n_sample, n_grid-2]
    rs  = rs.sum(dim=1).mean(dim=0) 

    return torch.abs(rs)

def gradient_enhance_1d(dx: Tensor, avgdx: Tensor, ys1: Tensor, ys2: Tensor) -> Tensor:

    dx = dx.unsqueeze(0).repeat(ys1.size(0), 1)
    dys1 = ys1[:, 0, 1:] - ys1[:, 0, :-1]
    dys2 = ys2[:, 0, 1:] - ys2[:, 0, :-1]

    # print(dx.size(), dys1.size(), dys2.size())

    return F.mse_loss(dys1 / dx, dys2 / dx) * avgdx