'''

the vae model classes(vanilla, conditional, etc.)
adapted from AntixK, PyTorch-VAE. https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

同一个encoder，接不同decoder，1.加上工况，直接预测cl，cd 2.不加工况，直接预测抖振攻角和升阻力

'''



import torch
from torch import nn
from torch.nn import functional as F

from flowvae.dataset import ConditionDataset

import numpy as np

from typing import List, Callable, NewType, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor

Tensor = NewType('Tensor', torch.tensor)

# from utils import get_force

from .post import get_aoa, get_vector, get_force_cl, WORKCOD, get_force_1dc

class frameVAE(nn.Module):

    def __init__(self,
                 latent_dim: int,
                 fldata: ConditionDataset,
                 encoder,
                 decoder,
                 decoder_input_layer = 0,
                 code_dim = 1,
                 code_layer = [],
                 code_mode = 'im',
                 device = 'cuda:0',
                 **kwargs) -> None:


        super(frameVAE, self).__init__()
        
        self.latent_dim = latent_dim        # the total dimension of latent variable (include code dimension)
        self.code_dim = code_dim   # the code dimension (same with which in the dataset)
        # self.in_channels = in_channels

        self.device = device

        self.paras = kwargs
        self.paras['decoder_input_layer'] = decoder_input_layer
        self.paras['code_mode'] = code_mode

        self.series_data = {}

        if fldata is not None:
            self.set_aux_data(fldata)

        # print(in_channels, latent_dim, hidden_dims, kernel_sizes, strides, paddings, max_pools)
        self.encoder = encoder
        self.decoder = decoder
        # if self.encoder.last_flat_size == self.decoder.last_flat_size and self.encoder.last_flat_size > 0:
        #     last_flat_size = self.encoder.last_flat_size
        # else:
        #     raise RuntimeError("The last flat size given by encoder and decoder is not the same:\n encoder: %d decoder: %d" 
        #                             % (self.encoder.last_flat_size, self.decoder.last_flat_size))

        ld = self.latent_dim
        cd = self.code_dim
        fd = ld - cd

        lf = self.encoder.last_flat_size
        if code_mode in ['ex']:
            self.fc_mu = nn.Sequential(
                nn.Linear(lf + cd, fd), nn.BatchNorm1d(fd), nn.LeakyReLU(),
                nn.Linear(fd, fd))
            self.fc_var = nn.Sequential(
                nn.Linear(lf + cd, fd), nn.BatchNorm1d(fd), nn.LeakyReLU(),
                nn.Linear(fd, fd))
        elif code_mode in ['ae', 'ed']:
            self.fc_mu = nn.Linear(lf, fd)
        elif code_mode in ['ved']:
            self.fc_mu = nn.Linear(lf, fd)
            self.fc_var = nn.Linear(lf, fd)    
        elif code_mode in ['semi', 'im']:
            self.fc_mu = nn.Linear(lf, ld)
            self.fc_var = nn.Linear(lf, ld)

        lf = self.decoder.last_flat_size
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
        
        if self.paras['decoder_input_layer'] == 0:
            self.decoder_input = nn.Identity()
            
        elif self.paras['decoder_input_layer'] == 1:
            self.decoder_input = nn.Linear(ld, lf)
        
        elif self.paras['decoder_input_layer'] == 2:
            self.decoder_input = nn.Sequential(
                nn.Linear(ld, ld*2), nn.BatchNorm1d(ld*2), nn.LeakyReLU(),
                nn.Linear(ld*2, lf), nn.BatchNorm1d(lf), nn.LeakyReLU())

        elif self.paras['decoder_input_layer'] == 2.5:
            self.decoder_input = nn.Sequential(
                nn.Linear(ld, ld), nn.BatchNorm1d(ld), nn.LeakyReLU(),
                nn.Linear(ld, lf), nn.BatchNorm1d(lf), nn.LeakyReLU())

        elif self.paras['decoder_input_layer'] == 3:
            self.decoder_input = nn.Sequential(
                nn.Linear(ld, ld), nn.BatchNorm1d(ld), nn.LeakyReLU(),
                nn.Linear(ld, ld*2), nn.BatchNorm1d(ld*2), nn.LeakyReLU(),
                nn.Linear(ld*2, lf), nn.BatchNorm1d(lf), nn.LeakyReLU())

        else:
            raise KeyError()
    

    def set_aux_data(self, fldata: ConditionDataset):
        n_airfoil = fldata.airfoil_num
        n_condi   = fldata.condis_num
        n_z       = self.latent_dim - self.code_dim
        self.series_data = {
            'series_latent': torch.cat(
                [torch.zeros((n_airfoil, n_condi, 1, n_z)), 
                 torch.ones((n_airfoil, n_condi, 1, n_z))], dim=2).to(self.device),
            'series_avg_latent': torch.cat(
                [torch.zeros((n_airfoil, 1, n_z)),
                 torch.ones((n_airfoil, 1, n_z))], dim=1).to(self.device)}


    def preprocess_data(self, fldata: ConditionDataset):
        
        self.geom_data = {}
        # print(' === Warning:  preprocess_data is not implemented. If auxilary geometry data is needed, please rewrite vae.preprocess_data')
        return
        '''
        produce geometry data
        - x01, x0p: the x coordinate data for smooth calculation
        '''
        from cst_modeling.foil import clustcos

        
        nn = 201
        xx = [clustcos(i, nn) for i in range(nn)]
        all_x = torch.from_numpy(np.concatenate((xx[::-1], xx[1:]), axis=0)).to(self.device).float()
        self.geom_data['xx']  = all_x
        self.geom_data['x01'] = all_x[2:] - all_x[1: -1]
        self.geom_data['x0p'] = all_x[1: -1] - all_x[:-2]

        if fldata is not None:

            geom = torch.cat((all_x.repeat(fldata.data.size(0), 1).unsqueeze(1), fldata.data[:, 0].unsqueeze(1)), dim=1)
            profile = fldata.data[:, 1]
            self.geom_data['all_clcd'] = get_force_1dc(geom, profile, fldata.cond.squeeze())
            print('all_clcd size:',  self.geom_data['all_clcd'].size())

            geom = torch.cat((all_x.repeat(fldata.refr.size(0), 1).unsqueeze(1), fldata.refr[:, 0].unsqueeze(1)), dim=1)
            profile = fldata.refr[:, 1]
            self.geom_data['ref_clcd'] = get_force_1dc(geom, profile, fldata.ref_condis.squeeze())
            print('ref_clcd size:',  self.geom_data['ref_clcd'].size())

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

        if self.paras['code_mode'] in ['ex']:
            result = torch.cat([result, kwargs['code']], dim=1)
            mu = self.fc_mu(result)
            log_var = self.fc_var(result)
        elif self.paras['code_mode'] in ['ae', 'ed']:
            mu = self.fc_mu(result)
            log_var = torch.zeros_like(mu)
        # Split the result into mu and var components of the latent Gaussian distribution
        elif self.paras['code_mode'] in ['im', 'semi', 'ved']:
            mu = self.fc_mu(result)
            # TODO here log_var's first element should be 0 for im and semi
            log_var = self.fc_var(result)

        # thr output of ae and ex is in dimension 11
        return [mu, log_var]

    def decode(self, z: Tensor, real_mesh: Tensor = None) -> Tensor:
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
        
        if self.paras['code_mode'] in ['ae', 'ed']:
            z = mu
        elif self.paras['code_mode'] in ['ex', 'semi', 'im']:
            z = self.reparameterize(mu, log_var)
        elif self.paras['code_mode'] in ['ved']:
            mu = torch.cat([kwargs['code'], mu], dim=1)
            #! this is log_var, so std_var=1 means log_var=0
            log_var = torch.cat([torch.zeros_like(kwargs['code'], device=self.device), log_var], dim=1)
            z = self.reparameterize(mu, log_var)
        
        if self.paras['code_mode'] in ['semi']:
            code_output = self.fc_code(kwargs['code'])
            # print("new, old:", kwargs['code'], z[:, 0])
            z[:, :code_output.size()[1]] = code_output
        elif self.paras['code_mode'] in ['ex', 'ae', 'ed']:
            code_output = self.fc_code(kwargs['code'])
            z = torch.cat([code_output, z], dim=1)
        elif self.paras['code_mode'] in ['im', 'ved']:
            pass
            # TODO: the mode `im` and `ved` is not consist with code layers!

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

        # print(recons.size(), ref.size(), real.size())
        # input()
        recons_loss = F.mse_loss(recons + ref, real)
        # the real reconstruction is done by setting ref = 0

        loss = recons_loss

        sm_loss = torch.FloatTensor([0.0])

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
                sm_loss = roughness_penalty(x01, x0p, recons, dev=self.device)

            else:
                raise AttributeError()
            
            # !!!!

            loss += sm_weight * sm_loss
        
        origin_loss = {'loss': loss, 'recons':recons_loss, 'smooth':sm_loss}

        origin_loss['code'] = torch.FloatTensor([0.0])
        origin_loss['indx'] = torch.FloatTensor([0.0])
        origin_loss['aero'] = torch.FloatTensor([0.0])     

        if self.paras['code_mode'] in ['ved']:
            mu = args[1][:, self.code_dim:]
            log_var = args[2][:, self.code_dim:]
            indx_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            origin_loss['loss'] += indx_loss * kwargs['indx_weight']
            origin_loss['indx'] = indx_loss
        
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

        if self.paras['code_mode'] in ['semi', 'im']:
            
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

                if self.paras['code_mode'] in ['ae']:
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

    def sample(self,
               num_samples:int,
               mode:str = 'rep', **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        if mode == 'giv':
            z = torch.cat((torch.Tensor([kwargs['code']]).to(self.device), self.series_data['series_avg_latent'][kwargs['indx']][0]))
            # print(z)
            z = z.unsqueeze(0).repeat(num_samples, 1)
            # z = torch.einsum('i,j->ij', torch.ones((num_samples,)).to(self.device), z)
        
        #! should be log_var, not sigma; so default (normal distribution) should be 0
        else:
            if mode == 'giv_rand':
                mu = self.series_data['series_avg_latent'][kwargs['indx']][0].unsqueeze(0)
                log_var = self.series_data['series_avg_latent'][kwargs['indx']][1].unsqueeze(0)
            elif mode == 'distr':
                mu = kwargs['mu']
                log_var = kwargs['log_var']
            else:
                mu = torch.zeros((num_samples, self.latent_dim))
                log_var = torch.zeros((num_samples, self.latent_dim))
            
            mu = mu.repeat(num_samples, 1)

            if log_var is None:
                zf = mu
            else:
                log_var = log_var.repeat(num_samples, 1)
                zf = self.reparameterize(mu, log_var)

            # print(zf.size())
            # print(z)
            # print(torch.Tensor(kwargs['code']).to(self.device).unsqueeze(0).size())
            # zc = 
            code_output = self.fc_code(torch.Tensor(kwargs['code']).to(self.device).unsqueeze(0).repeat(num_samples, 1))
            z = torch.cat([code_output, zf], dim=1)
            # z = torch.cat((zc, zf), dim=1)
            # print(z.size())		 
        
        # z = torch.randn(num_samples, self.latent_dim)

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