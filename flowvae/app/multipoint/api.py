'''
Oct 29 2024
Author: Yunjia Yang

Move buffet onset prediction client files from buffet working folder to here

Rearrange the methods, tested on `ropt1_test`, pass (probabilistic version leads to
some big uncertainty by itselt, quite amazing)


'''

from flowvae.vae import frameVAE, Unet
from flowvae.ml_operator import AEOperator
from flowvae.post import get_force_1d_t
from flowvae.app.buffet import Buffet, Series, Ploter
from flowvae.utils import warmup_lr, load_encoder_decoder

import matplotlib.pyplot as plt
from cfdpost.cfdresult import cfl3d
from cfdpost.section.physical import PhysicalSec
from cfdpost.utils import get_force_1d, get_moment_1d, clustcos

import numpy as np
import torch

from scipy.interpolate import PchipInterpolator as pchip

from typing import Tuple, NewType
Tensor = NewType('Tensor', torch.Tensor)


CD_NON_DIM = 15.522738  # new force data
# CD_NON_DIM = 15.84  # old force data
CF_NON_DIM = 1000.

def load_model(run, subrun, ref, decoder_idx, code_mode, code_dim, lat_dim, device='cpu') -> Tuple[frameVAE, dict]:
    date = '1025'
    # run = run
    latent_dim = lat_dim
    # ipt_ch = ipt_ch
    ml_dir = 'D:\\DeepLearning\\202210LRZdata'
    flags = {}

    flags['is_ref'] = ref

    file_name = '%s_%s_Run%d' % (date, str(run), subrun)

    encoder, decoder, VAE_class = load_encoder_decoder(decoder_idx)

    vae_model = VAE_class(latent_dim=latent_dim,
                        encoder=encoder,
                        decoder=decoder,
                        code_dim=code_dim,
                        code_layer=[],
                        decoder_input_layer=1,  #! original 0 for dense connection
                        code_mode=code_mode,
                        device=device)

    op = AEOperator(file_name, vae_model, None, shuffle=True, ref=ref, batch_size=16, init_lr=0.0001, output_folder=ml_dir+'\\save',
                                                         recover_split=file_name)
    op.set_scheduler('LambdaLR', lr_lambda=warmup_lr)
    op.load_checkpoint(299)
    op.model.eval()

    flags['op'] = op

    return vae_model, flags

def add_extra_channel(origin_ref_data, device, extra_value = None, isextrachannel = False):
    '''
    return reference data with added extra channels
    
    paras:
    ---
    `origin_ref_data`   shape -> (C0, N)
    
    returns:
    ---
    torch.Tensor, shape -> (1, C, N), C = C0 + Cext
    
    '''
    # afoil['ref'], fldata.get_index_info(i_f, 0, 5)
    
    if not isextrachannel:
        ref_data = torch.from_numpy(origin_ref_data).float().unsqueeze(0).to(device)
    else:
        #* for run no.57 - 59, add Mach channel (at the first)
        ref_data = torch.from_numpy(origin_ref_data).float()
        ref_data = torch.concatenate([extra_value * torch.ones((1, ref_data.size(1))), ref_data], dim=0)
        ref_data = ref_data.unsqueeze(0).to(device)
        
    return ref_data

def predict_field(vae_model, flags, ref_data, aoa_real, aoa_ref, mu1, log_var1, n_samples):
    '''
    call the decoder and do the residual add
    
    paras:
    - `vae_model` & `flags`:    obtained from `load_model`
    - `ref_data`:               the reference field to be added to the results (if `flag['is_ref]` is `True)
        shape -> (1, 2 -> Cp, Cf, N)
    - `aoa_real`:
    
    return:
    ---
    - `recons`: torch.Tensor, shape = n_sample, n_channel, H, W
    '''
    
    aoa_delt = aoa_real - aoa_ref * int(flags['is_ref'])
    if vae_model.cm == 'ed':
        n_samples = 1
        samp1 = vae_model.sample(num_samples=1, code=[aoa_delt], mu=mu1)#.squeeze() #ed
    elif vae_model.cm in ['ved1', 'ved']:
        samp1 = vae_model.sample(num_samples=n_samples, code=[aoa_delt], mu=mu1, log_var=log_var1)#.squeeze() #ved
    # print(samp1.shape, ref_data.shape, ref_data.repeat([n_samples] + [1 for _ in range(ref_data.dim()-1)]).shape)
    if flags['is_ref']:
        samp1 = samp1 + ref_data.repeat([n_samples] + [1 for _ in range(ref_data.dim()-1)])
    return samp1
    
def get_features_series(recons: Tensor, geom: Tensor, aoa_real, fF, n_samples, is_uq, nondim_cf = 1.):
    '''
    paras:
    ---
    
    - `is_uq` -> 
        - if `True`, 
        - if `False`,  the `recons` will first be averaged across samples (dim=0)
        
    returns:
    ---
    `dict`
    - `auxs`:   [`x1`, `mUy`, `cm`]
    - `recons`: numpy.ndarray (2, N)
    - `cl`:     lift coefficient (`float` if `is_uq` is `False`, `Tensor` of (B,) if `is_uq` is `True`)
    - `cd`:     drag coefficient 
    '''
    _geom = geom.cpu().detach().numpy()
    _recons = torch.mean(recons, dim=0).squeeze().cpu().detach().numpy()
    _cp = _recons[0]
    _cf = _recons[1] / nondim_cf
    
    fF.setlimdata(_geom[0], _geom[1], _cp, _cf)   # x, y, Cp, Cf
    # fF.locate_basic()
    fF.locate_sep()
    # fF.locate_geo()
    i_1 = fF.locate_shock(info=False)
    x1  = fF.getValue('1','X')
    mUy = fF.getValue('mUy','dudy')
    cm  = get_moment_1d(_geom.transpose(), _cp, _cf)
    # print(mUy)
    # plt.plot(_geom[0], _cf)
    # plt.show()
    
    # print(aoa_real, clcd[1], clcd[0], x1, mUy)
    returns = {'auxs':  [x1, mUy, cm], 'recons': _recons}

    if not is_uq:
        # only include Cp to cal. force
        # print(recons.shape, aoa_real.shape)
        # clcd = get_force_1d(geom, torch.mean(recons, dim=0)[0], aoa_real).cpu().detach().numpy()
        clcd = get_force_1d(_geom.transpose(1, 0), aoa_real, _cp, _cf)
        
        returns['cl'] = clcd[1]
        returns['cd'] = clcd[0]
    
    else:
        # print(recons.shape, aoa_real)

        clcd = get_force_1d_t(geom.permute(1, 0).unsqueeze(0).repeat(n_samples, 1, 1), aoa=float(aoa_real) * torch.ones(n_samples, ),
                                cp=recons[:, 0], cf=recons[:, 1])# .cpu().detach().numpy()

        returns['cl'] = clcd[:, 1]
        returns['cd'] = clcd[:, 0]

    return returns

def get_features_series0(recons: Tensor, geom: Tensor, aoa_real, fF, n_samples, is_uq, nondim_cf = 1.):
    
    returns = get_features_series(recons, geom, aoa_real, fF, n_samples, is_uq, nondim_cf)
    
    return returns['cl'], returns['cd'], returns['auxs'][0], returns['auxs'][1]

def predict_series(vae_model: frameVAE, flags: dict, ys: np.array, condition: dict = None, ocs: np.ndarray = None, ref_oc: float = 0.0, aoas_real: np.ndarray = None,
                    n_samples : int = 1, device: str = 'cpu', isforce: bool = True, ref_force: Tensor = None, isextrachannel: float = False):
    '''
    
    paras:
    ---
    - `condition` -> dict (`ma` for fF input)
    -  
    
    - `ref_force` -> (cd, cl) or (cd, aoa) or None (integrate from ref_field `ys`) [dimensional]
    - `isextrachannel` -> (float) if given, then added to the first channel before `ys`
    
    returns:
    ---
    dict
    - `recons` (numpy.ndarray) shape = n_c x n_channel (Cp, Cf) x N [dimensional]
    - `clcds`: (torch.Tensor) shape = n_c x n_samlpes x 2 (CL, CD) [dimensional] [dimensional]
    - `seri_value01`: (numpy.ndarray) shape = (n_c x 3) X1, Cf, cm
    - `mu1`
    - `log_var1`
    - `ref_data`
    - `geom`
    
    '''
    
    nn = 201
    xx = [clustcos(i, nn) for i in range(nn)]
    all_x = torch.from_numpy(np.concatenate((xx[::-1], xx[1:]), axis=0)).to(device).float()
    geom = torch.cat((all_x.unsqueeze(0), torch.from_numpy(ys[0]).to(device).float().unsqueeze(0)), dim=0)
    
    n_c = len(ocs)
    clcds = torch.zeros((n_c, n_samples, 2))
    ref_data = add_extra_channel(ys, device, isextrachannel, isextrachannel is not None)

    if isforce:
        if ref_force is None:
            ref_force = get_force_1d(geom, ref_data[0, 1], ref_oc)
        else:
            ref_force = torch.from_numpy(np.array(ref_force)).to(device).float()
        
        ref_force[0] *= CD_NON_DIM
        seri_recons, seri_value01 = None, None
    else:
        seri_recons, seri_value01 = [], [] # recon: x1, cf, cm

    mu1, log_var1 = vae_model.encode(ref_data)
    if vae_model.encoder.is_unet:
        vae_model.repeat_feature_maps(n_samples)

    for i_c in range(n_c):
        
        if isforce:
            recons_force = predict_field(vae_model, flags, ref_force.unsqueeze(0), ocs[i_c], ref_oc, mu1, log_var1, n_samples)
            clcds[i_c] = recons_force
            clcds[i_c, :, 0] /= CD_NON_DIM
        else:
            #* get results from profile-predicting model
            recons = predict_field(vae_model, flags, ref_data[:, 1+int(isextrachannel):], ocs[i_c], ref_oc, mu1, log_var1, n_samples)
            
            # dimensionalize
            recons[:, 1] = recons[:, 1] / CF_NON_DIM
            recons[:, 1, :int(recons.shape[1]/2.)] *= -1
            
            # from matplotlib import pyplot as plt
            # plt.plot(all_x.detach().cpu().numpy(), recons[0, 0].detach().cpu().numpy())
            # plt.show()
            
            # get integration
            if aoas_real is None:
                aoas_real = ocs

            fF = PhysicalSec(Minf=condition['ma'], AoA=aoas_real[i_c], Re=20e6)
            returns = get_features_series(recons, geom, aoas_real[i_c], fF, n_samples, is_uq=True)
            clcds[i_c, :, 1], clcds[i_c, :, 0] = returns['cl'], returns['cd']
            seri_recons.append(returns['recons'])
            seri_value01.append(returns['auxs'])
    
    if not isforce:
        seri_value01 = np.array(seri_value01)
        seri_recons  = np.array(seri_recons)

    return seri_recons, clcds, seri_value01, mu1, log_var1, ref_data, geom

def predict_given_cl(vae_model: frameVAE, flags: dict, ys: np.array, condition: dict, cltarg: float, ref_aoa: float = 0.0,
                    n_samples : int = 1, device: str = 'cpu', isforce: bool = True, isextrachannel: float = None, aoa_range=(1.0, 5.0)) -> dict:
    
    '''
    iterate angle of attack from 1.0 to 5.0 to find the critical AoA that achieve target lift coefficient
    
    paras:
    ---
    - 
    
    return:
    ---
    
    dict:
    - `aoa` -> critical AoA for target Cl
    - `result` -> result from `predict_series`
        - `seri_recons` (numpy.ndarray) shape = n_c x n_channel (Cp, Cf) x N
        - `clcds`:  shape = n_c x n_samlpes x 2 (CL, CD)
        - `seri_value01`:  shape = (3 x n_c) X1, Cf, cm
        - `mu1`
        - `log_var1`
    '''
    aoa0 = aoa_range[0]
    aoa1 = aoa_range[1]
    daoa = (aoa1 - aoa0) / 10.
    while daoa > 0.0001:
        aoas = np.linspace(aoa0, aoa1, 11)
        clss = np.mean(predict_series(vae_model, flags, ys, condition, ocs=[cltarg], ref_oc=0., 
                    n_samples=n_samples, device=device, isforce=isforce, isextrachannel=isextrachannel)[1][:, :, 1].detach().cpu().numpy(), axis=1) - cltarg
        # print(clss + cltarg)
        for i in range(len(clss)-1):
            if clss[i] < 0. and clss[i+1] > 0.:
                aoa0 = aoas[i]
                aoa1 = aoas[i+1]
                daoa = (aoa1 - aoa0) / 10.
                break
        else:
            raise RuntimeError('not found aoa in 0.0 ~ 5.0')

    return {'aoa':  0.5*(aoa0+aoa1),
            'result': predict_series(vae_model, flags, ys, condition, ocs=np.array([0.5*(aoa0+aoa1)]), ref_oc=0., 
                    n_samples=n_samples, device=device, isforce=isforce, isextrachannel=isextrachannel)[:5]}

def predict_given_cl1(vae_model: frameVAE, flags: dict, ys: np.array, condition: dict, cltarg: float, 
                    n_samples : int = 1, device: str = 'cpu', isforce: bool = True, isextrachannel: bool = False, aoa_range=(1.0, 5.0)) -> dict:
    
    '''
    iterate angle of attack from 1.0 to 5.0 to find the critical AoA that achieve target lift coefficient
    (cl model)
    
    paras:
    ---
    - 
    
    return:
    ---
    
    dict:
    - `aoa` -> critical AoA for target Cl
    - `result` -> result from `predict_series`
        - `seri_recons` (numpy.ndarray) shape = n_c x n_channel (Cp, Cf) x N
        - `clcds`:  shape = n_c x n_samlpes x 2 (CL, CD)
        - `seri_value01`:  shape = (3 x n_c) X1, Cf, cm
        - `mu1`
        - `log_var1`
    '''

    recons, clss, seri, mu, log, _, _ = predict_series(vae_model, flags, ys, condition, ocs=[cltarg], ref_ocs=0., aoas_real=[0.],
                n_samples=n_samples, device=device, isforce=isforce, isextrachannel=isextrachannel)
    
    cd_fake, cl_fake = clss[:, :, 0], clss[:, :, 1]
    print(cd_fake, cl_fake)
    force = (cl_fake**2 + cd_fake**2)**0.5
    clss[:, :, 0] = (cl_fake**2 + cd_fake**2 - cltarg**2)**0.5
    clss[:, :, 1] = cltarg
    print(clss[:, :, 0], clss[:, :, 1])
    
    aoa = torch.arccos(cl_fake / force) - torch.arccos(cltarg / force)

    return {'aoa':  0.5*(aoa),
            'result': [recons, clss, seri, mu, log]}

def predict_buffet_force(vae_model: frameVAE, flags: dict, ys: np.array, condition: dict, aoas: np.ndarray = None,
                    n_samples : int = 1, is_uq: int = 0, device: str = 'cpu', isforce: bool = True, isplot: bool = False):
    '''
    `aoas` if not specificed, be a series for buffet computation
    
    '''

    ref_aoa = condition['aoa_ref']
    aoas = np.concatenate([np.arange(-2., ref_aoa, 0.25), np.arange(ref_aoa, 4.5, 0.1)]) # 2023.8.8 modify to ref as aoa_ref
    
    _, clcds, seri_value01, mu1, log_var1, ref_data, geom = predict_series(vae_model, flags, ys, condition, aoas, ref_aoa,
                    n_samples, is_uq, device, isforce)
    
    if isforce:
        buf = Buffet(method='lift_curve_break', logtype=0, lslmtd=0, lsumtd='cruise', srst='none', intp='pchip', sep_check=False)
    else:
        buf = Buffet(method='lift_curve_break', logtype=0, lslmtd=0, lsumtd='cruise', srst='sep', intp='pchip', sep_check=True)
    
    if isforce:
        ref_force = get_force_1d_t(geom, ref_data[0, 1], ref_aoa)
        cl_cruise = ref_force[1].item()
        ref_force[0] *= CD_NON_DIM
    else:
        _, cl_cruise = get_force_1d_t(geom, ref_data[0, 1], ref_aoa).cpu().detach().numpy()

    if isplot:
        plt.plot(aoas, clcds[:, 0, 1], '-o')
        plot_obj = Ploter('1')
    else:
        plot_obj = None
    # print(ref_aoa, cl_cruise)
    # plt.show()

    if is_uq <= 0:
        if isforce:
            seri = Series(datas={'AoA': aoas, 'Cl': clcds[:, 0, 1]})
        else:
            seri = Series(datas={'AoA': aoas, 'Cl': clcds[:, 0, 1], 'X1': seri_value01[0], 'Cf': seri_value01[1]})
            
        # avg_bufs = buf.buffet_onset(seri, cl_c=cl_ref, p=Ploter('1'))
        avg_bufs = buf.buffet_onset(seri, cl_c=cl_cruise, p=plot_obj)
        sig_bufs = None
    else:
        # _bufs = np.zeros((n_samples, 2))
        # for idx in range(n_samples):
        #     _bufs[idx][0],  _bufs[idx][1] = get_buffet(aoas, clcds[:, idx, 1], clcds[:, idx, 0], cl_c=cl_ref, plot=plot)
        
        # avg_bufs = np.mean(_bufs, axis=0)
        # sig_bufs = np.std(_bufs, axis=0)
        buffet_values = buffet_uq(aoas, clcds.transpose((1,0,2)), cl_cruise, buf, is_uq, n_samples1=None, flow_features=seri_value01)
        # dim = 0:  avg, sig
        # dim = 1:  lb, mid, ub
        avg_bufs = buffet_values[0]
        sig_bufs = buffet_values[1]

        # in ropt1_vae2, the calculation of lower and upper bound is conduct here
        # in opts following ropt1_vae3, it is moved to in runfoils of opt files
        # sig_bufs = buffet_values[1, 1] + 0.5 * (buffet_values[0, 2] - buffet_values[0, 0])
        # lowbound, uppbound = stats.t.interval(confidence=0.95, df=15, scale=sig_bufs)

    return avg_bufs, sig_bufs, (ref_aoa, cl_cruise), mu1.squeeze().detach(), log_var1.squeeze().detach()

def buffet_uq(aoas1, recons_values, cl_cruise, buf, is_uq, n_samples1 = None, flow_features = None):
    '''
    
    `recons_values` `np.array` shape = N_sample x N_condition x N_quantity (Cd, Cl)
    
    '''
    
    _buf = []
    recons_values_mean = np.mean(recons_values, axis=0)
    recons_values_std  = np.std(recons_values, axis=0)
    # print(recons_values.shape)
    if flow_features is None:
        seri0  = Series(datas={'AoA': aoas1, 'Cl': recons_values_mean[:, 1]})
    else:
        seri0 = Series(datas={'AoA': aoas1, 'Cl': recons_values_mean[:, 1], 'X1': flow_features[0], 'Cf': flow_features[1]})
    if is_uq == 2:

        xxs = np.random.standard_normal(n_samples1)
        recons_values = recons_values_mean + np.einsum('i,jb->ijb', xxs, recons_values_std)
    if is_uq in [1, 2]:
        for i_sample in range(recons_values.shape[0]):

            seri   = Series(datas={'AoA': aoas1, 'Cl': recons_values[i_sample, :, 1]})
            # seri   = Series(datas={'AoA': aoas1, 'Cl': recons_values[i_sample, :, 1], 'X1': flow_features[0], 'Cf': flow_features[1]})

            # plt.plot(seri.AoA, seri.Cl)
            a_buf = buf.buffet_onset(seri, cl_c=cl_cruise)
            if (a_buf > 0.).all():
            # if a_buf.all() > 0.:
            
                # print(a_buf)
                _buf.append(a_buf)
        
        # plt.show()
        bufs = buf.buffet_onset(seri0, cl_c=cl_cruise)
        if not (bufs > 0.).all() and len(_buf) > 1: # add for ropt_t0.14, since the force predict has large error
                                                    # and buffet onset always not obtained with the centre series
            bufs = np.mean(np.array(_buf), axis=0)
        
        if len(_buf) > 1:
            std_bufs = np.std(np.array(_buf), axis=0)
        else:
            std_bufs = np.array([[3, 1], [3, 1], [3, 1]])
            

    # buffet_loss[i_f, 1] = np.mean(np.array(_buf), axis=0) 
    # if len(_buf) < recons_values.shape[0]:
    #     print(i_f, len(_buf))

    # if buffet_loss[subrun, i_f, 1, 0, 1, 0] < 0 or buffet_loss[subrun, i_f, 1, 1, 1, 0] < 0:
    # print(i_f, buffet_loss[subrun, i_f, :, 1, 0])
    # input()
    # plt.plot(_aoas, _cl_r, '--o', c='k')
    # plt.plot(_aoas, _cl1, '--o', c='r')
    # plt.show()
    if is_uq == 3:
        # use extra cls to evaluate lower and upper bound
        bufs = buf.buffet_onset(seri0, cl_c=cl_cruise, extra_cl=recons_values[:, :, 1])
        std_bufs = np.zeros_like(bufs)
    print(bufs.shape, std_bufs.shape)
    return np.array([bufs, std_bufs])

'''
def predict_buffet(vae_model: frameVAE, flags: dict, ys: np.array, condition: dict, sep_check: bool,
                    n_samples : int = 1, is_uq: bool = False, device: str = 'cpu', plot: bool = False):
    
    nn = 201
    xx = [clustcos(i, nn) for i in range(nn)]
    all_x = torch.from_numpy(np.concatenate((xx[::-1], xx[1:]), axis=0)).to(device).float()
    geom = torch.cat((all_x.unsqueeze(0), torch.from_numpy(ys[0]).to(device).float().unsqueeze(0)), dim=0)
    buf = Buffet(method='lift_curve_break', logtype=0, lslmtd=0, lsumtd='cruise', srst='sep', intp='pchip', sep_check=sep_check)

    # aoas = np.concatenate([np.arange(-2, 1.5, 0.25), np.arange(1.5, 5.0, 0.04)])
    ref_aoa = condition['aoa_ref']
    # aoas = np.concatenate([np.arange(ref_aoa - 2, ref_aoa, 0.25), np.arange(ref_aoa, ref_aoa + 4, 0.1)]) # 2023.5.20 modify to ref as aoa_ref
    aoas = np.concatenate([np.arange(-2., ref_aoa, 0.25), np.arange(ref_aoa, 4.5, 0.1)]) # 2023.8.8 modify to ref as aoa_ref
   
    n_c = len(aoas)
    
    if not is_uq:
        clss = np.zeros((n_c,))
        cdss = np.zeros((n_c,))
    else:
        clcds = np.zeros((n_c, n_samples, 2))

    x1ss = np.zeros((n_c,))
    cfss = np.zeros((n_c,))

    ref_data = torch.from_numpy(ys).float().unsqueeze(0).to(device)

    _, cl_ref = get_force_1d(geom, ref_data[0, 1], ref_aoa).cpu().detach().numpy()
    mu1, log_var1 = vae_model.encode(ref_data)
    if vae_model.encoder.is_unet:
        vae_model.repeat_feature_maps(n_samples)

    for i_c in range(n_c):
        fF = PhysicalSec(condition['ma'], aoas[i_c], condition['re'])
        recons = predict_field(vae_model, flags, ref_data[:, 1:], aoas[i_c], ref_aoa, mu1, log_var1, n_samples)
        clss[i_c], cdss[i_c], x1ss[i_c], cfss[i_c] = get_features_series(recons, geom, aoas[i_c], fF, n_samples, is_uq)
       
    # plt.plot(aoas, cdss)
    # print(ref_aoa, cl_ref)
    # plt.show()
    seri = Series(datas={'AoA': aoas, 'Cl': clss, 'X1':x1ss, 'Cf': cfss})
    if not is_uq:
        # avg_bufs = buf.buffet_onset(seri, cl_c=cl_ref, p=Ploter('1'))
        avg_bufs = buf.buffet_onset(seri, cl_c=cl_ref)
        sig_bufs = None
    else:
        _bufs = np.zeros((n_samples, 2))
        # for idx in range(n_samples):
        #     _bufs[idx][0],  _bufs[idx][1] = get_buffet(aoas, clcds[:, idx, 1], clcds[:, idx, 0], cl_c=cl_ref, plot=plot)
        
        # avg_bufs = np.mean(_bufs, axis=0)
        # sig_bufs = np.std(_bufs, axis=0)

    return avg_bufs, sig_bufs, (ref_aoa, cl_ref), mu1.squeeze().detach(), log_var1.squeeze().detach()
'''

def get_format_data(folder):
    _, _, foil, _ = cfl3d.readprt_foil(folder, j0=40, j1=341, fname='cfl3d.prt', coordinate='xy')
    _, AoA = cfl3d.readAoA(folder, n=30)

    iLE = np.argmin(foil[0])
    nn = 201
    xx = [clustcos(i, nn) for i in range(nn)]

    yss = []

    for iv in [1, 2, 4]:
        # lower surface
        fy = pchip(foil[0][:iLE+1][::-1], foil[iv][:iLE+1][::-1])
        y_l = fy(xx)

        # upper surface
        fy = pchip(foil[0][iLE:], foil[iv][iLE:])
        y_u = fy(xx)

        ys = np.concatenate((y_l[::-1], y_u[1:]), axis=0)

        if iv == 4:
            ys *= CF_NON_DIM
        yss.append(ys)


    return {'y': np.array(yss), 'aoa_ref': AoA}

def get_mu(path, model, flags):

    _, _, foil = cfl3d.readprt_foil(path, j0=40, j1=341, fname='cfl3d.prt', coordinate='xy')

    # print(succeed3, succeed4)

    # plt.plot(foil[0], foil[2])

    iLE = np.argmin(foil[0])
    nn = 201
    xx = [clustcos(i, nn) for i in range(nn)]

    # lower surface
    fy = pchip(foil[0][:iLE+1][::-1], foil[1][:iLE+1][::-1])
    fp = pchip(foil[0][:iLE+1][::-1], foil[2][:iLE+1][::-1])
    y_l = fy(xx)
    p_l = fp(xx)

    # upper surface
    fy = pchip(foil[0][iLE:], foil[1][iLE:])
    fp = pchip(foil[0][iLE:], foil[2][iLE:])

    y_u = fy(xx)
    p_u = fp(xx)

    y=np.concatenate((y_l[::-1], y_u[1:]), axis=0)
    p=np.concatenate((p_l[::-1], p_u[1:]), axis=0)

    # geom = torch.cat((all_x.unsqueeze(0), torch.from_numpy(y).to('cpu').float().unsqueeze(0)), dim=0)
    ref_data = torch.cat((torch.from_numpy(y).float().unsqueeze(0), torch.from_numpy(p).float().unsqueeze(0)), dim=0)
    ref_data = ref_data.unsqueeze(0).to('cpu')
    _mu, _log_var = model.encode(ref_data[:, :flags['ipt_ch']])

    return _mu, _log_var