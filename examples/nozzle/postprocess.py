from flowvae.vae import frameVAE, UEncoderDecoder, DecoderModel
from flowvae.dataset import MCFlowDataset, FlowDataset
from flowvae.base_model.conv import convDecoder, convDecoder_Unet, convEncoder, convEncoder_Unet
from flowvae.base_model.mlp import mlpDecoder
from flowvae.post import get_xyforce_1d_t
from flowvae.ml_operator import load_model_from_checkpoint

from utils import std_atomsphere, ideal_mfr, ideal_thrust, Fluid, npr2ma, tr2ma #, ideal_thrust_speed
# from cfdtools.cfdpp import cfdpp


import numpy as np
import os
import torch
import math

device = 'cuda:0'

def load_model(run, subrun, ipt_ch, ref, encoder_idx, decoder_idx, code_dim, code_layer, actv=torch.nn.LeakyReLU, batchnorm=True, epoch=299):
    date = '0728'
    latent_dim = 12
    ml_dir = 'D:\\jetSERN\\training'
    flags = {}

    # fldata = ConditionDataset('500', n_c=20, c_mtd='load', c_no=0, test=100, ref=ref, data_base=ml_dir+'\\data\\')
    flags['is_ref'] = ref
    flags['ipt_ch'] = ipt_ch
    flags['is_de'] = False

    file_name = '%s_%s_Run%d' % (date, run, subrun)

    if encoder_idx == 100:
        encoder = convEncoder(in_channels=ipt_ch, last_size=[5])
    elif encoder_idx == 101:
        encoder = convEncoder(in_channels=ipt_ch, last_size=[5], pool_strides=[2, 2, 1])
    elif encoder_idx == 201:
        encoder = convEncoder_Unet(in_channels=2, last_size=[7], hidden_dims=[64, 128, 256], pool_strides=[2, 2, 0], 
                                        basic_layers={'actv': actv, 'batchnorm': batchnorm})


    if decoder_idx == 100:
        decoder = convDecoder(out_channels=1, last_size=[5])
    elif decoder_idx == 101:
        decoder = convDecoder(out_channels=2, last_size=[5], sizes=[26, 101, 224], hidden_dims=[512, 256, 128, 128])
    elif decoder_idx == 102:
        decoder = convDecoder(out_channels=2, last_size=[5], sizes=[26, 101, 234], hidden_dims=[512, 256, 128, 128])
    elif decoder_idx == 201:
        decoder = convDecoder_Unet(out_channels=2, last_size=[7], hidden_dims=[256, 128, 64, 64], 
                                            sizes = [14, 58, 234], encoder_hidden_dims=[256, 128, 64, 2],
                                            basic_layers={'actv': actv, 'batchnorm': batchnorm})
    elif decoder_idx == 0:
        decoder = mlpDecoder(in_channels=latent_dim, out_sizes=[2], hidden_dims=[64, 128, 64])

    if encoder_idx >= 200:
        vae_class = UEncoderDecoder
    else:
        vae_class = frameVAE

    vae_model = vae_class(latent_dim=latent_dim,
                        encoder=encoder,
                        decoder=decoder,
                        decoder_input_layer=1,
                        code_dim=code_dim,
                        code_layer=code_layer,
                        code_mode='ed')

    load_model_from_checkpoint(vae_model, epoch, ml_dir+'\\save\\'+file_name, device=device)

    return vae_model, flags

def load_ae_model(run, subrun, decoder_idx, actv=torch.nn.LeakyReLU, batchnorm=True, epoch=299, decoder_input_layer=[32, 64, 128]):

    date = '0728'
    ml_dir = 'D:\\jetSERN\\training'

    file_name = '%s_%s_Run%d' % (date, run, subrun)
    flags = {}
    flags['is_ref'] = False
    flags['ipt_ch'] = None
    flags['is_de'] = True
    
    if decoder_idx == 201:
        decoder = convDecoder(out_channels=2, last_size=[7], hidden_dims=[256, 128, 64, 64], 
                                        sizes = [14, 58, 234], basic_layers={'actv': actv, 'batchnorm': batchnorm})
     
    vae_model = DecoderModel(input_channels=8, decoder=decoder, device=device, decoder_input_layer=decoder_input_layer)

    load_model_from_checkpoint(vae_model, epoch, ml_dir+'\\save\\'+file_name, device=device)

    return vae_model, flags

def cal_error(vae_model, flags, fldata, loss_file):

    # x_cor_upper = np.linspace(26.20, 29.90, 149)
    # x_cor_lower = np.linspace(26.00, 27.85, 75)
    # commons = {'mass': [], 'flux': []}
    commons = torch.load('commons')

    n_sum = 0
    deltap = 0.0
    deltat = 0.0
    # geom = get_geom(folder='F:\\DeepLearning\\202204JetSERN\\database0\\Calculation\\0\\0')
    # commons['geom'] = geom
    geom = (commons['geom'][0].to(device), commons['geom'][1].to(device))

    with open('D:\\jetSERN\\injectiondata\\database_4.dat', 'r') as f:
        cfd_lines = f.readlines()

    if flags['is_de']:
        split_paras = {
            'method': 'load',   # overall + condition selection methods (load, random, all)
            'indexFileName': 'D:\\jetSERN\\training\\data\\300n36_0dataindex.txt', # Number of the index file
        }
        fldata1 = FlowDataset(['300n36ndata', '300n36n2index'], split_paras=split_paras, input_channel_take=[3,4,5,6,15,16,17,18])
    

    for i_f in range(300):
        adata = fldata.get_series(i_f)
        ref_data = torch.from_numpy(adata['ref']).float().unsqueeze(0).to(device)

        # mass, flux = extract_throat('F:\\DeepLearning\\202204JetSERN\\database0\\Calculation\\%d\\baseline' % i_f)
        # commons['mass'].append(mass)
        # commons['flux'].append(flux)

        mass = commons['mass'][i_f]
        flux = commons['flux'][i_f]

        n_index4 = fldata.all_index[i_f * 37]
        main_condition = {}
        main_condition['ptotal'] = n_index4[11] 
        main_condition['patm'] = n_index4[12] 
        main_condition['ttotal'] = n_index4[13]
        # ref_lowerprofile = dim_wall(deepcopy(ref_data[0]), main_condition, sec_c=300)[0, :75].flip(dims=[0]).cpu()
        ref_lowerprofile = None

        for i_c in range(1, len(adata['condis'])):
            
            real_field = torch.from_numpy(adata['flowfields'][i_c]).float().to(device)
            conditions = adata['condis'][i_c]

            if flags['is_de']:
                ref_field = torch.from_numpy(fldata1.inputs[i_f*37+i_c]).float().unsqueeze(dim=0).to(device)
                recons_field = vae_model(ref_field)[0]

            else:
                mu, _ = vae_model.encode(ref_data)
                samp = vae_model.sample(num_samples=1, code=conditions-adata['ref_aoa'].cpu().numpy(), mu=mu)
                recons_field = samp[0] + ref_data[0] * flags['is_ref']

            n_sum += 1
            # a_dp = np.sqrt(float(F.mse_loss(recons_field[0], real_field[0]).cpu().detach().numpy()))
            # a_dt = np.sqrt(float(F.mse_loss(recons_field[1], real_field[1]).cpu().detach().numpy()))
            a_dp = float(torch.mean(torch.abs(recons_field[0] - real_field[0])).cpu().detach().numpy())
            a_dt = float(torch.mean(torch.abs(recons_field[1] - real_field[1])).cpu().detach().numpy())
            deltap += a_dp
            deltat += a_dt
 
            real_field = dim_wall(real_field, main_condition, is_sec=True, sec_c=conditions)
            recons_field = dim_wall(recons_field, main_condition, is_sec=True, sec_c=conditions)
            # real_field = dim_wall(real_field, main_condition, is_sec=True, sec_c=300)
            # recons_field = dim_wall(recons_field, main_condition, is_sec=True, sec_c=300)

            #! get the prediction error of thrust coefficients and thrust angle
            rel_rth, rel_the = get_thrust(*geom, real_field, mass, flux, main_condition, torch.from_numpy(conditions), ref_lowerprofile=ref_lowerprofile)
            rec_rth, rec_the = get_thrust(*geom, recons_field, mass, flux, main_condition, torch.from_numpy(conditions), ref_lowerprofile=ref_lowerprofile)

            print(i_f, i_c, '%.4f %.4f %.4f' % (rel_rth.item(), rec_rth.item(), rel_rth.item() - rec_rth.item()))
            cfd_rth = float(cfd_lines[39 * i_f + i_c + 3].split()[-8])
            cfd_the = float(cfd_lines[39 * i_f + i_c + 3].split()[-7])
            #                                                                                       
            loss_file.write('%3d %3d %18.9f %18.9f %18.9f %18.9f %18.9f %18.9f %18.9f %18.9f\n' 
                % (i_f, i_c, a_dp, a_dt, rel_rth, rel_the, rec_rth, rec_the, cfd_rth, cfd_the))
            #        0    1     2     3        4                5                6                7                8         9
        
        # if i_f % 50 == 49:
        #     print(i_f)
        #     torch.save(commons, 'D:\\DeepLearning\\202204JetSERN\\database0\\training\\commons')

    return deltap / n_sum, deltat / n_sum

def find_slit(xx):
    '''
    # ledge =  upstream edge of the slit
    # redge =  downstream edge of the slit
    # xslit = 0.5 * (ledge + redge)
    '''
    
    for ii in range(1,len(xx)-1):
        if xx[ii+1] - xx[ii] > 3.0 * (xx[ii] - xx[ii-1]):
            ledge = xx[ii]
            redge = xx[ii+1]
            xslit = 0.5 * (ledge + redge)
            break
    else:
        xslit, ledge, redge = 0.0, 0.0, 0.0
        print('!! no slit found !!')

    return xslit, ledge, redge

def find_sep(xx, tau_x):
    '''
    return:

    # PUV (primary upstream vortex)
    # SUV (secondary upstream vortex)
    # PDV (primary downstream vortex)
    '''
    
    _, ledge, redge = find_slit(xx)
    
    flag = False
    sepe = 0.0   #  first seperation in PUV (primary upstream vortex)
    reta = 0.0   #  the stagnation point between PUV and SUV (secondary upstream vortex)
    reta2 = 0.0  #  the reattachment point at the end of PDV (primary downstream vortex)
    
    for ii in range(len(tau_x)-1):
        if tau_x[ii] > 0 and tau_x[ii + 1] <= 0:
            sepe = (0.0-tau_x[ii])*(xx[ii+1]-xx[ii])/(tau_x[ii+1]-tau_x[ii])+xx[ii]
            flag = True
        if flag and tau_x[ii] < 0 and tau_x[ii + 1] >= 0:
            reta = (0.0-tau_x[ii])*(xx[ii+1]-xx[ii])/(tau_x[ii+1]-tau_x[ii])+xx[ii]
            break
    
    for ii in range(len(tau_x)-2, 0, -1):
        if tau_x[ii+1] > 0 and tau_x[ii] <= 0:
            reta2 = (0.0-tau_x[ii])*(xx[ii+1]-xx[ii])/(tau_x[ii+1]-tau_x[ii])+xx[ii]
            break

    print(sepe, reta, ledge, redge, reta2)

    if sepe > 0.0 and reta > 0.0 and sepe < ledge and reta < ledge:
        puv = reta - sepe
        suv = ledge - reta
    elif sepe > 0.0 and sepe < ledge:
        puv = ledge - sepe
        suv = 0.0
    else:
        puv = 0.0
        suv = 0.0

    if reta2 > 0.0 and reta2 > redge:
        pdv = reta2 - redge
    elif tau_x[-1] < 0.0 and reta2 > redge:
        pdv = xx[-1] - reta2
    else:
        pdv = 0.0

    return puv, suv, pdv

def read_wall(folder, forcenew=True):

    '''
    new version of the post_wall.read_from_wall, use the cfdpp.extract_bc to replace the export_wall.mcr
    to raise the accuracy.

    return
    ===
    (upper wall data, lower wall data)
        the variables in them are (x, y, p, T, tau_x)
    '''

    op = cfdpp(folder)
    data = op.extract_bc([2, 3], forcenew=forcenew)

    d_u = np.take(data['lines'][0]['data'],[0, 1, 3, 4, 13], axis=0).transpose()
    d_l = np.take(data['lines'][1]['data'],[0, 1, 3, 4, 13], axis=0).transpose()
    print(".. wall data in %s is read, the shape of lower/uper is " % folder, d_l.shape, d_u.shape)

    ind_u = np.lexsort([d_u[:, 0]])
    d_u_sorted = np.array([d_u[i] for i in ind_u])

    ind_l = np.lexsort([d_l[:, 0]])
    d_l_sorted = np.array([d_l[i] for i in ind_l])

    # print(d_u_sorted[:, 0])
    # print(d_l_sorted[:, 0])
    if not (np.all(np.diff(d_u_sorted[:, 0]) > 0)) or not (np.all(np.diff(d_l_sorted[:, 0]) > 0)):
        raise Exception(folder)

    return d_u_sorted, d_l_sorted

def interpolate_wall(upper, lower, folder, is_old=False):
    '''

    '''
    print(is_old)
    if is_old:
        print('!')
        x_cor_upper = np.linspace(26.20, 29.90, 149)
        x_cor_lower = np.linspace(26.00, 27.85, 75)
    else:
        x_cor_upper = np.linspace(26.00, 29.95, 159)
        x_cor_lower = np.linspace(26.00, 27.85, 75)

    y_upp = np.interp(x_cor_upper, upper[:, 0], upper[:, 1])
    y_low = np.interp(x_cor_lower, lower[:, 0], lower[:, 1])

    p_upp = np.interp(x_cor_upper, upper[:, 0], upper[:, 2])
    p_low = np.interp(x_cor_lower, lower[:, 0], lower[:, 2])

    t_upp = np.interp(x_cor_upper, upper[:, 0], upper[:, 3])
    t_low = np.interp(x_cor_lower, lower[:, 0], lower[:, 3])

    dudy_upp = np.interp(x_cor_upper, upper[:, 0], upper[:, 4])
    dudy_low = np.interp(x_cor_lower, lower[:, 0], lower[:, 4])


    with open(os.path.join(folder, 'ram_upper.dat'), 'w') as ff:
        for idx in range(p_upp.shape[0]):
            ff.write(' %18.9f  %18.9f %18.9f  %18.9f  %18.9f\n' % (x_cor_upper[idx], y_upp[idx], p_upp[idx], t_upp[idx], dudy_upp[idx]))
    with open(os.path.join(folder, 'ram_lower.dat'), 'w') as ff:
        for idx in range(p_low.shape[0]):
            ff.write(' %18.9f  %18.9f %18.9f  %18.9f  %18.9f\n' % (x_cor_lower[idx], y_low[idx], p_low[idx], t_low[idx], dudy_low[idx]))

    return [[p_upp, t_upp], [p_low, t_low]]

def nondim_sec(alpha, spr, tsec, ratio, ttotal, **kwargs):
    '''
    non-dimensionalize the secondary injection data

    '''
    sec_condition = np.zeros((4,))
    sec_condition[0] = (alpha - 30) / 120  #alpha
    sec_condition[1] = (spr - 0.1) / 0.8  #spr
    sec_condition[2] = (tsec - 300) / (ttotal - 300) / 0.6 #rtsec
    sec_condition[3] = (ratio - 0.1) / 0.85  #ratio

    return sec_condition

def dim_sec(sec_condition, ttotal, **kwargs):
    '''
    non-dimensionalize the secondary injection data

    '''
    alpha = sec_condition[0] * 120 + 30  #alpha
    spr   = 0.8 * sec_condition[1] + 0.1 #spr
    tsec  = sec_condition[2] * (ttotal - 300) * 0.6 + 300 #rtsec
    ratio = sec_condition[3] * 0.85 + 0.1  #ratio

    return alpha, spr, tsec, ratio

def nondim_wall(walldata, condition, is_sec=False, sec_c=0.0):
    '''
    non-dimensionalize the walldata

    condition should be a dict. containing:
    - npr
    - patm
    - ttotal

    if the wall data need to non-dimensionalize is a ff. with secondary injection, condition should also containing
    (set `is_sec` to True)
    - tsec

    #! In pre_process_4.py all the training data's wall results are non-dimensionalized with sec_c = 300.

    '''
    if 'npr' in condition.keys():
        ptotal = condition['npr'] * condition['patm']   # p_total
    elif 'ptotal' in condition.keys():
        ptotal = condition['ptotal']
    patm   = condition['patm']    # p_atm
    ttotal = condition['ttotal']     # t_total
    if is_sec:
        tslit = 300 + sec_c[2] * (ttotal - 300) * 0.6 #tsec     # t_slit
    else:
        tslit = sec_c

    walldata[0, :] = (ptotal - walldata[0, :]) / (ptotal - patm)
    walldata[1, :] = (ttotal - walldata[1, :]) / (ttotal - tslit)

    return walldata

def dim_wall(walldata, condition, is_sec=False, sec_c=0.0):
    '''
    dimensionalize the walldata
    
    condition should be a dict. containing:
    - npr
    - patm
    - ttotal

    if the wall data need to dimensionalize is a ff. with secondary injection, condition should also containing
    (set `is_sec` to True)
    - tsec

    #! In pre_process_4.py all the training data's wall results are non-dimensionalized with sec_c = 300.

    '''
    if 'npr' in condition.keys():
        ptotal = condition['npr'] * condition['patm']   # p_total
    elif 'ptotal' in condition.keys():
        ptotal = condition['ptotal']
    patm   = condition['patm']    # p_atm
    ttotal = condition['ttotal']     # t_total
    if is_sec:
        tslit = 300 + sec_c[2] * (ttotal - 300) * 0.6 #tsec     # t_slit
    else:
        tslit = sec_c
    if walldata.dim() == 2:
        walldata[0, :] = ptotal - walldata[0, :] * (ptotal - patm)
        walldata[1, :] = ttotal - walldata[1, :] * (ttotal - tslit)
    else:
        walldata[:, 0, :] = ptotal.unsqueeze(1) - walldata[:, 0, :] * (ptotal - patm).unsqueeze(1)
        walldata[:, 1, :] = ttotal.unsqueeze(1) - walldata[:, 1, :] * (ttotal - tslit).unsqueeze(1)
    return walldata

def get_format_data(folder, is_old=False, is_force=False):
    '''
    get data for training from folder

    condition should be a dict. containing:
    - npr
    - patm
    - ttotal

    if the folder containing a ff. with secondary injection, condition should also containing
    (set `is_sec` to True)
    - alpha
    - spr
    - tsec
    - ratio

    '''
    if not os.path.exists(folder + '\\ram_upper.dat') or is_force:
        rst = read_wall(folder)
        _ = interpolate_wall(*rst, folder, is_old=is_old)
    
    if is_old:
        total = 149
    else:
        total = 159

    alldata = np.zeros((2, 75 + total))
    with open(folder + '\\ram_upper.dat') as f:
        lines = f.readlines()
        for j in range(total):
            line_sp = lines[j].split()
            alldata[0, j + 75] = float(line_sp[2])
            alldata[1, j + 75] = float(line_sp[3])

    with open(folder + '\\ram_lower.dat') as f:
        lines = f.readlines()
        for j in range(75):
            line_sp = lines[j].split()
            alldata[0, 74 - j] = float(line_sp[2])
            alldata[1, 74 - j] = float(line_sp[3])
    
    return alldata

def get_geom(folder, is_old=False, device='cuda:0'):
    '''
    get the interpolated geometry from  the calculation folder

    ** should have interpolated datafile ram_upper.dat and ram_lower.dat **
    
    x_cor_upper = np.linspace(26.00, 29.95, 159)\n
    x_cor_lower = np.linspace(26.00, 27.85, 75)

    return:
    ===
    
    geomu: upper surface (x, y), np.array with size (2, 159)\n
    geoml: lower surface (x, y), np.array with size (2, 159)

    '''

    if not os.path.exists(folder + '\\ram_upper.dat'):
        rst = read_wall(folder)
        _ = interpolate_wall(*rst, folder, is_old=is_old)

    if is_old:
        delta = 8
        total = 149
    else:
        delta = 0
        total = 159
    
    geomu = torch.zeros((2, total)).to(device)
    geoml = torch.zeros((2, 75)).to(device)
    with open(folder + '\\ram_upper.dat', 'r') as f:
        lines = f.readlines()
    for i in range(total):
        geomu[0, i] = float(lines[i+delta].split()[0])
        geomu[1, i] = float(lines[i+delta].split()[1])
    
    with open(folder + '\\ram_lower.dat', 'r') as f:
        lines = f.readlines()
    for i in range(75):
        geoml[0, i] = float(lines[i].split()[0])
        geoml[1, i] = float(lines[i].split()[1])
    
    return geomu, geoml

def extract_throat(folder):

    op = cfdpp(op_dir=folder)

    throat_line = op.extract_line(st=(26.0, -1.4838, 0.0), ed=(26.0, -0.89913, 0.0), forcenew=False )
    # throat_line = op.extract_line(st=(26.0, -1.4838, 0.0), ed=(26.2, -0.8929, 0.0), forcenew=False)

    # print(throat_line['varnames'])
    # print(throat_line['lines'][0]['data'])
    xx = torch.FloatTensor(throat_line['lines'][0]['data'][2:4])
    pp = torch.FloatTensor(throat_line['lines'][0]['data'][5])
    uu = torch.FloatTensor(throat_line['lines'][0]['data'][7])
    vv = torch.FloatTensor(throat_line['lines'][0]['data'][8])
    rho = torch.FloatTensor(throat_line['lines'][0]['data'][10])

    mass_fr, flux_inlet = get_flux_1d(xx, pp, uu, vv, rho)

    return mass_fr, flux_inlet

def get_cfd_force(folder, p):
    
    op = cfdpp(op_dir=folder ,core=4, verbose='Warning')
    mram = -op.read_flux('mass', [4])
    thrust_x = -(op.read_flux('fx', [2, 3, 4, 8]) - op.read_area('x', [2, 3, 4, 8]) * p)
    thrust_y = -(op.read_flux('fy', [2, 3, 4, 8]) - op.read_area('y', [2, 3, 4, 8]) * p)

    return thrust_x, thrust_y, mram

def get_approx_cfx(folder, width=0.017):

    air = Fluid(name='Air')
    with open(os.path.join(folder, 'output.txt'), 'r') as f:
        lines = f.readlines()
    
    real = get_format_data(os.path.join(folder, '..\\baseline'))
    ref = real[0, :75][::-1]

    ratio = float(lines[7].split()[-1])
    patm = float(lines[8].split()[-1])
    ttotal = float(lines[3].split()[-1])
    ptotal = float(lines[2].split()[-1]) * patm
    mass_fr = float(lines[9].split()[-1])
    slit_pt = float(lines[5].split()[-1]) * ptotal
    tsec = float(lines[6].split()[-1])
    alpha = float(lines[4].split()[-1])
    msec = float(lines[10].split()[-1]) 
    fx = float(lines[13].split()[-1]) 

    th_id, _ = ideal_thrust(ptotal, ttotal, patm, mass_fr, fluid=air)
    
    slit_p = np.interp([ratio], np.linspace(0, 1, num=75), ref)[0]
    if slit_pt <= slit_p:
        m_slit = 0.0
        th_slit_id = 0.0
        # ma8 = 0.0
    else:
        ma8 = min(npr2ma(slit_pt/slit_p), 1.0)
        m_slit = ideal_mfr(pt8=slit_pt, tt8=tsec, A8=width*math.sin(alpha/180*3.14), ma8=ma8) # *torch.sin(alpha/180*3.14)
        th_slit_id, _ = ideal_thrust(pt7=slit_pt, tt7=tsec, p9=patm, m8=m_slit, fluid=None)
    new_cfx = fx / (th_id + th_slit_id)
    # print(slit_p, m_slit, msec, fx, th_id, th_slit_id)

    return new_cfx

def get_delta_cfx(folder, width=0.017):
    
    air = Fluid(name='Air')
    with open(os.path.join(folder, 'output.txt'), 'r') as f:
        lines = f.readlines()

    real_cfx_0 = get_approx_cfx(os.path.join(folder, '..\\baseline'))
    
    geom = get_geom(folder=os.path.join(folder, '..\\baseline'))
    mass, flux = extract_throat(folder=os.path.join(folder, '..\\baseline'))
    real = torch.from_numpy(get_format_data(os.path.join(folder, '..\\baseline'))).float()

    patm = float(lines[8].split()[-1])
    ttotal = float(lines[3].split()[-1])
    ptotal = float(lines[2].split()[-1]) * patm

    main_condition = {}
    main_condition['ptotal'] = ptotal
    main_condition['ttotal'] = ttotal
    main_condition['patm'] = patm

    rth0, the0, data0 = get_thrust(*geom, real, mass, flux, main_condition, sec_c=None, rtmore=True)

    print(real_cfx_0, rth0)

    return real_cfx_0 - rth0

def get_thrust(geomu, geoml, profile, mass_fr, flux_inlet, main_c, sec_c, width=0.017, ref_lowerprofile=None, rtmore=False):
    '''
    get data for training from folder

    profileu, profilel

    mass_fr, flux_inlet are obtained from get_throat()

    main_c should be a dict. containing:
    - npr
    - patm
    - ttotal

    sec_c is non-dimensionalized, secondary condition (which is input to the decoder)

    usage:
    ===
        geom = get_geom(folder='...')
        mass, flux = extract_throat(folder='...')
        ref_data = get_format_data(folder='...', condition=main_condition)  # .float().unsqueeze(0).to(device)  # output is non-dimensional
        ref_data = nondim_wall(ref_data, main_condition)

        sec_condition = nondim_sec(alpha, spr, tsec, ratio, ttotal)  # if input is array, use: nondim_sec(*x, ttotal)
        mu, _ = vae_model.encode(ref_data)
        samp = vae_model.sample(num_samples=1, mode='distr', code=sec_condition, mu=mu, log_var=None)
        recons_field = samp[0] + ref_data[0]

        recons_field = dim_wall(recons_field, main_condition, is_sec=True, sec_c=sec_condition)
            
        rel_rth, rel_the = get_thrust(*geom, recons_field, mass, flux, main_condition, sec_condition)

    '''
    if 'npr' in main_c.keys():
        ptotal = main_c['npr'] * main_c['patm']   # p_total
    elif 'ptotal' in main_c.keys():
        ptotal = main_c['ptotal']
    patm   = main_c['patm']    # p_atm
    ttotal = main_c['ttotal']     # t_total

    profileu = profile[0, 75:]
    profilel = profile[0, :75].flip(dims=[0])

    fxu, fyu = get_xyforce_1d_t(geomu.permute(1, 0).unsqueeze(0), profileu.unsqueeze(0), tail=False)[0]
    fxl, fyl = get_xyforce_1d_t(geoml.permute(1, 0).unsqueeze(0), profilel.unsqueeze(0), tail=False)[0]

    air = Fluid(name='Air')
    th_id, _ = ideal_thrust(ptotal, ttotal, patm, mass_fr, fluid=air)

    is_valid_sec = False
    
    if sec_c is not None:
        slit_pt = (sec_c[1] * 0.8 + 0.1) * ptotal #spr
        tsec = 300 + sec_c[2] * (ttotal - 300) * 0.6 #tsec
        alpha = 30 + 120 * sec_c[0]
        ratio = (sec_c[3] * 0.85 + 0.1)

        if ref_lowerprofile is not None:
            slit_p = np.interp([ratio], np.linspace(0, 1, num=75), ref_lowerprofile)[0]

            if slit_pt > slit_p:   is_valid_sec = True
        
        else:
            slit_p = slit_pt * 0.1
            is_valid_sec = True

    if is_valid_sec:
        ma8 = min(npr2ma(slit_pt/slit_p), 1.0)
        if ma8 < 1.0:
            print('!!!!!!!!!!!!!!')
        m_slit = ideal_mfr(pt8=slit_pt, tt8=tsec, A8=width*torch.sin(alpha/180*3.14), ma8=ma8) # *torch.sin(alpha/180*3.14)
        th_slit_id, _ = ideal_thrust(pt7=slit_pt, tt7=tsec, p9=patm, m8=m_slit, fluid=None)
        # th_sec = m_slit * (1.4 * 287 * tsec / (1 + 287 / 2.5 / 2))**0.5
        th_sec = m_slit * ma8 * (1.4 * 287 * tsec / (1 + 0.2 * ma8**2))**0.5
        th_sec_x = torch.cos((alpha - 5)/180*3.14) * th_sec
        th_sec_y = torch.sin((alpha - 5)/180*3.14) * th_sec
    
    
    else:
        m_slit = 0.0
        th_slit_id = 0.0
        th_sec_y = 0.0
        th_sec_x = 0.0
    # print(mass_fr, m_slit, th_id, th_slit_id)
    # print(flux_inlet[0].item(),fxu.item(),fxl.item(),abs(geomu[1, -1] - geoml[1, -1]),float(th_sec_x))

    fx = flux_inlet[0] + fxu - fxl - patm * abs(geomu[1, -1] - geoml[1, -1]) + th_sec_x
    fy = flux_inlet[1] + fyu - fyl + patm * abs(geomu[0, -1] - geoml[0, -1]) + th_sec_y

    r_th = fx / (th_id + th_slit_id)
    # r_th = torch.sqrt(fx**2 + fy**2) / (th_id + th_slit_id)
    theta = torch.atan(fy / fx) / 3.14 * 180
    if rtmore:
        return r_th, theta, (fx, fy, mass_fr, m_slit, th_id, th_slit_id)
    else:
        return r_th, theta
    

if __name__ == '__main__':

    run = '53n1'

    split_paras = {
        'method': 'load',  
        'indexFileNum': 0,  
    }
    fldata = MCFlowDataset('300n36n', d_c=4, split_paras=split_paras, getindex_type='normal')
    
    loss_file = open('D:\\jetSERN\\training\\save\\0728_%s.loss.n1.mae' % run, 'w')
    
    for subrun in range(3):

        vae_model, flags = load_ae_model(run=run, subrun=subrun, decoder_idx=201, decoder_input_layer=1, 
                                        actv=torch.nn.GELU, batchnorm=True, epoch=299)
        # vae_model, flags = load_model(run=run, subrun=subrun, ipt_ch=2, ref=True, decoder_idx=201, encoder_idx=201, 
        #                               code_dim=4, code_layer=[], actv=torch.nn.GELU, batchnorm=True, epoch=299)
        dp, dt = cal_error(vae_model, flags, fldata, loss_file)
        
        print(subrun, dp, dt)
    
    loss_file.close()

