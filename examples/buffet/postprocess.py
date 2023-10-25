from flowvae.vae import EncoderDecoder
from flowvae.ml_operator import AEOperator
from flowvae.dataset import ConditionDataset
from flowvae.base_model import convDecoder, convEncoder
from flowvae.app.buffet import Series, Buffet, Ploter
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.nn import functional as F

from flowvae.post import get_force_1d
from flowvae.utils import warmup_lr


if __name__ == '__main__':

    # define place to store loss
    all_loss = []
    buffet_loss = np.zeros((501, 2, 2))

    # define parameters
    folder = 'testrun'
    device = 'cuda:0'
    latent_dim = 12
    code_mode = 'ed'
    ref = True
    ml_dir = '.'

    cl_cruise = 0.8
    
    # load trained model
    fldata = ConditionDataset('500', n_c=20, c_mtd='load', c_no=0, test=100, data_base='data\\')
    _enc = convEncoder(in_channels=2, last_size=[5], hidden_dims=[32, 64, 128])
    _dec = convDecoder(out_channels=1, last_size=[5], hidden_dims=[64, 128, 256, 128], sizes=[26, 101, 401], dimension=1)
    vae_model = EncoderDecoder(latent_dim=12, encoder=_enc, decoder=_dec, decoder_input_layer=1, code_mode=code_mode, device=device)
    op = AEOperator(folder, vae_model, fldata, ref=ref, output_folder=ml_dir)    
    op.set_scheduler('LambdaLR', lr_lambda=warmup_lr)
    op.load_checkpoint(299)
    op.model.eval()
    all_x = vae_model.geom_data['xx']

    # define buffet onset critria
    buffet_cri = Buffet(method='lift_curve_break', lslmtd = 2, lsumtd = 'error', lsuth2=0.01, intp = '1d')

    # calculate the error of the training and testing dataset
    for i_f in range(0, 501):
        # fetch data of the airfoil series
        afoil = fldata.get_series(i_f)
        geom = torch.cat((all_x.unsqueeze(0), torch.from_numpy(afoil['ref'][0]).to(device).float().unsqueeze(0)), dim=0).to(device)

        # define the series data container
        seri_r = Series(['AoA', 'Cl'])
        seri_m = Series(['AoA', 'Cl'])
        
        # get the latent variables for the airfoil
        ref_data = torch.from_numpy(afoil['ref']).float().unsqueeze(0).to(device)
        mu1, log_var = vae_model.encode(ref_data)

        all_loss_ic = []
        n_c = len(afoil['flowfields'])
        for i_c in range(n_c):

            # the real (ground truth) profiles and cl, cd
            aoa_r = afoil['condis'][i_c]
            profile_r = torch.from_numpy(afoil['flowfields'][i_c][1]).float().to(device)
            cd_r, cl_r = get_force_1d(geom, profile_r, aoa_r).detach().cpu().numpy()

            # generate profiles with the model
            aoa = aoa_r - afoil['ref_aoa'].numpy() * int(ref)
            samp1 = vae_model.sample(num_samples=1, code=aoa, mu=mu1, log_var=None).squeeze()
            recons = samp1 + ref_data[0, 1] * int(ref)
            cd1, cl1 = get_force_1d(geom, recons, aoa_r).detach().cpu().numpy()

            # add to the series container
            seri_r.add(x={'AoA': float(aoa_r), 'Cl': cl_r})
            seri_m.add(x={'AoA': float(aoa_r), 'Cl': cl1})

            # record prediction loss
            a_loss = [float(F.mse_loss(recons, profile_r).cpu().detach().numpy()), cl1, cl_r, cd1, cd_r]
            all_loss_ic.append(a_loss)

        # calculate the buffet onset of the airfoil
        aoas = np.concatenate([np.arange(-2, 1.5, 0.25), np.arange(1.5, 4.5, 0.1)])
        seri_r.down_sampling(aoas, method='1d')
        seri_m.down_sampling(aoas, method='1d')
        plotter = None
        buf_r =buffet_cri.buffet_onset(seri_r, cl_c=cl_cruise, p=plotter)
        buf_m =buffet_cri.buffet_onset(seri_m, cl_c=cl_cruise, p=plotter)
        print(buf_m, buf_r)
        
        # record the loss for the airfoil
        all_loss.append(all_loss_ic)
        buffet_loss[i_f] = np.array([buf_r[0], buf_m[0]])

        if i_f % 50 == 0: print('%d done' % i_f)
            
    torch.save(all_loss, 'test\\reconstruct_loss1')
    torch.save(buffet_loss, 'test\\buffet_loss1')

    plt.plot(buffet_loss[:, 0, 1], buffet_loss[:, 1, 1], 'o')
    plt.show()

    print(np.mean(np.abs(buffet_loss[:400, 0, :] - buffet_loss[:400, 1, :]), axis=0))
    print(np.mean(np.abs(buffet_loss[400:, 0, :] - buffet_loss[400:, 1, :]), axis=0))

    