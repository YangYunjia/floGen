from flowvae.dataset import FlowDataset
import numpy as np


def cal_loss(vae_model, errors: list, fldata: FlowDataset, paras, wing_class, nn: int, device, ref=-1, ref_channels=(None, None), recon_channels=(None, None), input_channels=(None, None), loss_type='L2'):
    errs = []
    for i_f in range(nn):
        inputs = fldata.inputs[i_f].unsqueeze(0).to(device)
        outputs = fldata.output[i_f].numpy()
        real_field = fldata.all_output[i_f]
        # print(i_f in fldata.data_idx)

        if ref > -1:
            auxs   = fldata.auxs[i_f].unsqueeze(0).to(device)
            output = vae_model(inputs[:, input_channels[0]: input_channels[1]], code=auxs)[0].squeeze(0).cpu().detach().numpy()
            if ref > 0:
                output[ref_channels[0]: ref_channels[1]] += inputs[:, recon_channels[0]: recon_channels[1]].squeeze(0).cpu().detach().numpy()
        else:
            output = vae_model(inputs).squeeze(0).cpu().detach().numpy()

        wg1 = wing_class(geometry=paras[i_f])
        wg1.read_formatted_surface(geometry=real_field[0:3], data=outputs, isnormed=True)
        wg1.lift_distribution()
        cl_real = wg1.cl

        wg2 = wing_class(geometry=paras[i_f])
        wg2.read_formatted_surface(geometry=real_field[0:3], data=output, isnormed=True)
        wg2.lift_distribution()
        cl_recon = wg2.cl
        
#         if i_f in fldata.data_idx and np.abs(cl_real - cl_recon)[0] > 0.02:
        if np.abs(cl_real - cl_recon)[0] > 0.05:
            print(i_f, paras[i_f][0], np.abs(cl_recon - cl_real))
        
        if loss_type == 'L2':
            errs.append(np.concatenate((np.mean((output - outputs)**2, axis=(1,2))**0.5, cl_real, cl_recon)))
        elif loss_type == 'L1':
            errs.append(np.concatenate((np.mean(abs(output - outputs), axis=(1,2)), cl_real, cl_recon)))

    errors.append(errs)