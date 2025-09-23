'''
This is the training code for paper 'Rapid aerodynamic prediction of swept wings via physics-embedded transfer learning'

Author: Yunjia Yang


'''


from flowvae.ml_operator.operator import ModelOperator, BasicCondAEOperator, load_model_from_checkpoint
from flowvae.ml_operator.kfold import K_fold, K_fold_evaluate
from flowvae.dataset import FlowDataset
from flowvae.utils import warmup_lr, warmup_lr_4, load_encoder_decoder
import matplotlib.pyplot as plt
import torch
import numpy as np
import os, copy
from flowvae.app.wing.models import ounetbedmodel, triinput_simplemodel1, bresnetunetmodel
from flowvae.app.wing.utils import cal_loss


from cfdpost.wing.basic import Wing, KinkWing, plot_compare_2d_wing
from torch.utils.data import Subset

folder = '../wing2/data'
device = "cuda:0"
paras = np.load(os.path.join(folder, 'wingindex.npy'))

class Train_and_Test():

    def __init__(self, file_name, ref, input_ref, ref_type, sa_type=0.25) -> None:
        '''
        ref_type    = 0     cl_real
                    = 1     cl_vlm 
                    = 2     cl model
                    = 3     cl - cl_vlm model  
                    = 4     aoa2d
                    = 5     aoa2d1 (downwash)   
        
        '''
        self.file_name = file_name
        self.ref_type = ref_type
        self.ref = ref
        self.input_ref = input_ref
        self.sa_type = sa_type

        if ref_type == -1:
            # just a placeholde, the `2dgeomdata1_25` will not be used under `ref_type == 0`
            # self.fldata_3d = FlowDataset(['wing101', '2dgeomdata1_25', 'wing101deltaindex'], \
            #                      c_mtd='all', output_channel_take=[3,4,5], input_channel_take=[1,3,4], data_base=folder)
            self.fldata_3d = FlowDataset(['wing101', '2dxyzdata1_25', 'wingindexnondim'], c_mtd='all', output_channel_take=[3,4,5], data_base=folder)
        
        if self.ref_type == 2:
            # self.fldata_cl = FlowDataset(['wingcl101', 'wing101deltaindex'], c_mtd='all', output_channel_take=[1], data_base=folder, swap_axis=(1,2))
            self.fldata_cl = FlowDataset(['wingcl101', 'wingindexnondim'], c_mtd='all', output_channel_take=[1], input_channel_take=list(range(1,9))+list(range(10,31)), data_base=folder)
        elif self.ref_type == 3:
            # self.fldata_cl = FlowDataset(['wingcl101_minus_vlm', 'wing101deltaindex'], c_mtd='all', output_channel_take=[1], data_base=folder, swap_axis=(1,2))
            self.fldata_cl = FlowDataset(['wingcl101_minus_vlm', 'wingindexnondim'], c_mtd='all', output_channel_take=[1], input_channel_take=list(range(1,9))+list(range(10,31)), data_base=folder)
            
        if self.ref_type in [0, 1, 2, 3, 4, 5]:
            self.model_2d = bresnetunetmodel(h_e=[16, 32, 64], h_d1=[64, 128], h_d2=[512, 256, 128, 64], h_out=2, h_in=1, device=device)
            if self.ref_type in [0, 1, 2, 3]:  model_2d_folder = '0330_25_Run3'
            else:                              model_2d_folder = '0330_26_Run4'
            load_model_from_checkpoint(self.model_2d, epoch=299, folder=os.path.join('../airfoil2/save', model_2d_folder), device=device)


    def func_train(self, irun, itrain):
        batch_size = 16

        if self.ref_type in [2, 3]:
            #* train cl prediction model
            train_cl = Subset(self.fldata_cl, itrain)

            # encoder = cat_mlpconv_encoder(h_e1=[32, 64, 128], h_e2=[32, 64], kernel=7, nt=101, dropout=0.0)
            # model_cl = simpledecode(h_d=[192, 64, 32, 16, 1], h_e=encoder, nt=101, kernel=7, device=device)
            # model_cl = triinput_simplemodel1(h_e1=[32, 64], h_e2=[16, 32], h_e3=[16], h_d1=[128, 101], nt=101)
            model_cl = triinput_simplemodel1(h_e1=[32, 32], h_e2=[16, 16], h_e3=[16], h_d1=[64, 101], nt=101)
            
            op_cl = ModelOperator(f'{self.file_name}{irun}_cl', model_cl, train_cl, shuffle=True, 
                                                            batch_size=batch_size, 
                                                            init_lr=1e-4,
                                                            num_epochs=900, split_train_ratio=1.0)
            
            op_cl.set_scheduler('LambdaLR', lr_lambda=warmup_lr_4) 
            op_cl.train_model(save_check=900, v_tqdm=False) 

            predicted_cldis = []
            for i_f in range(len(self.fldata_cl.all_output)):
                
                inputs = self.fldata_cl.inputs[i_f].unsqueeze(0).to(device)
                output = model_cl(inputs).squeeze(0).cpu().detach().numpy()[0]

                predicted_cldis.append(output)
                
            predicted_cldis = np.array(predicted_cldis)
            if self.ref_type == 3:
                predicted_cldis += np.load(os.path.join(folder, 'cldistribution_vlm.npy'))

        if self.ref_type in [0, 1, 2, 3, 4, 5]:

            #* use 2d model to predict prior distributions
            print('use 2d model to predict prior distributions')
            
            indxs = np.load(os.path.join(folder, 'wing101deltaindex.npy'))
            paras = np.load(os.path.join(folder, 'wingindex.npy'))
            geoms = np.load(os.path.join(folder, 'winggeom.npy'))
            if self.ref_type in [2, 3]: cldis = predicted_cldis
            elif self.ref_type in [1]:  cldis = np.load(os.path.join(folder, 'cldistribution_vlm.npy'))
            elif self.ref_type in [0]:  cldis = np.load(os.path.join(folder, 'wingcl101.npy'))[:, 1]

            all_inputs = []
            all_output = []

            for i_f in range(indxs.shape[0]):
                ma3d = paras[i_f, 2]
                re3d = 6.429 

                inputs = indxs[i_f]
                wg = Wing(paras[i_f])

                wg_sections = (inputs[:, 2] / wg.g['half_span'])
                
                if self.sa_type > 0.:
                    sa4 = wg.swept_angle(self.sa_type) / 180*np.pi
                else:
                    sa4 = 0.

                if self.ref_type in [0, 1, 2, 3]:
                    # use cl as transfer                
                    cl3d = cldis[i_f]
                    cl2d = cl3d / np.cos(sa4)**2

                elif self.ref_type in [4]:
                    # use aoa
                    aoa3ds = inputs[:, 3]
                    cl2d = aoa3ds / np.cos(sa4)
                    # print(aoa2ds)
                elif self.ref_type in [5]:
                
                    AoA0 = -0.5
                    xx = wg.thin_wing()
                    downwash = wg.downwash_angle(wg_sections, xx)
                    aoa3ds1 = inputs[:, 3] - downwash * (wg.g['AoA'] - AoA0)
                    cl2d = aoa3ds1 / np.cos(sa4)
                
                re2ds = re3d * np.cos(sa4) * wg.sectional_chord_eta(wg_sections)
                
                auxs2d = np.vstack((np.tile([[ma3d * np.cos(sa4)]], (1, 101)), 
                                    cl2d.reshape(1, -1), 
                                    re2ds.reshape(1, -1))).transpose(1,0)
                
                geoms2d = geoms[i_f][:, 1:2] / np.cos(sa4)
                
                output = self.model_2d(torch.from_numpy(geoms2d).float().to(device), 
                                       code=torch.from_numpy(auxs2d).float().to(device))
                
                output = output[0].cpu().squeeze(0).detach().numpy() * np.cos(sa4)**2

                all_output.append(output)
                # print(inputs)

            all_output = np.array(all_output).transpose((0, 2, 1, 3))
            # all_output = np.hstack((geoms.transpose((0,2,1,3)),all_output))

            # np.save(f'../wing2/data/2dgeomdata1_{self.file_name}{irun}', all_output)
            # self.fldata_3d = FlowDataset(['wing101', f'2dgeomdata1_{self.file_name}{irun}', 'wing101deltaindex'], \
            #                               c_mtd='all', output_channel_take=[3,4,5], input_channel_take=[1,3,4], data_base=folder)
            
            wingdata = np.load(os.path.join(folder, 'wing101.npy'))
            all_output = np.hstack((wingdata[:, :3], all_output))

            np.save(f'../wing2/data/2dxyzdata1_{self.file_name}{irun}', all_output)
            self.fldata_3d = FlowDataset(['wing101', f'2dxyzdata1_{self.file_name}{irun}', 'wingindexnondim'], \
                                          c_mtd='all', output_channel_take=[3,4,5], data_base=folder)
            
        # print("network have {} paramerters in total".format(sum(x.numel() for x in vae_model.parameters())))

        train_3d = Subset(self.fldata_3d, itrain)
        # model_3d = ounetbedmodel(h_e=[16, 32, 64, 64, 128, 128], h_e1=None, h_e2=[64, 256, 768], h_d=[257, 128, 128, 64, 64, 32, 32], 
        #                           h_in=self.input_ref, h_out=3, de_type='cat', coder_type ='onlywing', coder_kernel=3, device=device)
        model_3d = ounetbedmodel(h_e=[32, 64, 64, 128, 128, 256], h_e1=None, h_e2=None, h_d=[258, 128, 128, 64, 64, 32, 32],
                                  h_in=self.input_ref, h_out=3, de_type='cat', coder_type ='onlycond', coder_kernel=3, device=device)
        
        op = BasicCondAEOperator(f'{self.file_name}{irun}', model_3d, train_3d, shuffle=True, 
                                                            batch_size=batch_size, 
                                                            init_lr=1e-4,
                                                            num_epochs=300,
                                                            ref=self.ref, ref_channels=(None, 2), 
                                                            # recon_channels=(1, None), 
                                                            recon_channels=(3, None), 
                                                            input_channels=(None, self.input_ref),
                                                            split_train_ratio=1.0)

        op.set_scheduler('LambdaLR', lr_lambda=warmup_lr)
        
        op.train_model(save_check=300, v_tqdm=False)

        return op.model


    def func_eval(self, irun, model, itrain, itest):

        # load_model_from_checkpoint(model, epoch=9, folder=os.path.join('save', file_name + str(irun)), device=device)
        # cal_loss(vae_model, errors, fldata, paras, KinkWing, 67)
        errors = []

        # cal_loss(model, errors, self.fldata_3d, paras, Wing, len(self.fldata_3d.inputs), ref=self.ref, 
        #         ref_channels=(None, 2), recon_channels=(1, None), input_channels=(None, self.input_ref), device=device)
        cal_loss(model, errors, self.fldata_3d, paras, Wing, len(self.fldata_3d.inputs), ref=self.ref, 
                ref_channels=(None, 2), recon_channels=(3, None), input_channels=(None, self.input_ref), device=device)
        absarrors = np.array(copy.deepcopy(errors[0]))
        absarrors[:, 3:6] = np.abs(absarrors[:, 6:9] - absarrors[:, 3:6])
        errors_stats = np.vstack((np.mean(np.take(absarrors[:, 0:6], itrain, axis=0), axis=0), np.mean(np.take(absarrors[:, 0:6], itest, axis=0), axis=0)))

        return errors[0], errors_stats

if __name__ == '__main__':

    file_name = 'aoa11_1658_Run'


    tt = Train_and_Test(file_name=file_name, ref=True, input_ref=5, ref_type=2, sa_type=0.25)

    history = K_fold(dataset_len=len(paras), func_train=tt.func_train, func_eval=tt.func_eval, k=10, krun=10, num_train=-1)
    

    torch.save(history, os.path.join('save', file_name + '_history'))

    print(np.array(history['errors']).shape, np.array(history['errstats']).shape)
    print(np.array(history['errstats']))
    # print(history['train_index'], history['test_index'])
    # print(len(history['train_index'][0]), len(history['test_index'][0]))

