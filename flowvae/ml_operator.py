'''
Yang Yunjia @ 20210221

Rewrite the vae operater part of Deng's code

'''
import torch
from torch.utils.data import Subset, DataLoader, random_split
import torch.optim as opt

import sys, os
from tqdm import tqdm

from .vae import frameVAE
from .dataset import ConditionDataset
from .utils import MDCounter
# from .post import _get_vector, _get_force_cl, get_aoa, WORKCOD, _get_force_o
from .post import get_force_1dc


class AEOperator:
    '''

    The operator to run a vae model
    
    initial params:
    ===
    `opt_name`  (str) name of the problem
    `model`     (frameVAE) the model to be trained
    `dataset`   (ConditionDataset)
    `  


    '''
    def __init__(self, opt_name: str, 
                       model: frameVAE, 
                       dataset: ConditionDataset,
                       recon_type='field',
                       input_channels=(None, None), recon_channels=(1, None),
                       num_epochs=50, batch_size=8, 
                       reco_data=False, reco_name='',
                       output_folder="save", 
                       shuffle=True, num_workers=0,
                       init_lr=0.01, 
                       device='cuda:0'):
        
        self.output_folder = output_folder
        self.set_optname(opt_name)
        self.device = device
        self.recon_type = recon_type
        self.channel_markers = (input_channels, recon_channels)
        
        self._model = model
        self._model.to(self.device)   

        self.paras = {}
        self.paras['num_epochs'] = num_epochs
        self.paras['batch_size'] = batch_size
        self.paras['num_workers'] = num_workers
        self.paras['init_lr'] = init_lr
        self.paras['shuffle'] = shuffle
        self.paras['code_mode'] = model.paras['code_mode']
        self.paras['loss_parameters'] = {'sm_mode':     'NS',
                                         'sm_epoch':        1, 
                                         'sm_weight':       1, 
                                         'sm_offset':       2,
                                         'moment_weight':   0.0,
                                         'code_weight':     0.1,
                                         'indx_weight':     0.0001,
                                         'indx_epoch':      10,
                                         'indx_pivot':      -1,
                                         'aero_weight':     1e-5,
                                         'aero_epoch':      299}
        
        self.all_dataset = dataset
        # split training data
        self.dataset = {}
        self.split_dataset(recover=reco_data, name=reco_name)
        self.dataloaders = {phase: DataLoader(self.dataset[phase], batch_size=self.paras['batch_size'], shuffle=self.paras['shuffle'], drop_last=False, num_workers=self.paras['num_workers'], pin_memory=True)
                                for phase in ['train', 'val']}     
        # optimizer
        self._optimizer = None
        self._optimizer_name = 'Not set'
        self._optimizer_setting = None
        self._scheduler = None
        self._scheduler_name = 'Not set'
        self._scheduler_setting = None

        # runing parameters
        self.history = {'loss': {'train':MDCounter(), 'val':MDCounter()}, 'lr':[]}
        self.epoch = 0
        self.best_loss = 1e4       

        self.initial_oprtor()

        self.set_optimizer('Adam', lr=self.paras['init_lr'])
        # self.set_scheduler('ReduceLROnPlateau', mode='min', factor=0.1, patience=5)


    @property
    def model(self):
        return self._model

    def set_lossparas(self, **kwargs):
        '''

        set the parameters for loss terms when training, the keys include:\n
              key          default         info
            sm_mode         'NS'    the mode to calculate smooth loss
        >>>>> `NS`:     use navier-stokes equation's conservation of mass and moment to calculate smooth\n
        >>>>> `NS_r`    use the residual between mass/moment flux of reconstructed and ground truth as loss\n
        >>>>> `offset`  offset diag. the field for several distance and sum the difference between field before and after move\n
        >>>>> `1d`      one-dimensional data (adapted from Runze)\n
            sm_epoch        1       (int) the epoch to start counting the smooth loss 
            sm_weight       0.001   (float)the weight of the smooth loss 
            sm_offset       2       (int) (need for `offset`) the diag. direction move offset
            moment_weight   0.0     (float) (need for `NS`, `NS_r`) the weight of momentum flux residual
            code_weight     0.1     (float) the weight of code loss, better for 0.1         
            indx_weight     0.0001  (float) the weight of index KLD loss  
            indx_epoch      10      (int) the epoch to start counting the index KLD loss 
            indx_pivot      -1      (int) the method to get avg_latent value
        >>>>>  if is a positive number, use the latent value of that condition index\n
        >>>>>  if `-1`, use the averaged value of all condition index
            aero_weight     1e-5    (float) the weight of aerodynamic loss
            aero_epoch      299     (int) the epoch to start counting the aero loss 

        '''
        for key in kwargs:
            self.paras['loss_parameters'][key] = kwargs[key]

    def set_optname(self, opt_name):
        self.optname = opt_name

        self.output_path = self.output_folder + "/" + opt_name

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        print('data will be saved in ' + self.output_path)
    
    def initial_oprtor(self, init_model=True, init_optimitor=True):
        '''
        initial operator: things below will be done:
        1. clear history of training, set epoch current epoch to 0
        2. initial model
        3. initial optimizer (initial scheduler at the same time in func. set_optimizer)
        4. set best loss to a BIG number
        '''
        self.history = {'loss': {'train':MDCounter(), 'val':MDCounter()}, 'lr':[]}
        self.epoch = 0
        if init_model and self._model is not None:
            self._model.apply(reset_paras)
        if init_optimitor and self._optimizer is not None:
            self.set_optimizer(self._optimizer_name, **self._optimizer_setting)
        self.best_loss = 1e4

    def set_optimizer(self, optimizer_name, **kwargs):
        if optimizer_name not in dir(opt):
            raise AttributeError(optimizer_name + ' not a vaild optimizer name!')
        else:
            print('optimizer Using Method: ' + optimizer_name, kwargs)
            opt_class = getattr(opt, optimizer_name)
            self._optimizer_name = optimizer_name
            self._optimizer_setting = kwargs
            self._optimizer = opt_class(self._model.parameters(), **kwargs)
        
        if self._scheduler is not None:
            self.set_scheduler(self._scheduler_name, **self._scheduler_setting)
        
    def set_scheduler(self, scheduler_name, **kwargs):
        if scheduler_name not in dir(opt.lr_scheduler):
            raise AttributeError(scheduler_name + ' not a vaild pressure method!')
        else:
            print('scheduler Using Method: ' + scheduler_name, kwargs)
            sch_class = getattr(opt.lr_scheduler, scheduler_name)
            self._scheduler_name = scheduler_name
            self._scheduler_setting = kwargs
            self._scheduler = sch_class(self._optimizer, **kwargs)
    
    def train_model(self, save_check=20, save_best=True, v_tqdm=True):

        if self.paras['code_mode'] in ['ex', 'ae', 'ved']:
            print('*** set code, indx, coef loss to ZERO')
            self.paras['loss_parameters']['code_weight'] = 0.0
        elif self.paras['code_mode'] in ['ed']:
            print('*** set code, indx, coef loss to ZERO')
            self.paras['loss_parameters']['code_weight'] = 0.0
            self.paras['loss_parameters']['indx_weight'] = 0.0
            self.paras['loss_parameters']['aero_weight'] = 0.0

        if os.path.exists('data/preprocessdata'):
            self._model.geom_data = torch.load('data/preprocessdata', map_location=self.device)
        else:
            self._model.preprocess_data(self.all_dataset)


        #* *** Training section ***
        print(' === ========Training begin========= ===                                    loss  recons KLD   smooth')
        # reference marker
        _is_ref = int(self.all_dataset.isref)
        # input channel marker
        ipt1, ipt2 = self.channel_markers[0]
        # reconstruction type and markers
        recon_type = self.recon_type
        if recon_type in ['1d-clcd']:
            all_x = self.model.geom_data['xx']
        elif recon_type in ['field']:
            rst1, rst2 = self.channel_markers[1]

        while self.epoch < self.paras['num_epochs']:

            print('Epoch {}/{}'.format(self.epoch, self.paras['num_epochs'] - 1))

            # torch.autograd.set_detect_anomaly(True)
            # 每个epoch都有一个训练和验证阶段
            for phase in ['train', 'val']:
                if phase == 'train':
                    self._model.train()  # Set model to training mode
                else:
                    self._model.eval()   # Set model to evaluate mode

                running_loss = MDCounter()

                # 迭代数据.
                sys.stdout.flush()
                with tqdm(self.dataloaders[phase], ncols = 100) as tbar:
                    if not v_tqdm:
                        tbar = self.dataloaders[phase]
                    
                    for batch_data in tbar:
                        
                        real_field = batch_data['flowfields'].to(self.device)
                        refs_field = batch_data['ref'].to(self.device)
                        # the size of real_field and refs_field should be same, and both have geometry field if given

                        indxs = batch_data['index'].reshape((-1,))
                        cods = batch_data['code_index'].reshape((-1,))
                        real_labels = batch_data['condis'].reshape((-1, self._model.code_dim)).to(self.device)
                        delt_labels = real_labels - batch_data['ref_aoa'].reshape((-1, self._model.code_dim)).to(self.device) * _is_ref

                        # 零参数梯度
                        self._optimizer.zero_grad()

                        # 前向, track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            
                            #* model forward process
                            if self.paras['code_mode'] in ['ex', 'semi', 'ae']:
                                result_field = self._model(real_field[:, ipt1: ipt2], code=delt_labels)
                            elif self.paras['code_mode'] in ['ed', 'ved']:
                                result_field = self._model(refs_field[:, ipt1: ipt2], code=delt_labels)
                            elif self.paras['code_mode'] in ['im']:
                                result_field = self._model(real_field[:, ipt1: ipt2])

                            #* calculating loss
                            # update loss weight for current epoch
                            for nn in ['sm', 'indx', 'aero']:
                                self.paras['loss_parameters'][nn + '_weight'] = (
                                    0.0, self.paras['loss_parameters'][nn + '_weight'])[self.epoch >= self.paras['loss_parameters'][nn + '_epoch']]
                            
                            if recon_type == 'field':
                                # A. reconstruct field
                                real = real_field[:, rst1: rst2]
                                ref  = refs_field[:, rst1: rst2] * _is_ref
                            elif recon_type == '1d-clcd':
                                # B. reconstruct aerodynamic coefficients
                                geom = torch.cat((all_x.repeat(real_field.size(0), 1).unsqueeze(1), real_field[:, 0].unsqueeze(1)), dim=1)
                                profile = real_field[:, 1]
                                real = get_force_1dc(geom, profile, real_labels.squeeze(), dev=self.device) # squeeze is for code dimension, since only the aoa data is need
                                # TODO this should be done once at beginning
                                # ref  = torch.zeros_like(real, device=self.device)
                                ref = self.model.geom_data['ref_clcd'].index_select(0, indxs) * _is_ref

                            if self.paras['code_mode'] in ['ed', 'ved']:
                                # the vae without prior index loss (must be ed mode)
                                # if not reference, the parameter ref is set to zero by multiply with the flag
                                loss_dict = self._model.loss_function(real, *result_field,
                                                                       ref=ref,
                                                                       **self.paras['loss_parameters'])
                            else:
                                loss_dict = self._model.series_loss_function(real, *result_field, 
                                                                             ref=ref,
                                                                             real_code=delt_labels,
                                                                             real_code_indx=cods,
                                                                             real_indx=indxs,
                                                                             **self.paras['loss_parameters'])
                            loss = loss_dict['loss']

                            if not torch.isnan(loss).sum() == 0:
                                print("NAN")
                                raise Exception()
                            
                            if v_tqdm:
                                tbar.set_postfix(loss=loss.item(), smt=loss_dict['smooth'].item(), reco=loss_dict['recons'].item(), code=loss_dict['code'].item())
                            
                            #* model backward process (only in training phase)
                            if phase == 'train':
                                loss.backward()
                                self._optimizer.step()

                        # loss.item() gives the value of the tensor, multiple in the later is for weighted average
                        running_loss += MDCounter(loss_dict) * real_field.size(0)

                epoch_loss = running_loss / self.dataset_size[phase]
                
                if phase == 'train':
                    self.history['lr'].append(self._optimizer.param_groups[0]['lr'])
                    if self._scheduler_name in ['ReduceLROnPlateau']:
                        self._scheduler.step(epoch_loss['loss'])
                    else:
                        self._scheduler.step()

                print(' loss: %.5f %.5f %.5f %.5f %.5f %.5f' % (epoch_loss['loss'], epoch_loss['recons'], epoch_loss['code'],
                                                      epoch_loss['indx'], epoch_loss['smooth'], epoch_loss['aero']))

                self.history['loss'][phase].append(epoch_loss)

                # 深度复制mo
                if phase == 'val' and epoch_loss['loss'] < self.best_loss:
                    self.best_loss = epoch_loss['loss']
                    if save_best:
                        self.save_model()
            
            
            if self.paras['code_mode'] in ['ex', 'semi', 'im', 'ae']:
                self._model.cal_avg_latent(pivot=self.paras['loss_parameters']['indx_pivot'])
            
            if (self.epoch + 1) % save_check == 0:
                self.save_checkpoint(epoch=self.epoch)

            self.epoch += 1

        print('Best val Loss: {:4f}'.format(self.best_loss))

        

    def save_model(self, name='best_model'):
        path = self.output_path + '/' + name
        torch.save(self._model.state_dict(), path)
    
    def save_checkpoint(self, epoch, name='checkpoint'):
        path = self.output_path + '/' + name + '_epoch_' + str(epoch)
        torch.save({'epoch': epoch, 
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                    'scheduler_state_dict': self._scheduler.state_dict(),
                    'history': self.history,
                    'series_data': self._model.series_data,
                    'geom_data': self._model.geom_data}, path)
        print('checkpoint saved to' + path)
    
    def load_model(self, name, fro='c'):
        if fro == 'd':
            saved_model = name
        elif fro == 'c':
            path = self.output_path + '/' + name
            saved_model = torch.load(path, map_location=self.device)['model_state_dict']
        else:
            raise AttributeError
        
        self._model.load_state_dict(saved_model, strict=False)

    def load_checkpoint(self, epoch, name='checkpoint', load_opt=True):
        path = self.output_path + '/' + name + '_epoch_' + str(epoch)
        if not os.path.exists(path):
            raise IOError("checkpoint not exist in {}".format(self.output_path))
        save_dict = torch.load(path, map_location=self.device)
        self.epoch = save_dict['epoch'] + 1
        self._model.load_state_dict(save_dict['model_state_dict'], strict=False)
        if load_opt:
            self._optimizer.load_state_dict(save_dict['optimizer_state_dict'])
            self._scheduler.load_state_dict(save_dict['scheduler_state_dict'])
        self.history = save_dict['history']
        self._model.series_data = save_dict['series_data']
        self._model.geom_data = save_dict['geom_data']

        self.split_dataset(recover=True, name=self.optname)
        print('checkpoint loaded from' + path)


    def split_dataset(self, recover, name, test_r=0.0):
        self.dataset_size = {}

        if recover:
            path = self.output_folder + '//' + name + '//' + 'dataset_indice'
            if not os.path.exists(path):
                raise IOError("checkpoint not exist in {}".format(self.output_folder))
            dataset_dict = torch.load(path, map_location=self.device)
            for phase in ['train', 'val', 'test']:
                self.dataset[phase] = Subset(self.all_dataset, dataset_dict[phase])
                self.dataset_size[phase] = len(dataset_dict[phase])

            path = self.output_path + '//'  + 'dataset_indice'
            if os.path.exists(path):
                print("Press any key to recover exist dataset division:")
            torch.save({phase: self.dataset[phase].indices for phase in ['train', 'val', 'test']}, path)
            
            print("Load Split Dataset length: (Train: %d, Val: %d, Test: %d)" % (self.dataset_size['train'], self.dataset_size['val'], self.dataset_size['test']))

        else:
            train_val_size = int((1.0 - test_r) * len(self.all_dataset))
            self.dataset_size['test'] = len(self.all_dataset) - train_val_size
            train_val_dataset, self.dataset['test'] = random_split(self.all_dataset, [train_val_size, self.dataset_size['test']])

            
            self.dataset_size['train'] = int(0.9 * train_val_size)
            self.dataset_size['val'] = train_val_size - self.dataset_size['train']
            self.dataset['train'], self.dataset['val'] = random_split(train_val_dataset, [self.dataset_size['train'], self.dataset_size['val']])

            path = self.output_path + '//'  + 'dataset_indice'

            torch.save({phase: self.dataset[phase].indices for phase in ['train', 'val', 'test']}, path)
            # torch.save({'train': self.train_dataset.indices, 'val': self.val_dataset.indices, 'test': self.test_dataset.indices}, path)
            print("Random Split Dataset length: (Train: %d, Val: %d, Test: %d)" % (self.dataset_size['train'], self.dataset_size['val'], self.dataset_size['test']))

def reset_paras(layer):
    if 'reset_parameters' in dir(layer):
        layer.reset_parameters()
