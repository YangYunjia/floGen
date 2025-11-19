'''
Yang Yunjia @ 20210221

Rewrite the vae operater part of Deng's code

'''
import torch
from torch import nn
from torch.nn.modules import Module
from torch.utils.data import Subset, DataLoader, random_split
import torch.optim as opt

import sys, os
import numpy as np
from tqdm import tqdm
from typing import List, Union, Any, NewType, Tuple, Optional, Dict

from flowvae.vae import EncoderDecoder
from flowvae.dataset import ConditionDataset, FlowDataset
# from .post import _get_vector, _get_force_cl, get_aoa, WORKCOD, _get_force_o
from .ema import EmaGradClip
from ..utils import MDCounter
from .lora import add_lora_to_model, enable_gradient
from flowvae.post import get_xyforce_2d_t

is_conflictfree_import = True
try:
    from conflictfree.grad_operator import ConFIG_update
    from conflictfree.utils import get_gradient_vector,apply_gradient_vector
except ImportError:
    is_conflictfree_import = False

def _check_existance_checkpoint(epoch, folder):

    if epoch == -1:
        paths = os.path.join(folder, 'best_model')
    else:
        paths = os.path.join(folder, 'checkpoint_epoch_' + str(epoch))
    if not os.path.exists(paths):
        raise IOError("checkpoint not exist in {}".format(paths))
    return paths

def load_model_from_checkpoint(model: nn.Module, epoch: int, folder: str, device: str, set_to_eval: bool = True):
    '''
    loading trained model from saved checkpoints

    ### paras

    - `model`:  (`nn.Module`) The model that need to be load
        the model will not be returned, just use it after this function
    - `epoch`:  (`int`) The epoch number want load
    - `folder`: (`str`) The folder of saved checkpoints
        the model will be load from `<folder>\checkpoint_epoch_<epoch>`
    - `device`: (`str`) Device to load the model
    - `set_to_eval`:    (`bool`, default = `True`) Whether set the model to `.eval()` for evaluation
    
    
    '''
    model.to(device)
    path = _check_existance_checkpoint(epoch=epoch, folder=folder)
    save_dict = torch.load(path, map_location=device, weights_only=False)
    
    if 'history' not in save_dict.keys():
    # if epoch == -1:
        model.load_state_dict(save_dict, strict=True)
        last_error = None
    else:
        model.load_state_dict(save_dict['model_state_dict'], strict=True)
        last_error = save_dict['history']['loss']
#     print('loss of last iter.:  train, vali = %.4e  %.4e' % (last_error['train']['loss'][-1], last_error['val']['loss'][-1]))
    if set_to_eval: model.eval()

    print(f'load checkpoint from {path}')

    return last_error



class ModelOperator():
    '''
    simple operator class for universial models
    
    ### paras

    - `opt_name`:   (`str`)   name of operator, will be used as the folder name of save files
    - `model`:  (`nn.Module`)   model to be trained
    - `dataset`:    (`FlowDataset` or `ConditionDataset`)   dataset to train the model
    - `output_folder`:  the folder to store save folders, **must exist**
    - `init_lr`:    (`float`, default = 0.01)   learning rate at first epoch
    - `num_epoch`:  (`int`, default = 50)   number of epochs to train the model
    - `split_train_ratio`:  (`float`, default = 0.9)    the ratio of training and validation samples for training
        - if `<= 0.`, the validation step will not take place
    - `recover_split`:  (`str`, default = `None`)   if is not `None`, the split index of train & validation will be
            recovered from the file
    - `batch_size`: (`int`, default = `8`)  batch size
    - `shuffle`:    (`bool`, default = `True`)  shuffle
    
    '''
    def __init__(self, opt_name: str, 
                       model: nn.Module, 
                       dataset: Union[FlowDataset, ConditionDataset],
                       output_folder: str = "save", 
                       init_lr: float = 0.01, 
                       num_epochs: int = 50,
                       split_train_ratio: float = 0.9,
                       split_dataset: Optional[Dict[str, Subset]] = None,
                       recover_split: str = None,
                       batch_size: int = 8, 
                       shuffle: bool = True,
                       num_workers: int = 4,
                       ema_optimizer: bool = False
                       ) -> None:
        
        self.output_folder = output_folder
        self.set_optname(opt_name)
        self.device = model.device
        
        self._model = model
        # self._model.to(self.device)
        self._transfer_output_bias = None

        self.paras = {}
        self.paras['num_epochs'] = num_epochs
        self.paras['batch_size'] = batch_size
        self.paras['num_workers'] = num_workers
        # self.paras['init_lr'] = init_lr
        self.paras['shuffle'] = shuffle
        self.paras['loss_parameters'] = {}
        
        self.dataset = {}
        if dataset is not None:
            self.all_dataset = dataset
            if split_dataset is not None:
                # use the given split dataset
                assert isinstance(split_dataset, dict) and set(split_dataset.keys()) == set(['train', 'val']), \
                    'Wrong input with `split`, it should be a dict with keys = `train`, `val'
                self.dataset = split_dataset
                self.phases = ['train', 'val']
                print("Dataset split based on input: (Train: %d, Val: %d)" % (len(self.dataset['train']), len(self.dataset['val'])))
                
            else:
                # split training data
                self.phases = self.split_train_valid_dataset(recover=recover_split, train_r=split_train_ratio)
            
            self.dataset_size = {phase: len(self.dataset[phase]) for phase in self.phases}
                
            
            self.dataloaders = {phase: DataLoader(self.dataset[phase], 
                                                  batch_size=self.paras['batch_size'], 
                                                  shuffle=self.paras['shuffle'], 
                                                  drop_last=False, 
                                                  num_workers=self.paras['num_workers'], 
                                                  pin_memory=True)
                                    for phase in self.phases}     
        # optimizer
        self._optimizer: torch.optim.Optimizer = None
        self._optimizer_name = 'Not set'
        self._optimizer_setting = {
            'lr': init_lr,
            'ema_optimizer': ema_optimizer,
        }
        self._scheduler = None
        self._scheduler_name = 'Not set'
        self._scheduler_setting = None

        # runing parameters
        self.history = {'loss': {'train':MDCounter(), 'val':MDCounter()}, 'lr':[]}
        self.epoch = 0
        self.best_loss = 1e4       

        self.initial_oprtor()

        self.set_optimizer('Adam', **self._optimizer_setting)
        # self.set_scheduler('ReduceLROnPlateau', mode='min', factor=0.1, patience=5)
        print("network have {} paramerters in total".format(sum(x.numel() for x in self.model.parameters())))

        # parameters controling loss
        self.paras['loss_parameters'] = {'conFIG': False,
                                         'aero_weight':     0.,
                                         'aero_epoch':      0.2}    # default 0.1 before 2023.8.4

    def set_lossparas(self, **kwargs):
        '''

        set the parameters for loss terms when training, the keys include:
        
        >>> key          default         info
        
        - `sm_mode`         'NS'    the mode to calculate smooth loss
            - `NS`:     use navier-stokes equation's conservation of mass and moment to calculate smooth\n
            - `NS_r`    use the residual between mass/moment flux of reconstructed and ground truth as loss\n
            - `offset`  offset diag. the field for several distance and sum the difference between field before and after move\n
            - `1d`      one-dimensional data (adapted from Runze)\n
        - `sm_epoch`        1       (int) the epoch to start counting the smooth loss 
        - `sm_weight`       0.001   (float)the weight of the smooth loss 
        - `sm_offset`       2       (int) (need for `offset`) the diag. direction move offset
        - `moment_weight`   0.0     (float) (need for `NS`, `NS_r`) the weight of momentum flux residual
        - `code_weight`     0.1     (float) the weight of code loss, better for 0.1         
        - `indx_weight`     0.0001  (float) the weight of index KLD loss  
        - `indx_epoch`      10      (int) the epoch to start counting the index KLD loss 
        - `indx_pivot`      -1      (int) the method to get avg_latent value
            - if is a positive number, use the latent value of that condition index\n
            - if `-1`, use the averaged value of all condition index
        - `aero_weight `    1e-5    (float) the weight of aerodynamic loss
        - `aero_epoch`      299     (int) the epoch to start counting the aero loss 

        '''
        print('------------------------------------')
        print('the following parameters are updated:')
        for key in kwargs:
            if key in self.paras['loss_parameters'].keys():
                self.paras['loss_parameters'][key] = kwargs[key]
            else:
                raise NotImplementedError(f'\'{key}\' not implemented in the current operator' )

        if self.paras['loss_parameters']['conFIG']:
            if not is_conflictfree_import:
                raise ImportError('conFIG not installed!')
            else:
                print(f'> ConFig is actived, all actived weights are set with 1.0 and epoch 0.0')
                for key in self.paras['loss_parameters'].keys():
                    if 'weight' in key and self.paras['loss_parameters'][key] > 0.0:
                        self.paras['loss_parameters'][key] = 1.0
                    if 'epoch' in key:
                        self.paras['loss_parameters'][key] = 0

        for key in self.paras['loss_parameters'].keys():
            print(f'> ---  {key} = {self.paras["loss_parameters"][key]}')

    @property
    def model(self):
        return self._model
    
    def set_optname(self, opt_name):
        self.optname = opt_name

        self.output_path = os.path.join(self.output_folder, opt_name)

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        print('data will be saved in ' + self.output_path)
    
    def initial_oprtor(self, init_model=True, init_optimitor=True):
        '''
        initial operator: things below will be done:
        1. clear history of training, set epoch current epoch to 0
        2.0. compile model if needed
        2. initial model
        3. initial optimizer (initial scheduler at the same time in func. set_optimizer)
        4. set best loss to a BIG number
        '''
        self.history = {'loss': {'train':MDCounter(), 'val':MDCounter()}, 'lr':[]}
        self.epoch = 0
        
        if hasattr(self.model, 'is_compiled'):
            load_kwargs, forward_kwargs, loss_kwargs = self._init_training()
            for test_data in self.dataloaders['train']:
                test_data, _ = self._load_batch_data(test_data, load_kwargs)
                input_args, input_kwargs = self._forward_model(test_data, forward_kwargs)
                break
            self.model.compile(*input_args, **input_kwargs)
            self.model.to(self.device)   
            
        if init_model and self._model is not None:
            self._model.apply(reset_paras)
        if init_optimitor and self._optimizer is not None:
            self.set_optimizer(self._optimizer_name, **self._optimizer_setting)
        self.best_loss = 1e4

    def set_optimizer(self, optimizer_name, ema_optimizer, **kwargs):
        if optimizer_name not in dir(opt):
            raise AttributeError(optimizer_name + ' not a vaild optimizer name!')
        else:
            print('optimizer Using Method: ' + optimizer_name, kwargs)
            opt_class = getattr(opt, optimizer_name)
            self._optimizer_name = optimizer_name
            self._optimizer_setting = kwargs
            self.clip_active = False
            if isinstance(ema_optimizer, bool) and ema_optimizer:
                self.clip_active = 1
                self.ema_clip = EmaGradClip(ema_coef1=0.99, ema_coef2=0.999)
            elif isinstance(ema_optimizer, float):
                self.clip_active = 0
                self.clip_max_norm = ema_optimizer
            else:
                raise AttributeError()
            self._optimizer = opt_class(self._model.parameters(), **kwargs)
            self._optimizer_setting['ema_optimizer'] = ema_optimizer

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
    
    def split_train_valid_dataset(self, recover: Optional[str] = None, train_r: float = 0.9):
        '''
        Split the validation set from the training one; the testing dataset should 
        be split before training use functions in `FlowDataset` classes
        
        '''

        if recover is not None:
            path = os.path.join(self.output_folder, recover, 'dataset_indice')
            if not os.path.exists(path):
                raise IOError("checkpoint not exist in {}".format(self.output_folder))
            dataset_dict = torch.load(path, map_location=self.device)
            for phase in dataset_dict.keys():
                self.dataset[phase] = Subset(self.all_dataset, dataset_dict[phase])

            # path = self.output_path + '//'  + 'dataset_indice'
            # if os.path.exists(path):
            #     print("Press any key to recover exist dataset division:")
            # torch.save({phase: self.dataset[phase].indices for phase in ['train', 'val', 'test']}, path)
            
            print("Load Split Dataset length: (Train: %d, Val: %d)" % (len(self.dataset['train']), len(self.dataset['val'])))

        else:
            
            if train_r >= 1.0:
                self.dataset['train'] = self.all_dataset
                print("All dataset is used for training: (Train: %d, Val: %d)" % (len(self.all_dataset), 0))
            
            else:
            
                train_size = int(train_r * len(self.all_dataset))
                val_size =   len(self.all_dataset) - train_size
                self.dataset['train'], self.dataset['val'] = random_split(self.all_dataset, [train_size, val_size])
                print("Random Split Dataset length: (Train: %d, Val: %d)" % (train_size, val_size))

            path = os.path.join(self.output_folder, self.optname, 'dataset_indice')
            torch.save({phase: self.dataset[phase].indices for phase in ['train', 'val']}, path)
            # torch.save({'train': self.train_dataset.indices, 'val': self.val_dataset.indices, 'test': self.test_dataset.indices}, path)
            return self.dataset.keys()

    def train_model(self, save_check, save_best=True, v_tqdm=True, update_lr_batch=False):

        load_kwargs, forward_kwargs, loss_kwargs = self._init_training()
        #* *** Training section ***
        print(' === ========Training begin========= ===')

        while self.epoch < self.paras['num_epochs']:

            print('Epoch %d/%d   lr = %.4e' % (self.epoch, self.paras['num_epochs'] - 1, self._optimizer.param_groups[0]['lr']))
            load_kwargs, forward_kwargs, loss_kwargs = self._start_of_epoch(load_kwargs, forward_kwargs, loss_kwargs)
            # torch.autograd.set_detect_anomaly(True)
            for phase in self.phases:
                if phase == 'train':
                    self._model.train()  # Set model to training mode
                else:
                    self._model.eval()   # Set model to evaluate mode

                running_loss = MDCounter()
                # Iteration of dataset
                sys.stdout.flush()
                with tqdm(self.dataloaders[phase], ncols = 100) as tbar:
                    if not v_tqdm:
                        tbar = self.dataloaders[phase]
                    
                    for batch_data in tbar:

                        data, batch_size = self._load_batch_data(batch_data, load_kwargs)

                        # forward, track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):

                            input_args, input_kwargs = self._forward_model(data, forward_kwargs)
                            output = self._model(*input_args, **input_kwargs)
                            output = self._apply_transfer_output_bias(output)
                            
                            # print(output.size(), data['label'].size())
                            loss_dict = self._calculate_loss(data, output, loss_kwargs)

                            if 'loss' not in loss_dict.keys():
                                loss_values = list(loss_dict.values())
                                total_loss = sum(loss_values)
                            else:
                                total_loss = loss_dict['loss']
                            
                            loss_dict['loss'] = total_loss.item()

                            if not torch.isnan(total_loss).sum() == 0:
                                print("NAN")
                                raise Exception()
                            
                            if v_tqdm:
                                tbar.set_postfix(loss=loss_dict['loss'])
                            
                            # print(self.model.decoder_input._parameters['weight'].requires_grad, self.model.fc_mu._parameters['weight'].requires_grad)
                            #* model backward process (only in training phase)
                            if phase == 'train':

                                if self.paras['loss_parameters']['conFIG']:# and len(loss_values) > 1:
                                    grads=[]
                                    n_loss = len(loss_values)
                                    for i, loss_i in enumerate(loss_values):
                                        self._optimizer.zero_grad()
                                        loss_i.backward(retain_graph = (i < n_loss - 1))
                                        grads.append(get_gradient_vector(self._model)) # get loss-specfic gradient
                                    g_config = ConFIG_update(grads) # calculate the conflict-free direction
                                    apply_gradient_vector(self._model, g_config) # set the conflict-free direction to the network
                                else:
                                    self._optimizer.zero_grad()
                                    total_loss.backward()

                                if self.clip_active == 1:
                                    self.ema_clip.on_before_optimizer_step(self._model)
                                elif self.clip_active == 0:
                                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=self.clip_max_norm)
                                self._optimizer.step()
                                if update_lr_batch:
                                    self._scheduler.step()

                        # loss.item() gives the value of the tensor, multiple in the later is for weighted average
                        running_loss += MDCounter(loss_dict) * batch_size
                        
                epoch_loss = running_loss / self.dataset_size[phase]
                
                if phase == 'train':
                    self.history['lr'].append(self._optimizer.param_groups[0]['lr'])
                    if not update_lr_batch:
                        if self._scheduler_name in ['ReduceLROnPlateau']:
                            self._scheduler.step(epoch_loss['loss'])
                        else:
                            self._scheduler.step()

                output_str = ''
                for key in epoch_loss.keys():
                    output_str += '%s=%.5f ' % (key, epoch_loss[key])
                print(output_str)

                self.history['loss'][phase].append(epoch_loss)

                if phase == 'val' and epoch_loss['loss'] < self.best_loss:
                    self.best_loss = epoch_loss['loss']
                    if save_best:
                        self.save_checkpoint(epoch=self.epoch, name='best_model')

            self._end_of_epoch()
            if (self.epoch + 1) % save_check == 0:
                self.save_checkpoint(epoch=self.epoch, name='checkpoint_epoch_' + str(self.epoch))
            self.epoch += 1

        print('Best val Loss: {:4f}'.format(self.best_loss))

    def _init_training(self):
        load_kwargs = {}
        forward_kwargs = {}
        loss_kwargs = {}
        return load_kwargs, forward_kwargs, loss_kwargs

    def _start_of_epoch(self, load_kwargs, forward_kwargs, loss_kwargs):
        return load_kwargs, forward_kwargs, loss_kwargs

    def _load_batch_data(self, batch_data, kwargs):
        for key in batch_data.keys():
            batch_data[key] = batch_data[key].to(self.device)
            batch_size = batch_data[key].size(0)

        return batch_data, batch_size

    def _forward_model(self, data, kwargs):
        return [data['input']], {}

    def _calculate_loss(self, data, output: torch.Tensor, kwargs):
        # print(output.shape, data['label'].shape)
        recons = torch.nn.functional.mse_loss(output, data['label'])
        loss = {'recons': recons}
        
        if self.paras['loss_parameters']['aero_weight'] > 0. and self.epoch >= self.paras['loss_parameters']['aero_epoch'] * self.paras['num_epochs']:

            cp = output[:, 0]
            cf = torch.concat((output[:, [1]].permute(0, 2, 3, 1) * data['geom_t2d'] / 150, output[:, [2]].permute(0, 2, 3, 1) / 300), axis=-1)

            forces = get_xyforce_2d_t([data['geom_n3d'], data['geom_a']], cp, cf)[:, :2]
            forces_loss = torch.nn.functional.mse_loss(forces, data['force']) * self.paras['loss_parameters']['aero_weight']
            loss['force'] = forces_loss

        return loss
    
    def _end_of_epoch(self):
        pass

    def save_checkpoint(self, epoch, name='best_model', save_dict: dict = {}):
        path = os.path.join(self.output_path, name)
        bias_to_save = None
        if self._transfer_output_bias is not None:
            bias_to_save = self._transfer_output_bias.detach().to('cpu')
        save_dict.update({
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler_state_dict': self._scheduler.state_dict(),
            'history': self.history,
            'transfer_output_bias': bias_to_save
        })
        torch.save(save_dict, path)
        print('checkpoint saved to' + path)
    
    def load_checkpoint(self, epoch, load_opt=True, load_data_split=True):

        path = _check_existance_checkpoint(epoch, self.output_path)
        
        save_dict = torch.load(path, map_location=self.device)
        # print(save_dict['epoch'])
        self.epoch = save_dict['epoch'] + 1
        self._model.load_state_dict(save_dict['model_state_dict'], strict=True)

        if load_opt:
            self._optimizer.load_state_dict(save_dict['optimizer_state_dict'])
            self._scheduler.load_state_dict(save_dict['scheduler_state_dict'])
        self.history = save_dict['history']

        load_bias = save_dict.get('transfer_output_bias', None)
        if load_bias is None:
            self._transfer_output_bias = None
        else:
            self._transfer_output_bias = load_bias.detach().clone().to(self.device)

        if load_data_split and len(self.dataset) > 0:
            self.split_train_valid_dataset(recover=self.optname)

        print('checkpoint at epoch %d loaded from %s' % (save_dict['epoch'], path))

        return save_dict

    def set_transfer_model(self, transfer_name, grad_require_layers=None, grad_require_parents=None,
                           reset_param=True,
                           is_lora:bool=False, lora_params={}):
        '''
        load base model and set part of the parameters with grad off
        ['fc_code', 'decoder_input']

        '''
        self.global_start_epoch = self.epoch
        self.epoch = 0
        self.set_optname(self.optname + '_' + transfer_name)

        if grad_require_layers is not None:
            
            if not is_lora:

                for param in self.model.parameters():
                    param.requires_grad = False

                enable_gradient(self.model, grad_require_layers, grad_require_parents, reset_param)

                print('------- The layers below are set grad require ------')
                for n, p in self._model.named_parameters():
                    if p.requires_grad:
                        print(n, p.shape)

            else:

                for p in self.model.parameters():
                    p.requires_grad = False

                print('------- Lora applied, layers below are set grad require ------')
                default_lora_params = {
                    'r': 4,
                    'alpha': 'r', 
                    'dropout': 0.05,
                    'is_lora_k': False,
                }
                for k in lora_params:
                    default_lora_params[k] = lora_params[k]

                if default_lora_params['alpha'] == 'r':
                    default_lora_params['alpha'] = default_lora_params['r']
                elif default_lora_params['alpha'] == '2r':
                    default_lora_params['alpha'] = 2 * default_lora_params['r']

                add_lora_to_model(self._model, grad_require_layers, grad_require_parents, **default_lora_params)

                # set optimizer and scheduler again to register the lora parameters
                self.set_optimizer(self._optimizer_name, **self._optimizer_setting)
                self.set_scheduler(self._scheduler_name, **self._scheduler_setting)

                for n, p in self._model.named_parameters():
                    if p.requires_grad:
                        print(n, p.shape)
            
        else:

            print('------- All layers are set grad require ------')

        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        print(f' > trainable parameters amount: {trainable_params}')

    def set_transfer_output_bias(self, bias: Union[np.ndarray, torch.Tensor]):
        '''
        Assign a constant bias to model outputs after loading a transfer model.
        The bias will be added to every prediction during fine-tuning.

        Args:
            bias: None, scalar, numpy array, or torch.Tensor representing the offset.
        '''
        formatted_bias = self._format_transfer_output_bias(bias)
        self._transfer_output_bias = formatted_bias
        if formatted_bias is None:
            print('transfer output bias cleared')
        else:
            print(f'set transfer output bias with shape {tuple(formatted_bias.shape)}')

    def _format_transfer_output_bias(self, bias):
        if bias is None:
            return None
        if torch.is_tensor(bias):
            tensor_bias = bias.detach().clone()
        elif isinstance(bias, np.ndarray):
            tensor_bias = torch.from_numpy(bias)
        else:
            tensor_bias = torch.tensor(bias, dtype=torch.float32)
        return tensor_bias.to(self.device)

    def _apply_transfer_output_bias(self, output):
        if self._transfer_output_bias is None:
            return output

        return output + self._transfer_output_bias

class BasicAEOperator(ModelOperator):
    '''
    Auto-encoder operator, the shape for input and output should be same
    - The reference flowfield can be add

    ### paras

    - `ref`:    (`bool`, default = `False`)    if `True`, the input is added to the output, then compare to the ground truth
        
        output of the model:

        >>>      output[:, self.ref_channels[0]: self.ref_channels[1]] 
        >>>     + input[:, self.recon_channels[0]: self.recon_channels[1]]
        >>>     = label

        input to the model: 

        >>>      input[:, self.input_channels[0]: self.input_channels[1]]

    
    ### paras

    - `opt_name`:   (`str`)   name of operator, will be used as the folder name of save files
    - `model`:  (`nn.Module`)   model to be trained
    - `dataset`:    (`FlowDataset` or `ConditionDataset`)   dataset to train the model
    - `output_folder`:  the folder to store save folders, **must exist**
    - `init_lr`:    (`float`, default = 0.01)   learning rate at first epoch
    - `num_epoch`:  (`int`, default = 50)   number of epochs to train the model
    - `split_train_ratio`:  (`float`, default = 0.9)    the ratio of training and validation samples for training
        - if `<= 0.`, the validation step will not take place
    - `recover_split`:  (`str`, default = `None`)   if is not `None`, the split index of train & validation will be
            recovered from the file
    - `batch_size`: (`int`, default = `8`)  batch size
    - `shuffle`:    (`bool`, default = `True`)  shuffle


    
    '''

    def __init__(self, opt_name: str, model: Module, dataset: FlowDataset, 
                 output_folder: str = "save", init_lr: float = 0.01, num_epochs: int = 50, 
                 split_train_ratio: float = 0.9, split_dataset: Optional[Dict[str, Subset]] = None, recover_split: str = None, 
                 batch_size: int = 8, shuffle: bool = True, num_workers: int = 4, ema_optimizer: bool = False,
                 ref: bool = False, ref_channels: Tuple[int] = (None, 2), recon_channels = (None, None), input_channels = (None, None)):
        
        self.ref = ref
        self.ref_channels = ref_channels
        self.recon_channels = recon_channels
        self.input_channels = input_channels
        super().__init__(opt_name, model, dataset, output_folder, init_lr, num_epochs, split_train_ratio, split_dataset, recover_split, batch_size, shuffle, num_workers, ema_optimizer)

    def _forward_model(self, data, kwargs):
        return [data['input'][:, self.input_channels[0]: self.input_channels[1]]], {}
    
    def _calculate_loss(self, data, output, kwargs):
        # print(output[0].size(), data['label'].size())
        # labels = data['label']
        if self.ref: data['label'][:, self.ref_channels[0]: self.ref_channels[1]] -= data['input'][:, self.recon_channels[0]: self.recon_channels[1]]

        return super()._calculate_loss(data, output[0], kwargs)
    
    def _apply_transfer_output_bias(self, output):
        if self._transfer_output_bias is None:
            return output
        # print(output[0].shape, self._transfer_output_bias.shape)
        output[0] = output[0] + self._transfer_output_bias
        return output 

class BasicCondAEOperator(BasicAEOperator):
    '''
    Conditional AE operator

    the model has an extra arguments of `code` with `aux` in dataset
    
    
    '''

    def _forward_model(self, data, kwargs):
        # print(data['aux'].size(), data['input'].size())
        return [data['input'][:, self.input_channels[0]: self.input_channels[1]]], {'code':data['aux']}

class AEOperator(ModelOperator):
    '''

    The operator to run a vae model
    
    initial params:
    ===
    `opt_name`  (str) name of the problem
    `model`     (EncoderDecoder) the model to be trained
    `dataset`   (ConditionDataset)
    `  


    '''
    def __init__(self, opt_name: str, 
                       model: EncoderDecoder, 
                       dataset: ConditionDataset,
                       output_folder="save", 
                       init_lr=0.01, 
                       num_epochs: int = 50,
                       split_train_ratio = 0.9,
                       split_dataset: Optional[Dict[str, Subset]] = None,
                       recover_split: str = None,
                       batch_size: int = 8, 
                       shuffle=True,
                       num_workers: int = 4,
                       ema_optimizer: bool = False,
                       ref=False,
                       input_channels: Tuple[int, int] = (None, None), 
                       recon_channels: Tuple[int, int] =(1, None),
                       recon_type: str = 'field'
                       ):
        
        super().__init__(opt_name, model, dataset, output_folder, init_lr, num_epochs, split_train_ratio, split_dataset, recover_split, batch_size, shuffle, num_workers, ema_optimizer)
        
        self.recon_type = recon_type
        # channel markers are not include extra reference channels
        self.channel_markers = (input_channels, recon_channels)
        
        self.paras['ref'] = ref
        self.paras['code_mode'] = model.cm
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
                                         'aero_epoch':      299,
                                         'ge_epoch':        1,
                                         'ge_weight':       0.0}    # default 0.1 before 2023.8.4

    def _init_training(self):
        
        if self.paras['code_mode'] in ['ex', 'ae', 'ved', 'ved1']:
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

        ipt1, ipt2 = self.channel_markers[0]
        rst1, rst2 = self.channel_markers[1]
        # reconstruction type
        if self.recon_type in ['1d-clcd', 'force', 'cltarg', 'aoatarg']:
            assert not self.paras['code_mode'] in ['ex', 'semi', 'ae', 'im'], 'Can not use force model in ex, semi, ae, im'
            
            if self.recon_type in ['1d-clcd', 'force']: info = 'non-dim'    # compatitiy with old version code -> cd, cl (calculated with field data)
            else: info = self.recon_type                                    # 'cltarge' -> cd, AoA (from all_index)
            
            self.all_dataset.change_to_force(info=info)  # change dataset.flowfield to forces 
                                                                # (currently only applied to 1D distribution )
            is_force = True
            
        elif self.recon_type in ['field']:
            erc = self.all_dataset.n_extra_ref_channel
            # input channel marker
            if ipt1 is not None and erc > 0:    raise ValueError('can not add extra reference channel when input channel markers are assigned')
            if ipt2 is not None:    ipt2 += erc
            # reconstruction channel marker
            if rst1 is None:    rst1 = erc
            else:   rst1 += erc
            if rst2 is not None:    rst2 += erc
            is_force = False
        
        # reference marker
        load_kwargs = {'is_ref':    int(self.paras['ref']),
                       'is_force':  is_force}
        forward_kwargs = {'ipt':    (ipt1, ipt2)}
        loss_kwargs = {'is_ref':    int(self.paras['ref']),
                       'rst':       (rst1, rst2),
                       'is_force':  is_force}

        return load_kwargs, forward_kwargs, loss_kwargs

    def _load_batch_data(self, batch_data, kwargs):

        data = {
            'real_field' : batch_data['flowfields'].to(self.device),
            'refs_field' : batch_data['ref'].to(self.device),
            # the size of real_field and refs_field should be same, and both have geometry field if given
            'indxs' : batch_data['index'].reshape((-1,)),
            'cods' : batch_data['code_index'].reshape((-1,)),
            'real_labels' : batch_data['condis'].to(self.device),
            'delt_labels' : batch_data['condis'].to(self.device) - batch_data['ref_aoa'].to(self.device) * kwargs['is_ref']
        }
        if kwargs['is_force']:
            data['ref_force'] = batch_data['ref_force'].to(self.device)

        batch_size = data['real_field'].size(0)
        return data, batch_size

    def _forward_model(self, data, kwargs):
        real_field = data['real_field']
        refs_field = data['refs_field']
        delt_labels = data['delt_labels']
        ipt1, ipt2 = kwargs['ipt']

        if self.paras['code_mode'] in ['ex', 'semi', 'ae']:
            return [real_field[:, ipt1: ipt2]], {'code': delt_labels}
        elif self.paras['code_mode'] in ['ed', 'ved', 'ved1']:
            return [refs_field[:, ipt1: ipt2]], {'code': delt_labels}
        elif self.paras['code_mode'] in ['im']:
            return [real_field[:, ipt1: ipt2]], {}
        
    def _calculate_loss(self, data, output, kwargs):
        weights = {}
        for cm in ['sm', 'indx', 'aero', 'ge']:
            weights[cm + '_weight'] = (
                0.0, self.paras['loss_parameters'][cm + '_weight'])[self.epoch >= self.paras['loss_parameters'][cm + '_epoch']]
        
        real_field = data['real_field']
        refs_field = data['refs_field']

        if kwargs['is_force']:
            # B. reconstruct aerodynamic coefficients
            real = real_field
            ref  = data['ref_force'] * kwargs['is_ref']
        else:
            rst1, rst2 = kwargs['rst']
            # A. reconstruct field
            real = real_field[:, rst1: rst2]
            ref  = refs_field[:, rst1: rst2] * kwargs['is_ref']
            
        if self.paras['code_mode'] in ['ed', 'ved', 'ved1']:
            # the vae without prior index loss (must be ed mode)
            # if not reference, the parameter ref is set to zero by multiply with the flag
            loss_dict = self._model.loss_function(real, *output,
                                                    ref=ref,
                                                    **weights)
        else:
            delt_labels = data['delt_labels']
            cods = data['cods']
            indxs = data['indxs']
            loss_dict = self._model.series_loss_function(real, *output, 
                                                            ref=ref,
                                                            real_code=delt_labels,
                                                            real_code_indx=cods,
                                                            real_indx=indxs,
                                                            **weights)
        return loss_dict
    
    def _end_of_epoch(self):
        if self.paras['code_mode'] in ['ex', 'semi', 'im', 'ae']:
            self._model.cal_avg_latent(pivot=self.paras['loss_parameters']['indx_pivot'])

    def save_checkpoint(self, epoch, name='best_model', save_dict: dict = {}):
        save_dict = {'series_data': self._model.series_data,
                    'geom_data': self._model.geom_data}
        super().save_checkpoint(epoch=epoch, name=name, save_dict=save_dict)
    
    def load_model(self, name, fro='c'):
        if fro == 'd':
            saved_model = name
        elif fro == 'c':
            path = self.output_path + '/' + name
            saved_model = torch.load(path, map_location=self.device)['model_state_dict']
        elif fro == 'cp':
            path = self.output_path + '/' + name + '_epoch_' + str(299)
            save_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(save_dict['model_state_dict'], strict=False)
            self.model.series_data = save_dict['series_data']
            self.model.geom_data = save_dict['geom_data']
        else:
            raise AttributeError
        
        self._model.load_state_dict(saved_model, strict=False)

    def load_checkpoint(self, epoch, load_opt=True, load_data_split=True):

        save_dict = super().load_checkpoint(epoch=epoch, load_opt=load_opt, load_data_split=load_data_split)
        self._model.series_data = save_dict['series_data']
        self._model.geom_data = save_dict['geom_data']

def reset_paras(layer):
    if 'reset_parameters' in dir(layer):
        layer.reset_parameters()
