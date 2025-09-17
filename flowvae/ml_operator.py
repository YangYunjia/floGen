'''
Yang Yunjia @ 20210221

Rewrite the vae operater part of Deng's code

'''
import torch
from torch import nn
from torch.nn.modules import Module
from torch.utils.data import Subset, DataLoader, random_split
import torch.optim as opt
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support#, _device_has_foreach_support
def _device_has_foreach_support(device): return True
import sys, os, random, time, copy
import numpy as np
from tqdm import tqdm
from typing import List, Callable, NewType, Union, Any, NewType, Tuple, Iterable, Optional, Dict
import threading

from flowvae.vae import EncoderDecoder
from flowvae.dataset import ConditionDataset, FlowDataset
# from .post import _get_vector, _get_force_cl, get_aoa, WORKCOD, _get_force_o
from .utils import MDCounter

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
    
    if epoch == -1:
        model.load_state_dict(save_dict, strict=True)
        last_error = None
    else:
        model.load_state_dict(save_dict['model_state_dict'], strict=True)
        last_error = save_dict['history']['loss']
#     print('loss of last iter.:  train, vali = %.4e  %.4e' % (last_error['train']['loss'][-1], last_error['val']['loss'][-1]))
    if set_to_eval: model.eval()
    return last_error

def K_fold(dataset_len: int, func_train: Callable, func_eval: Callable = None, k=10, krun=-1, num_train=-1):
    '''
    K-fold validation training pipline for models. The training and evaluation process is assigned with to
    Callable function, `func_train` and `func_eval`. The current function's major task is to split dataset
    and run the training / evaluating

    ### paras

    - `fldata`: (`dataset`):    the dataset to train the model. The index for each training and evaluating 
    will be selected from this dataset
    - `func_train`: (`Callable`):   the function for training the model. Its inputs and outputs must be 
    exactly the same as follows to run the function
        - **args** for function
            - `irun`:   (`int`)   number of run
            - `train_dataset`: (`dataset`)    the dataset for this run
        - **return** for function
            - `model`: (`nn.Module`)  the trained model
        - remarks
            - the dataset is well set by k-fold function, and in most cases there's no need for validation
            during each training. So, the argument `split_train_ratio` for `ModelOperator` in `func_train` 
            to train the model is Recommonded to set to `1.0`
    - `func_test`: (`Callable`, default = `None`): the function for evaluating the model. Its inputs and 
    outputs must be exactly the same as follows to run the function. If is `None`,  there will not be 
    evaluation
        - **args** for function
            - `irun`:   (`int`)   number of run
            - `model`: (`nn.Module`)  the trained model
            - `fldata`: (`dataset`)    the whole dataset to test
            - `training_indexs`: (`List[int]`) the index number of training samples in `fldata` for this run 
            - `testing_indexs`: (`List[int]`) the index number of samples not involved in training for this run 
        - **return** for function
            - `errors`: (`np.ndarray`) an array of error for each sample, the size should be (`Ni x Nc`) where Ni
            is number of samples, and Nc is number of error values for each sample (can be the used assigned 
            value)
            - `errstats`: (`np.ndarray`) an array of overall error statistics for training and testing sample
            of this run. The size should be (`2 x Nc'`), where 2 stands for train and test, Nc' is also the number
            of error values
    - `k`: (`int`)  number of folds for k-fold run
    - `krun`: (`int`, default `-1` which means the same as `k`) number of folds actually run
    - `number_train`: (`int`, default `-1` which means all samples for each fold) if is a number < size of 
    the training folds for each run, the `number_train` samples will be randomly selected from training folds
    for each run to train the model with.

    ### retures:

    returns is a dictionary with keys:

    - `train_index`:   (`List[List[int]]`) index of training samples in `fldata` for each k-fold run
    - `test_index`:   (`List[List[int]]`) index of not-training samples in `fldata` for each k-fold run
    - `errors`: (`List[np.ndarray]`) a list of all `errors` return by each run
    - `errstats`: (`List[np.ndarray]`) a list of all `errstats` return by each run

    
    
    '''
    
    avg = int(dataset_len / k)
    fold_data_number = [avg+(1, 0)[i > dataset_len - avg * k] for i in range(k)]
    all_dataset_indexs = random.sample(range(dataset_len), dataset_len)

    if krun <= 0:   krun = k

    print('---------------------------------------')
    print('Start K-fold training with k = %d, krun = %d' % (k, krun))

    history = {'train_index': [],
               'test_index': [],
               'errors': [],
               'errstats': [], 
               'train_time': []}

    for irun in range(krun):

        print('---------------------------------------')
        print('')
        print('K-fold Run %d' % irun)

        t0 = time.time()

        idx1 = sum(fold_data_number[:irun])
        idx2 = sum(fold_data_number[:irun+1])
        testing_indexs = all_dataset_indexs[idx1:idx2]
        training_indexs = all_dataset_indexs[:idx1] + all_dataset_indexs[idx2:]

        if num_train > 0 and num_train < len(training_indexs):
            training_indexs = random.sample(training_indexs, num_train)

        history['train_index'].append(training_indexs)
        history['test_index'].append(testing_indexs)
        
        print('    training:  %d from  0 ~ %d, %d ~ end' % (len(training_indexs), idx1, idx2))
        print('    testing :  %d from  %d ~ %d' % (len(testing_indexs), idx1, idx2))

        # training_dataset = Subset(fldata, training_indexs)
        # testing_dataset = Subset(fldata, testing_indexs)

        trained_model = func_train(irun, training_indexs)

        t1 = time.time()

        print('    Training finish in %.2f sec. ' % (t1 - t0))
        history['train_time'].append((t1 - t0))

        if func_eval is not None:
            print('  Evaluating results...')
            trained_model.eval()
            error, errstat = func_eval(irun, trained_model, training_indexs, testing_indexs)
            history['errors'].append(error)
            history['errstats'].append(errstat)

    # if func_eval is not None:
    #     errors   = np.array(errors)
    #     errstats = np.array(errstats)
    return history

def K_fold_evaluate(fldata, model, func_eval, folder, history, device):

    k = len(history['train_index'])

    history['errors'] = []
    history['errstats'] = []

    for irun in range(k):

        load_model_from_checkpoint(model, epoch=299, folder=os.path.join('save', folder + str(irun)), device=device)
        error, errstat = func_eval(irun, model, fldata, history['train_index'][irun], history['test_index'][irun])
        history['errors'].append(error)
        history['errstats'].append(errstat)

    return history

# ----- EMA Gradient Clipper -----
def _clip_grad_norm(
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]], 
        max_norm: float,
        clip_norm: float, 
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False, 
        foreach: Optional[bool] = None) -> torch.Tensor:
    r"""Clip the gradient norm of an iterable of parameters.

    The norm is computed.ema_coef2 over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            falle total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for ((device, _), ([device_grads], _)) in grouped_grads.items():  # type: ignore[assignment]
        if (
            (foreach is None and _has_foreach_support(device_grads, device))
            or (foreach and _device_has_foreach_support(device))
        ):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device) back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    clip_norm = float(clip_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    first_device = grads[0].device
    grouped_grads: Dict[Tuple[torch.device, torch.dtype], Tuple[List[List[torch.Tensor]], List[int]]] \
        = _group_tensors_by_device_and_dtype([grads])  # type: ignore[assignment]

    norms: List[torch.Tensor] = []
    for ((device, _), ([device_grads])) in grouped_grads.items():  # type: ignore[assignment]
        if (
            (foreach is None and _has_foreach_support(device_grads, device))
            or (foreach and _device_has_foreach_support(device))
        ):
            norms.extend(torch._foreach_norm(device_grads, norm_type))
        elif foreach:
            raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_grads])

    total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clipped=total_norm>max_norm
    if clipped:
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_norm / (total_norm + 1e-6), max=1.0)
        for ((device, _), ([device_grads])) in grouped_grads.items():  # type: ignore[assignment]
            if (
                (foreach is None and _has_foreach_support(device_grads, device))
                or (foreach and _device_has_foreach_support(device))
            ):
                torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
            elif foreach:
                raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
            else:
                clip_coef_clamped_device = clip_coef_clamped.to(device)
                for g in device_grads:
                    g.mul_(clip_coef_clamped_device)
    return total_norm, clipped

class EmaGradClip():
    
    def __init__(self,
                 ema_coef1: float = 0.9,
                 ema_coef2: float = 0.99,
                 max_norm_ratio: float = 2.0,
                 clip_norm_ratio: float = 1.1,
                 ):
        super().__init__()
        self.ema_coef1 = ema_coef1
        self.ema_coef2 = ema_coef2
        self.max_norm_ratio = max_norm_ratio
        self.clip_norm_ratio = clip_norm_ratio
        self._grad_norm_ema1=0.0
        self._grad_norm_ema2=0.0
        self.ema_index=0

    def _record_norm(self,new_norm:float):
        self.ema_index+=1
        self._grad_norm_ema1=self.ema_coef1*self._grad_norm_ema1+(1-self.ema_coef1)*new_norm
        self._grad_norm_ema2=self.ema_coef2*self._grad_norm_ema2+(1-self.ema_coef2)*new_norm

    @property
    def _current_ema1(self):
        return self._grad_norm_ema1/(1-self.ema_coef1**self.ema_index)

    @property
    def _current_ema2(self):
        return self._grad_norm_ema2/(1-self.ema_coef2**self.ema_index)
    
    def on_before_optimizer_step(self, pl_module: nn.Module):
        if self._grad_norm_ema2==0.0:
            total_norm, clipped = _clip_grad_norm(pl_module.parameters(), 
                                                         max_norm=10000,
                                                         clip_norm=1,)
        else:
            total_norm, clipped = _clip_grad_norm(
                pl_module.parameters(),
                max_norm=self.max_norm_ratio*self._current_ema2,
                clip_norm=self.clip_norm_ratio*self._current_ema1
            )
        norm=self.clip_norm_ratio*self._current_ema1 if clipped and self.ema_index!=0 else total_norm
        if clipped: print('!', end='')
        self._record_norm(norm)

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
                       recover_split: str = None,
                       batch_size: int = 8, 
                       shuffle: bool = True,
                       ema_optimizer: bool = False
                       ) -> None:
        
        self.output_folder = output_folder
        self.set_optname(opt_name)
        self.device = model.device
        
        self._model = model
        self._model.to(self.device)   

        self.paras = {}
        self.paras['num_epochs'] = num_epochs
        self.paras['batch_size'] = batch_size
        self.paras['num_workers'] = 0
        # self.paras['init_lr'] = init_lr
        self.paras['shuffle'] = shuffle
        self.paras['loss_parameters'] = {}
        
        self.dataset = {}
        if dataset is not None:
            self.all_dataset = dataset
            # split training data
            if split_train_ratio >= 1.0:    self.phases = ['train']
            else:   self.phases = ['train', 'val']
            self.split_dataset(recover=recover_split, train_r=split_train_ratio)
            self.dataloaders = {phase: DataLoader(self.dataset[phase], 
                                                  batch_size=self.paras['batch_size'], 
                                                  shuffle=self.paras['shuffle'], 
                                                  drop_last=False, 
                                                  num_workers=self.paras['num_workers'], 
                                                  pin_memory=True)
                                    for phase in self.phases}     
        # optimizer
        self._optimizer = None
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
            if ema_optimizer:
                self.ema_clip = EmaGradClip(ema_coef1=0.99, ema_coef2=0.999)
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
    
    def split_dataset(self, recover, train_r=0.9, test_r=0.0):
        '''
        Do not use test_r here! 
        
        '''
        self.dataset_size = {}

        if train_r >= 1.0:
            self.dataset['train'] = self.all_dataset
            self.dataset_size['train'] = len(self.all_dataset)
            return

        if recover is not None:
            path = os.path.join(self.output_folder, recover, 'dataset_indice')
            if not os.path.exists(path):
                raise IOError("checkpoint not exist in {}".format(self.output_folder))
            dataset_dict = torch.load(path, map_location=self.device)
            for phase in ['train', 'val', 'test']:
                self.dataset[phase] = Subset(self.all_dataset, dataset_dict[phase])
                self.dataset_size[phase] = len(dataset_dict[phase])

            # path = self.output_path + '//'  + 'dataset_indice'
            # if os.path.exists(path):
            #     print("Press any key to recover exist dataset division:")
            # torch.save({phase: self.dataset[phase].indices for phase in ['train', 'val', 'test']}, path)
            
            print("Load Split Dataset length: (Train: %d, Val: %d, Test: %d)" % (self.dataset_size['train'], self.dataset_size['val'], self.dataset_size['test']))

        else:
            train_val_size = int((1.0 - test_r) * len(self.all_dataset))
            self.dataset_size['test'] = len(self.all_dataset) - train_val_size
            train_val_dataset, self.dataset['test'] = random_split(self.all_dataset, [train_val_size, self.dataset_size['test']])

            
            self.dataset_size['train'] = int(train_r * train_val_size)
            self.dataset_size['val'] = train_val_size - self.dataset_size['train']
            self.dataset['train'], self.dataset['val'] = random_split(train_val_dataset, [self.dataset_size['train'], self.dataset_size['val']])

            path = os.path.join(self.output_folder, self.optname, 'dataset_indice')

            torch.save({phase: self.dataset[phase].indices for phase in ['train', 'val', 'test']}, path)
            # torch.save({'train': self.train_dataset.indices, 'val': self.val_dataset.indices, 'test': self.test_dataset.indices}, path)
            print("Random Split Dataset length: (Train: %d, Val: %d, Test: %d)" % (self.dataset_size['train'], self.dataset_size['val'], self.dataset_size['test']))

    def train_model(self, save_check, save_best=True, v_tqdm=True, update_lr_batch=False):

        load_kwargs, forward_kwargs, loss_kwargs = self._init_training()
        #* *** Training section ***
        print(' === ========Training begin========= ===')

        while self.epoch < self.paras['num_epochs']:

            print('Epoch %d/%d   lr = %.4e' % (self.epoch, self.paras['num_epochs'] - 1, self._optimizer.param_groups[0]['lr']))
            load_kwargs, forward_kwargs, loss_kwargs = self._start_of_epoch(load_kwargs, forward_kwargs, loss_kwargs)
            # torch.autograd.set_detect_anomaly(True)
            # 每个epoch都有一个训练和验证阶段
            for phase in self.phases:
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

                        data, batch_size = self._load_batch_data(batch_data, load_kwargs)
                        # 零参数梯度
                        self._optimizer.zero_grad()

                        # 前向, track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):

                            input_args, input_kwargs = self._forward_model(data, forward_kwargs)
                            output = self._model(*input_args, **input_kwargs)
                            
                            # print(output.size(), data['label'].size())
                            loss_dict = self._calculate_loss(data, output, loss_kwargs)
                            loss = loss_dict['loss']

                            if not torch.isnan(loss).sum() == 0:
                                print("NAN")
                                raise Exception()
                            
                            if v_tqdm:
                                tbar.set_postfix(loss=loss.item())
                            
                            # print(self.model.decoder_input._parameters['weight'].requires_grad, self.model.fc_mu._parameters['weight'].requires_grad)
                            #* model backward process (only in training phase)
                            if phase == 'train':
                                loss.backward()
                                if self._optimizer_setting['ema_optimizer']:
                                    self.ema_clip.on_before_optimizer_step(self._model)
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
                        self.save_model()

            self._end_of_epoch()
            if (self.epoch + 1) % save_check == 0:
                self.save_checkpoint(epoch=self.epoch)
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

    def _calculate_loss(self, data, output, kwargs):
        return {'loss': torch.nn.functional.mse_loss(output, data['label'])}
    
    def _end_of_epoch(self):
        pass

    def save_model(self, name='best_model'):
        path = self.output_path + '/' + name
        torch.save(self._model.state_dict(), path)
    
    def save_checkpoint(self, epoch, name='checkpoint', save_dict={}):
        path = self.output_path + '/' + name + '_epoch_' + str(epoch)
        save_dict['epoch'] = epoch
        save_dict['model_state_dict'] = self._model.state_dict()
        save_dict['optimizer_state_dict'] = self._optimizer.state_dict()
        save_dict['scheduler_state_dict'] = self._scheduler.state_dict()
        save_dict['history'] = self.history
        
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

        if load_data_split and len(self.dataset) > 0:
            self.split_dataset(recover=self.optname)

        print('checkpoint at epoch %d loaded from %s' % (save_dict['epoch'], path))

        return save_dict

    def set_transfer_model(self, transfer_name, grad_require_layers=['fc_code', 'decoder_input'], reset_param=True):
        '''
        load base model and set part of the parameters with grad off

        '''
        self.global_start_epoch = self.epoch
        self.epoch = 0
        self.set_optname(self.optname + '_' + transfer_name)

        if grad_require_layers is not None:

            for param in self.model.parameters():
                param.requires_grad = False
            
            print('------- The layers below are set grad require ------')
            for ly in grad_require_layers:
                print(ly, ' :   ', self.model._modules[ly])
                if reset_param:
                    self.model._modules[ly].apply(reset_paras)

                for param in self.model._modules[ly].parameters():
                    param.requires_grad = True
            
        else:

            print('------- All layers are set grad require ------')

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
                 split_train_ratio: float = 0.9, recover_split: str = None, 
                 batch_size: int = 8, shuffle: bool = True, ema_optimizer: bool = False,
                 ref: bool = False, ref_channels: Tuple[int] = (None, 2), recon_channels = (None, None), input_channels = (None, None)):
        
        self.ref = ref
        self.ref_channels = ref_channels
        self.recon_channels = recon_channels
        self.input_channels = input_channels
        super().__init__(opt_name, model, dataset, output_folder, init_lr, num_epochs, split_train_ratio, recover_split, batch_size, shuffle, ema_optimizer)

    def _forward_model(self, data, kwargs):
        return [data['input'][:, self.input_channels[0]: self.input_channels[1]]], {}
    
    def _calculate_loss(self, data, output, kwargs):
        # print(output[0].size(), data['label'].size())
        labels = data['label']
        if self.ref: labels[:, self.ref_channels[0]: self.ref_channels[1]] -= data['input'][:, self.recon_channels[0]: self.recon_channels[1]]

        return {'loss': torch.nn.functional.mse_loss(output[0], labels)}

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
                       recover_split: str = None,
                       batch_size: int = 8, 
                       shuffle=True,
                       ema_optimizer: bool = False,
                       ref=False,
                       input_channels: Tuple[int, int] = (None, None), 
                       recon_channels: Tuple[int, int] =(1, None),
                       recon_type: str = 'field'
                       ):
        
        super().__init__(opt_name, model, dataset, output_folder, init_lr, num_epochs, split_train_ratio, recover_split, batch_size, shuffle, ema_optimizer)
        
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
        for key in kwargs:
            self.paras['loss_parameters'][key] = kwargs[key]

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

    def save_checkpoint(self, epoch, name='checkpoint'):
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
