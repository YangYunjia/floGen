
'''
Yang Yunjia @ 20210221

Rewrite the dataset part of Deng's code (prepare_data.py), 
adapted from the original module of pyTorch to simplify

'''

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from typing import List, Callable, NewType, Union, Tuple

def load_with_float_check(file: str, target_type = np.float32) -> np.ndarray:

    data: np.ndarray = np.load(file)
    info = f'> load from file: {file}'
    if data.dtype != target_type:
        data = data.astype(target_type)
        info += f' - change format to {target_type}'

    print(info)
    return data


class FlowDataset(Dataset):
    '''
    This class formulate a normal input-output-pair dataset

    ### params

    - `file_name` (`str` or `List[str]`):  the data name
        - if input is a str, the output will be `%file_name%data.npy`; the input will be `%file_name%index.npy`
        - if input is a list, the output will be `%file_name[0]%.npy`; the input will be `%file_name[1]%.npy`
    
    
    
    '''
    default_split_paras = {
        'method': 'load',   # overall selection methods (load, random, all)
        'nTest':  0,        # overall test amounts
        'isLastTest': True,
        'indexFileNum': -1, # Number of the index file
        'indexFileName': None,
    }

    def __init__(self, file_name: Union[str, List[str]], 
                 split_paras: dict = {},
                 data_base: str = 'data', 
                 input_channel_take: List[int] = None,
                 output_channel_take: List[int] = None,
                 aux_channel_take: List[int] = None,
                 flatten: bool = False, 
                 swap_axis: tuple = None, 
                 unsqueeze: int = None,
                 marker_idx: int = 1) -> None:
        
        super().__init__()

        
        # input and output data (swapaxis, flatten, take channel)
        self.data_base = data_base
        self.flatten = flatten
        
        if isinstance(file_name, str):
            self.all_output = load_with_float_check(os.path.join(data_base, file_name + 'data.npy'))
            self.all_input  = load_with_float_check(os.path.join(data_base, file_name + 'index.npy'))
            self.fname = file_name
        elif isinstance(file_name, List):
            self.all_output = load_with_float_check(os.path.join(data_base, file_name[0] + '.npy'))
            self.all_input = load_with_float_check(os.path.join(data_base, file_name[1] + '.npy'))
            self.fname = file_name[0] + file_name[1]

        if swap_axis is not None:
            self.all_input = np.swapaxes(self.all_input, *swap_axis)

        if flatten:
            # print(self.all_input.shape)
            self.all_output = np.swapaxes(self.all_output, 1, 2)
            self.all_output = self.all_output.reshape(-1, *self.all_output.shape[2:])
            self.all_input  = np.swapaxes(self.all_input, 1, 2)
            self.all_input  = self.all_input.reshape(-1, *self.all_input.shape[2:])
            print('> shape after flatten: ', self.all_output.shape)

        if output_channel_take is None:
            self.output = self.all_output
        else:
            self.output = np.take(self.all_output, output_channel_take, axis=1)
        
        if input_channel_take is None:
            self.inputs = self.all_input
        else:
            self.inputs = np.take(self.all_input, input_channel_take, axis=1)
        
        if unsqueeze is not None:
            self.inputs = self.inputs.unsqueeze(unsqueeze)
        
        # aux data
        if isinstance(file_name, List) and len(file_name) > 2:
            self.all_aux = load_with_float_check(os.path.join(data_base, file_name[2] + '.npy'))
            if aux_channel_take is None:
                self.auxs = self.all_aux
            else:
                self.auxs = np.take(self.all_aux, aux_channel_take, axis=1)
        else:
            self.all_aux = None
            self.auxs    = None

        # select training and testing dataset
        # decide default parameters
        self._split_paras = {}
        for k in self.__class__.default_split_paras.keys():
            self._split_paras[k] = split_paras[k] if k in split_paras.keys() else self.__class__.default_split_paras[k]
        if self._split_paras['indexFileName'] is None:
            self._split_paras['indexFileName'] = os.path.join(self.data_base, self.fname + '_%ddataindex.txt' % self._split_paras['indexFileNum'])
        
        # decide which array contains the index information
        self.all_index = [self.all_output, self.all_input, self.all_aux][marker_idx]
        
        self.data_idx = []
        self._select_index()

    def _load_index_file(self):
        
        fname = self._split_paras['indexFileName']
        
        assert os.path.exists(fname), ' *** ERROR *** Data index file \'%s\' not exist, use random instead!' % fname
        self.data_idx = np.loadtxt(fname, dtype=np.int32)
            
        print(f'> load flow field index from {fname}')
        print(f'> total size of data:')
        print(f'>   inputs -> {self.inputs.shape}')
        print(f'>   output -> {self.output.shape}')
        if self.auxs is not None:
            print(f'>   auxs   -> {self.auxs.shape}')
            
    def _save_data_idx(self):
        assert not os.path.exists(self._split_paras['indexFileName']), ' *** ERROR *** Data index file \'%s\' exist' % self._split_paras['indexFileName']
        np.savetxt(self._split_paras['indexFileName'], self.data_idx, fmt='%d')

    def _select_index(self):
        '''
        select among the conditions of each airfoil for training
        '''
        self.data_idx = []

        print('# selecting data from data.npy #')

        if self._split_paras['method'] == 'load':
            self._load_index_file()

        else:
            
            if self._split_paras['method'] == 'all':
                self.data_idx = list(range(self.all_output.shape[0]))
            
            elif self._split_paras['method'] == 'random':
                    
                if self._split_paras['isLastTest']:
                    self.data_idx = list(range(self.all_output.shape[0] - self._split_paras['nTest']))
                else:
                    self.data_idx = random.sample(range(self.all_output.shape[0]), self.all_output.shape[0] - self._split_paras['nTest'])

            else:
                raise KeyError(f"split method {self._split_paras['method']} not available")
            
            self._save_data_idx()

        self.dataset_size = len(self.data_idx)

    def normalize(self) -> None:
        self.inputs = self.inputs.detach().numpy()
        output_min = np.min(self.inputs, axis=0)
        output_max = np.max(self.inputs, axis=0)
        print('dataset input is normalized with', output_min, output_max)
        self.inputs = torch.from_numpy((self.inputs - output_min) / (output_max - output_min)).float()

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, index) -> dict:
        d_index = self.data_idx[index]
        inputs  = torch.from_numpy(self.inputs[d_index])
        labels  = torch.from_numpy(self.output[d_index])
        if self.all_aux is None:
            return {'input': inputs, 'label': labels}
        else:
            auxs = torch.from_numpy(self.auxs[d_index])
            return {'input': inputs, 'label': labels, 'aux': auxs}
    
class MCFlowDataset(FlowDataset):
    
    default_split_paras = {
        'method': 'load',   # overall + condition selection methods (load, random, all)
        'nTest':  0,        # overall test amounts
        'nCond': None,
        'mapCond': None,
        'isLastTest': True,
        'indexFileNum': -1, # Number of the index file
        'indexFileName': None,
    }
    
    def __init__(self, file_name, 
                 d_c: int = None,
                 is_ref: bool = True,
                 split_paras: dict = {},
                 data_base = 'data', 
                 input_channel_take = None, 
                 output_channel_take = None, 
                 aux_channel_take = None, 
                 flatten = False, 
                 swap_axis = None, 
                 unsqueeze = None, 
                 marker_idx: int = 1,
                 getindex_type: str = 'noref',
                 channel_cond: list = None):
        
        self.is_ref       = is_ref
        
        # decide where is the working condition
        if channel_cond is not None:
            self.condis_dim = len(channel_cond)
            self.channel_cond = channel_cond
        else:
            if d_c is not None:
                self.condis_dim = d_c
                channel_cond_st = [2, 3][is_ref]
                self.channel_cond = range(channel_cond_st, channel_cond_st + self.condis_dim)
            else:
                self.condis_dim = 0
                self.channel_cond = []
                print('[Warning] D_C is not set, do not use ref, get_series, etc.')
        
        self.airfoil_num = 0
        
        super().__init__(file_name, split_paras, data_base, input_channel_take, output_channel_take, aux_channel_take, flatten, swap_axis, unsqueeze, marker_idx)
                
        if getindex_type in ['noref']:
            self.get_item = super().__getitem__
        elif getindex_type in ['normal']:
            self.get_item = self._get_normal_item   # with reference
        elif getindex_type in ['origeom']:
            self.get_item = self._get_origeom_item
            
        self._output_force_flag = False
        self.n_extra_ref_channel = 0

    def _check_index(self):
        '''
        find the start index of each airfoil in the sequencial all-database, also
        find the reference index and the reference conditions of each airfoil

        '''

        self.airfoil_num = len(np.unique(self.all_index[:, 0]))   #   amount of airfoils in dataset
        self.condis_all_num = np.zeros((self.airfoil_num,), dtype=np.int32)       #   amount of conditions for each airfoil, a array of (N_airfoil, )
        self.condis_st      = np.zeros((self.airfoil_num,), dtype=np.int32)       #   the start index of each airfoil in the serial dataset
        self.cond_index = np.take(self.all_index, self.channel_cond, axis=1)
        
        if self.is_ref:
            self.ref_index      = np.zeros((self.airfoil_num,), dtype=np.int32)       #   the index of reference flowfield for each airfoil in the serial dataset
            self.ref_condis     = np.zeros((self.airfoil_num, self.condis_dim), dtype=np.float64)     #   the aoa of the reference flowfield 
            self.refr = None            # reference data, size: (N_airfoil, C, H, W)

        print(f'---------------------------------------')
        print(f'> checking the index in the index.npy')

        airfoil_idx = -1
        last_idx    = -1
        # if input igroup is not contiune, it will be replaced with a continue one in running
        for i, idx in enumerate(self.all_index):
            if idx[0] != last_idx:
                airfoil_idx += 1
                last_idx     = int(idx[0])

                self.condis_st[airfoil_idx] = i
                
                if self.is_ref:
                    self.ref_index[airfoil_idx] = int(idx[2]) + i  # i_ref
                    self.ref_condis[airfoil_idx] = self.cond_index[int(idx[2]) + i]           # aoa_ref
            
            if idx[0] != airfoil_idx:
                self.all_index[i, 0] = airfoil_idx
                
            self.condis_all_num[airfoil_idx] += 1
        
        if self.is_ref:
            self.ref_condis = torch.from_numpy(self.ref_condis).float()
            self.refr = torch.from_numpy(np.take(self.all_output, self.ref_index, axis=0)).float()

        print(f'> number of geometries:   {self.airfoil_num:d}')
        print(f'> number of flow fields:  {len(self.all_index):d}')
        print(f'> min & max number of flow fields for one geometry:  {min(self.condis_all_num):d}  {max(self.condis_all_num):d}')
        print(f'---------------------------------------')

    def _select_index(self):
        
        '''
        select among the conditions of each airfoil for training
        '''
        self.data_idx = []
        self.airfoil_idx = []
        self._check_index()

        print(f'selecting data from data.npy')
        print(f'> selecting method will be {self._split_paras["method"]}')

        if self._split_paras['method'] == 'load':
            
            self._load_index_file()

            for iidx in self.data_idx:
                if self.all_index[iidx][0] not in self.airfoil_idx:
                    # load the geometry index
                    self.airfoil_idx.append(self.all_index[iidx][0])
            
        else:

            if self._split_paras['method'] in ['fix', 'random', 'all', 'exrf']:
                
                # select airfoil testing part
                if self._split_paras['isLastTest']:
                    self.airfoil_idx = list(range(self.airfoil_num - self._split_paras['nTest']))
                else:
                    self.airfoil_idx = random.sample(range(self.airfoil_num), self.airfoil_num - self._split_paras['nTest'])

                for i in self.airfoil_idx:
                    if self._split_paras['method'] == 'random':
                        # print(self.condis_st[i], self.condis_num)
                        c_map = random.sample(range(self.condis_all_num[i]), self._split_paras['nCond'])
                    elif self._split_paras['method'] == 'all':
                        c_map = list(range(self.condis_all_num[i]))
                    elif self._split_paras['method'] == 'exrf':
                        c_map = list(range(self.condis_all_num[i]))
                        c_map.remove((self.ref_index[i] - self.condis_st[i]))
                    elif self._split_paras['method'] == 'fix':
                        c_map = list(self._split_paras['mapCond'])

                    for a_c_map in c_map:
                        self.data_idx.append(a_c_map + self.condis_st[i])
                        
            else:
                raise KeyError(f"split method {self._split_paras['method']} not available")
            
            self._save_data_idx()

        # self.data = torch.from_numpy(np.take(self.all_data, self.data_idx, axis=0)).float()
        # self.cond = torch.from_numpy(np.take(self.all_index[:, 3:3+self.condis_dim], self.data_idx, axis=0)).float() 
        self.dataset_size = len(self.data_idx)

        print(f'---------------------------------------')

    def __getitem__(self, index):
        return self.get_item(index)
    
    def _get_origeom_item(self, index):
        '''
        get input, output, aux pairs. The input is only geometry. So there is only one
        for each shape.
        '''

        d_index = self.data_idx[index]
        op_idx =  int(self.all_index[d_index, 0])
        inputs  = torch.from_numpy(self.inputs[op_idx])
        labels  = torch.from_numpy(self.output[d_index]).float()
        if self.all_aux is None:
            return {'input': inputs, 'label': labels}
        else:
            auxs = torch.from_numpy(self.auxs[d_index])
            return {'input': inputs, 'label': labels, 'aux': auxs}

    
    def _get_normal_item(self, idx):

        # op_cod = idx % self.condis_num
        # op_idx = int(idx / self.condis_num)
        op_idx =  int(self.all_index[self.data_idx[idx], 0])
        op_cod =  int(self.all_index[self.data_idx[idx], 1])
        # print(idx, cod)
        flowfield   = torch.from_numpy(self.all_output[self.data_idx[idx]]).float()
        condis      = torch.from_numpy(self.cond_index[self.data_idx[idx]]).float()
        # condis      = self.cond[idx]
        refence     = self.refr[op_idx]
        ref_cond    = self.ref_condis[op_idx]

        sample = {'flowfields': flowfield, 'condis': condis, 
                  'index': op_idx, 'code_index': op_cod,
                  'ref': refence, 'ref_aoa': ref_cond}  # all the reference of the flowfield is transfered, the airfoil geometry (y) is also.

        return sample

    def get_series(self, idx, ref_idx=None):

        st = self.condis_st[idx]
        ed = self.condis_st[idx] + self.condis_all_num[idx]
        
        if self._output_force_flag: flowfield = self.all_force[st: ed]
        else: flowfield   = self.all_output[st: ed]
        condis      = self.cond_index[st: ed]

        if self.is_ref:
            if ref_idx is None:
                ref         = self.all_output[self.ref_index[idx]]
                ref_aoa     = self.ref_condis[idx]
            else:   
                ref         = self.all_output[st + ref_idx]
                ref_aoa     = self.cond_index[st + ref_idx]
            
        else:
            ref      = self.all_input[st: ed]
            ref_aoa  = None

        samples = {'flowfields': flowfield, 'condis': condis, 'ref': ref, 'ref_aoa': ref_aoa}
        if self._output_force_flag: 
            samples['ref_force'] = self.ref_force[idx]
        
        return samples 

    def allnum_condition(self, idx):
        return self.condis_all_num[idx]
    
    def get_index_info(self, i_f, i_c, i_idx):
        return self.all_index[int(self.condis_st[i_f] + i_c), i_idx]

class ConditionDataset(Dataset):
    '''

    dataset for flow data for <i>different</i> airfoils under different flow condition

    initial params
    ===

    - `file_name`:   name of the data file
    - `d_c`:         dimension of the condition
    - `c_mtd`:       how to choose the conditions used in training
        - `fix` :   by the index in `c_map`
        - `random`: ramdomly selecte `n_c` conditions when initialize the dataset
        - `all`:    all the conditions will be used to training
        - `exrf`:   except reference conditions
        - `load`:   load the selection method by the file number `c_no` (use `save_data_idx(c_no)` to save)
    - `n_c`:        (int)  number of the condition of one airfoil should the dataset give out
    - `c_map`:      (list) fix index number
    - `c_no`:       (int)  number of saved data index
    - `test`:       (int) the number of data not involved
    - `is_last_test`(bool) if true, the last `test` samples will be used as test samples; if false, randomly selected
    - `data_base`:  (str) the folder path of data

    dataset file requirment
    ===

    The datafile should contain two parts: the `data.npy` and the `index.npy`
    
    data.npy
    ----
    shape: `(N_Foils * N_Conditions) x N_Channel x SHAPE_OF_A_SAMPLE`
    
    index.npy
    ----
    shape: `(N_Foils * N_Conditions) x N_Info`\n
    the information obey the following format:
    >    0:          Index of the foil\n
    >    1:          Index of the condition\n
    >    2:          Index of the reference condtion of this foil (local index)\n
    >    3~3+DC:     The condition values of the current point (Amount: Dimension of a condition)\n
    >    4+DC~4+2DC: The condition values of the reference point (Amount: Dimension of a condition)\n
    >    more:       Aux data\n
    '''

    def __init__(self, 
                 file_name, 
                 d_c=1, 
                 c_mtd='fix', 
                 n_c=None, 
                 c_map=None, 
                 c_no=-1, 
                 test=-1, 
                 data_base='data/', 
                 is_last_test=True, 
                 channel_take=None,
                 channel_cond: list = None):

        super().__init__()

        self.fname = file_name
        self.data_base = data_base

        if channel_take is None:
            self.all_data = np.load(data_base + file_name + 'data.npy')
        else:
            self.all_data = np.take(np.load(data_base + file_name + 'data.npy'), channel_take, axis=1)
        self.all_index = np.load(data_base + file_name + 'index.npy')
        
        self.condis_dim = d_c
        if channel_cond is None:
            self.cond_index = np.take(self.all_index, range(3, 3+self.condis_dim), axis=1)
        else:
            self.cond_index = np.take(self.all_index, channel_cond, axis=1)

        self.airfoil_num = len(np.unique(self.all_index[:, 0]))   #   amount of airfoils in dataset
        # print(self.all_index[-1][0])
        self.condis_all_num = np.zeros((self.airfoil_num,), dtype=np.int32)       #   amount of conditions for each airfoil, a array of (N_airfoil, )
        self.condis_st      = np.zeros((self.airfoil_num,), dtype=np.int32)       #   the start index of each airfoil in the serial dataset
        self.ref_index      = np.zeros((self.airfoil_num,), dtype=np.int32)       #   the index of reference flowfield for each airfoil in the serial dataset
        self.ref_condis     = np.zeros((self.airfoil_num, self.condis_dim), dtype=np.float64)     #   the aoa of the reference flowfield 
        # self.condis_num = n_c                               #   amount of conditions used in training for each airfoil
        # self.data = None            # flowfield data selected from all data, size: (N_airfoil * N_c, C, H, W)
        # self.cond = None            # condition data (aoa) selected, size: (N_airfoil * N_c, )
        self.refr = None            # reference data, size: (N_airfoil, C, H, W)
        self.dataset_size = 0
        
        self._check_index()
        self._select_index(c_mtd=c_mtd, n_c=n_c, c_map=c_map, test=test, no=c_no, is_last=is_last_test)

        print("dataset %s of size %d loaded, shape:" % (file_name, len(self)), self.all_data.shape)

        self.get_item = self._get_normal_item
        self._output_force_flag = False
        self.n_extra_ref_channel = 0
    
    def _check_index(self):
        '''
        find the start index of each airfoil in the sequencial all-database, also
        find the reference index and the reference conditions of each airfoil

        '''

        print(f'---------------------------------------')
        print(f'> checking the index in the index.npy')

        airfoil_idx = -1
        last_idx    = -1
        # if input igroup is not contiuns, it will be replaced with a continues one in running
        for i, idx in enumerate(self.all_index):
            if idx[0] != last_idx:
                airfoil_idx += 1
                last_idx     = int(idx[0])

                self.condis_st[airfoil_idx] = i
                self.ref_index[airfoil_idx] = int(idx[2]) + i  # i_ref
                self.ref_condis[airfoil_idx] = self.cond_index[int(idx[2]) + i]           # aoa_ref
            
            if idx[0] != airfoil_idx:
                self.all_index[i, 0] = airfoil_idx
                
            self.condis_all_num[airfoil_idx] += 1
        
        self.ref_condis = torch.from_numpy(self.ref_condis).float()
        self.refr = torch.from_numpy(np.take(self.all_data, self.ref_index, axis=0)).float()

        print(f'> number of geometries:   {self.airfoil_num:d}')
        print(f'> number of flow fields:  {len(self.all_index):d}')
        print(f'> min & max number of flow fields for one geometry:  {min(self.condis_all_num):d}  {max(self.condis_all_num):d}')
        print(f'---------------------------------------')

    def _select_index(self, c_mtd, n_c, c_map, test, no, is_last):
        '''
        select among the conditions of each airfoil for training
        '''
        self.data_idx = []
        self.airfoil_idx = []

        print(f'selecting data from data.npy')
        print(f'> selecting method will be {c_mtd}')
        minnc = 1000
        maxnc = -1

        if c_mtd == 'load':
            
            if fname is None:
                fname = os.path.join(self.data_base, self.fname + '_%ddataindex.txt' % no)
            
            assert os.path.exists(fname), ' *** ERROR *** Data index file \'%s\' not exist, use random instead!' % fname
            self.data_idx = np.loadtxt(fname, dtype=np.int32)

            for iidx in self.data_idx:
                if self.all_index[iidx][0] not in self.airfoil_idx:
                    # load the geometry index
                    self.airfoil_idx.append(self.all_index[iidx][0])
            
            print(f'> load flow field index from {fname}')
        
        else:

            if c_mtd in ['fix', 'random', 'all', 'exrf']:
                
                if is_last:
                    self.airfoil_idx = list(range(self.airfoil_num - test))
                else:
                    self.airfoil_idx = random.sample(range(self.airfoil_num), self.airfoil_num - test)

                for i in self.airfoil_idx:
                    if c_mtd == 'random':
                        # print(self.condis_st[i], self.condis_num)
                        c_map = random.sample(range(self.condis_all_num[i]), n_c)
                    elif c_mtd == 'all':
                        c_map = list(range(self.condis_all_num[i]))
                    elif c_mtd == 'exrf':
                        c_map = list(range(self.condis_all_num[i]))
                        c_map.remove((self.ref_index[i] - self.condis_st[i]))
                    elif c_mtd == 'fix':
                        pass
                    else:
                        raise KeyError()

                    for a_c_map in c_map:
                        self.data_idx.append(a_c_map + self.condis_st[i])

                    minnc = min(len(c_map), minnc)
                    maxnc = max(len(c_map), maxnc)
            else:
                raise KeyError()
            
            print(f'> number of conditions:  {minnc:d} ~ {maxnc:d}')
            self.save_data_idx(no)

        # self.data = torch.from_numpy(np.take(self.all_data, self.data_idx, axis=0)).float()
        # self.cond = torch.from_numpy(np.take(self.all_index[:, 3:3+self.condis_dim], self.data_idx, axis=0)).float() 
        self.dataset_size = len(self.data_idx)

        print(f'> total size of data: {self.all_data.shape}')
        print(f'---------------------------------------')

    def save_data_idx(self, no):
        fname = self.data_base + self.fname + '_%ddataindex.txt' % no
        assert not os.path.exists(fname), ' *** ERROR *** Data index file \'%s\' exist' % fname
        
        print(f'> save flow field index from {fname}')
        np.savetxt(fname, self.data_idx, fmt='%d')

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.get_item(idx)
    
    def _get_normal_item(self, idx):

        # op_cod = idx % self.condis_num
        # op_idx = int(idx / self.condis_num)
        op_idx =  int(self.all_index[self.data_idx[idx], 0])
        op_cod =  int(self.all_index[self.data_idx[idx], 1])
        # print(idx, cod)
        flowfield   = torch.from_numpy(self.all_data[self.data_idx[idx]]).float()
        condis      = torch.from_numpy(self.cond_index[self.data_idx[idx]]).float()
        # condis      = self.cond[idx]
        refence     = self.refr[op_idx]
        ref_cond    = self.ref_condis[op_idx]

        sample = {'flowfields': flowfield, 'condis': condis, 
                  'index': op_idx, 'code_index': op_cod,
                  'ref': refence, 'ref_aoa': ref_cond}  # all the reference of the flowfield is transfered, the airfoil geometry (y) is also.

        return sample
    
    def _get_extra_channel_item(self, idx):
        '''
        This part is for add a extra reference channel input for geometry -> flowfield prediction problem
        Because the input may not include all the freestream conditions need to generate the flowfield, while
        for ref.flowfield -> target flowfield prediction problem, the extra freestream conditions are implicit
        contained in the refernce input.
        The extra input is added as a uniform new channel for the same size of the geometry
        '''
        sample = self._get_normal_item(idx)
        sample['ref'] = torch.concatenate([self.extra_refr[sample['index']] * torch.ones((self.n_extra_ref_channel, sample['ref'].size(1))), sample['ref']], dim=0)
        return sample

    def _get_force_item(self, idx):
        
        op_idx =  int(self.all_index[self.data_idx[idx], 0])
        op_cod =  int(self.all_index[self.data_idx[idx], 1])
        force   = torch.from_numpy(self.all_force[self.data_idx[idx]]).float()
        condis  = torch.from_numpy(self.cond_index[self.data_idx[idx]]).float()
        refence     = self.refr[op_idx]
        ref_cond    = self.ref_condis[op_idx]
        ref_force   = torch.from_numpy(self.ref_force[op_idx]).float()
          
        sample = {'flowfields': force, 'condis': condis, 
                  'index': op_idx, 'code_index': op_cod,
                  'ref': refence, 'ref_aoa': ref_cond, 'ref_force': ref_force}
        # print(sample['condis'], sample['ref_force'], sample['ref_aoa'])
        return sample
    
    def change_to_force(self, info=''):
        '''

        
        '''
        from cfdpost.utils import clustcos, get_force_1d
        
        if info in ['cltarg', 'aoatarg']:
            if info == 'cltarg':    
                forces = np.take(self.all_index, [10, 3], axis=1)    # cd, aoa (for ma model)
                _repr = 'AoA / Cd'
            if info == 'aoatarg':   
                forces = np.take(self.all_index, [10, 9], axis=1)    # cd, cl (for aoa model)
                _repr = 'Cl / Cd'
                
            print('Dataset is changed to output force %s as flowfield...' % _repr)
            # nondim
            param = (max(forces[:, 1]) - min(forces[:, 1])) / (max(forces[:, 0]) - min(forces[:, 0]))
            print('>    non-dimensional parameters for %s will be %.6f' % (_repr, param))
            forces[:, 0] *= param
        
            print('Dataset is changed to output force as flowfield...')
            self.all_force = forces
            
        else:
            force_file_path = os.path.join(self.data_base, 'force.npy')
            if os.path.exists(force_file_path):
                self.all_force = np.load(force_file_path)

            else:
                forces = np.zeros((len(self.all_data), 2))

                nn = 201
                xx = [clustcos(i, nn) for i in range(nn)]
                all_x = np.concatenate((xx[::-1], xx[1:]), axis=0)

                for i, sample in enumerate(self.all_data):
                    geom = np.stack((all_x, sample[0]), axis=0).transpose(1, 0)
                    aoa  = self.all_index[i, 3]
                    forces[i] = get_force_1d(geom, aoa, sample[1], sample[2] / 1000.)

                if info == 'non-dim':
                    param = (max(forces[:, 1]) - min(forces[:, 1])) / (max(forces[:, 0]) - min(forces[:, 0]))
                    print('>    non-dimensional parameters for Cl/Cd will be %.6f' % param)
                    forces[:, 0] *= param

                self.all_force = forces
                np.save(force_file_path, self.all_force)

        self.ref_force = np.take(self.all_force, self.ref_index, axis=0)
        self.get_item = self._get_force_item
        self._output_force_flag = True
        
        return param

    '''
    def change_to_force(self, info=''):
        # change to the difference between current airfoil and buffet onset
        
        from flowvae.post import clustcos, get_force_1d

        print('Dataset is changed to output buffet difference...')

        forces = torch.zeros(len(self.all_data), 2)

        nn = 201
        xx = [clustcos(i, nn) for i in range(nn)]
        all_x = np.concatenate((xx[::-1], xx[1:]), axis=0)

        buffet_data = torch.load('D:\\Deeplearning\\202210LRZdata\\save\\1025_84.blossn6')

        for i, sample in enumerate(self.all_data):
            geom = np.concatenate((all_x.reshape((1, -1)), sample[0].reshape((1, -1))), axis=0)
            aoa  = self.all_index[i, 3]
            profile = sample[1]
            forces[i, 0] = aoa
            forces[i, 1] = get_force_1d(torch.from_numpy(geom).float(), 
                                     torch.from_numpy(profile).float(), aoa)[1] # cl
            # print(i, forces[i], buffet_data[0, int(self.all_index[i, 0]), 3, 1, 1])
            forces[i] = forces[i] - buffet_data[0, int(self.all_index[i, 0]), 3, 1, 1]

        if info == 'non-dim':
            param = (max(forces[:, 1]) - min(forces[:, 1])) / (max(forces[:, 0]) - min(forces[:, 0]))
            print('>    non-dimensional parameters aoa / cl will be %.2f' % param)
            forces[:, 0] *= param

        self.all_force = forces.detach().numpy()
        self.ref_force = np.take(self.all_force, self.ref_index, axis=0)
        self.get_item = self._get_force_item
        self._output_force_flag = True
    '''

    def add_extra_ref_channel(self, extra_ref_channel):
        self.extra_ref_channel = extra_ref_channel
        self.n_extra_ref_channel = len(extra_ref_channel)
        self.extra_refr = np.take(self.all_index, self.extra_ref_channel, axis=1)
        # print(self.extra_refr.repeat())
        ref_channel_shape = list(self.all_data.shape[2:]) + [1, 1]
        self.all_data = np.concatenate([np.moveaxis(np.tile(self.extra_refr, (ref_channel_shape)), 0, -1), self.all_data], axis=1)
        self.refr = torch.from_numpy(np.take(self.all_data, self.ref_index, axis=0)).float()
        # self.get_item = self._get_extra_channel_item

    def get_series(self, idx, ref_idx=None):

        st = self.condis_st[idx]
        ed = self.condis_st[idx] + self.condis_all_num[idx]
        if self._output_force_flag: flowfield = self.all_force[st: ed]
        else: flowfield   = self.all_data[st: ed]
        condis      = self.cond_index[st: ed]

        if ref_idx is None:
            ref         = self.all_data[self.ref_index[idx]]
            ref_aoa     = self.ref_condis[idx]
        else:   
            ref         = self.all_data[st + ref_idx]
            ref_aoa     = self.cond_index[st + ref_idx]

        samples = {'flowfields': flowfield, 'condis': condis, 'ref': ref, 'ref_aoa': ref_aoa}
        if self._output_force_flag: 
            samples['ref_force'] = self.ref_force[idx]
        
        return samples 

    def allnum_condition(self, idx):
        return self.condis_all_num[idx]
    
    def get_index_info(self, i_f, i_c, i_idx):
        return self.all_index[int(self.condis_st[i_f] + i_c), i_idx]

    def get_buffet(self, idx):

        return self.get_index_info(idx, self.all_index[self.condis_st[idx], 8], 3), self.get_index_info(idx, self.all_index[self.condis_st[idx], 8], 6)
    
class DatasetIterator():
    '''
    for FlowDataset, return every sample and whether it is inside training dataset
    for MCFlowDataset, return all flowfield of every airfoil and whether it is inside training dataset
    
    '''
    
    def __init__(self, dataset: Union[FlowDataset, MCFlowDataset]):
        self.dataset = dataset
        self.current = 0
        if isinstance(dataset, FlowDataset):
            self._getitem = self._getitem_flowdataset
            self.n = dataset.dataset_size
        elif isinstance(dataset, MCFlowDataset):
            self._getitem = self._getitem_mcflowdataset
            self.n = dataset.airfoil_num
    
    def __next__(self):
        if self.current < self.n:
            results = self._getitem(self.current)
            self.current += 1
            return results
        else:
            raise StopIteration
    
    def __iter__(self):
        return self
    
    def _getitem_flowdataset(self, idx):
        raise NotImplementedError
    
    def _getitem_mcflowdataset(self, idx):
        return {'sample': self.dataset.get_series(idx),
                'flag': idx in self.airfoil_idx}
        