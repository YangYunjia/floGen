
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
    >    2:          Index of the reference condtion of this foil\n
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
                 extra_ref_channel=[]):

        super().__init__()

        self.fname = file_name
        self.data_base = data_base

        if channel_take is None:
            self.all_data = np.load(data_base + file_name + 'data.npy')
        else:
            self.all_data = np.take(np.load(data_base + file_name + 'data.npy'), channel_take, axis=1)
        self.all_index = np.load(data_base + file_name + 'index.npy')

        self.condis_dim = d_c
        self.airfoil_num = int(self.all_index[-1][0]) + 1   #   amount of airfoils in dataset
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
        self.extra_ref_channel = extra_ref_channel
        self.n_extra_ref_channel = len(extra_ref_channel)
        
        self._check_index()
        self._select_index(c_mtd=c_mtd, n_c=n_c, c_map=c_map, test=test, no=c_no, is_last=is_last_test)

        print("dataset %s of size %d loaded, shape:" % (file_name, len(self)), self.all_data.shape)

    
    def _check_index(self):
        '''
        find the start index of each airfoil in the sequencial all-database, also
        find the reference index and the reference conditions of each airfoil

        '''

        print('# checking the index in the index.npy #')
        print(' *** number of airfoils:  %d' % self.airfoil_num)

        airfoil_idx = -1
        for i, idx in enumerate(self.all_index):
            if idx[0] != airfoil_idx:
                airfoil_idx = int(idx[0])
                self.condis_st[airfoil_idx] = i
                self.ref_index[airfoil_idx] = int(idx[2]) + i  # i_ref
                self.ref_condis[airfoil_idx] = idx[3 + self.condis_dim: 3 + 2 * self.condis_dim]           # aoa_ref
            
            self.condis_all_num[airfoil_idx] += 1
        
        self.ref_condis = torch.from_numpy(self.ref_condis).float()
        self.refr = torch.from_numpy(np.take(self.all_data, self.ref_index, axis=0)).float()

        if self.n_extra_ref_channel > 0:
            self.extra_refr = np.take(self.all_index, self.extra_ref_channel, axis=1)

    def _select_index(self, c_mtd, n_c, c_map, test, no, is_last):
        '''
        select among the conditions of each airfoil for training
        '''
        self.data_idx = []
        self.airfoil_idx = []

        print('# selecting data from data.npy #')
        minnc = 1000
        maxnc = -1

        if c_mtd == 'load':
            fname = self.data_base + self.fname + '_%ddataindex.txt' % no
            if not os.path.exists(fname):
                raise IOError(' *** ERROR *** Data index file \'%s\' not exist, use random instead!' % fname)
            else:
                self.data_idx = np.loadtxt(fname, dtype=np.int32)

            for iidx in self.data_idx:
                if self.all_index[iidx][0] not in self.airfoil_idx:
                    self.airfoil_idx.append(self.all_index[iidx][0])

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
            
            self.save_data_idx(no)

        # self.data = torch.from_numpy(np.take(self.all_data, self.data_idx, axis=0)).float()
        # self.cond = torch.from_numpy(np.take(self.all_index[:, 3:3+self.condis_dim], self.data_idx, axis=0)).float() 
        self.dataset_size = len(self.data_idx)

        print(' *** number of conditions:  %d ~ %d, total size of data: %d ' % (minnc, maxnc, self.dataset_size), self.all_data.shape)

    def save_data_idx(self, no):
        np.savetxt(self.data_base + self.fname + '_%ddataindex.txt' % no, self.data_idx, fmt='%d')

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        
        # op_cod = idx % self.condis_num
        # op_idx = int(idx / self.condis_num)
        op_idx =  int(self.all_index[self.data_idx[idx], 0])
        op_cod =  int(self.all_index[self.data_idx[idx], 1])
        # print(idx, cod)
        flowfield   = torch.from_numpy(self.all_data[self.data_idx[idx]]).float()
        condis      = torch.from_numpy(self.all_index[self.data_idx[idx], 3:3+self.condis_dim]).float()
        # condis      = self.cond[idx]
        refence     = self.refr[op_idx]
        ref_cond    = self.ref_condis[op_idx]

        #! This part is for add a extra reference channel input for geometry -> flowfield prediction problem
        #! Because the input may not include all the freestream conditions need to generate the flowfield, while
        #! for ref.flowfield -> target flowfield prediction problem, the extra freestream conditions are implicit
        #! contained in the refernce input.
        #! The extra input is added as a uniform new channel for the same size of the geometry
        # refence   = torch.concatenate([self.extra_refr[op_idx] * torch.ones((self.n_extra_ref_channel, refence.size(1))), refence], dim=0)

        sample = {'flowfields': flowfield, 'condis': condis, 
                  'index': op_idx, 'code_index': op_cod,
                  'ref': refence, 'ref_aoa': ref_cond}  # all the reference of the flowfield is transfered, the airfoil geometry (y) is also.

        return sample
          
    
    def get_series(self, idx, ref_idx=None):

        st = self.condis_st[idx]
        ed = self.condis_st[idx] + self.condis_all_num[idx]
        flowfield   = self.all_data[st: ed]
        condis      = self.all_index[st: ed, 3:3+self.condis_dim]

        if ref_idx is None:
            ref         = self.all_data[self.ref_index[idx]]
            ref_aoa     = self.ref_condis[idx]
        else:   
            ref         = self.all_data[st + ref_idx]
            ref_aoa     = self.all_index[st + ref_idx, 3:3+self.condis_dim] 

        return {'flowfields': flowfield, 'condis': condis, 'ref': ref, 'ref_aoa': ref_aoa}

    
    def allnum_condition(self, idx):
        return self.condis_all_num[idx]
    
    def get_index_info(self, i_f, i_c, i_idx):
        return self.all_index[int(self.condis_st[i_f] + i_c), i_idx]

    
    def get_buffet(self, idx):

        return self.get_index_info(idx, self.all_index[self.condis_st[idx], 8], 3), self.get_index_info(idx, self.all_index[self.condis_st[idx], 8], 6)