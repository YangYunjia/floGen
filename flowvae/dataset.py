
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

    `file_name`:   name of the data file\n
    `channels`:    (Tuple(int)) the channels of input geometry and flowfields\n
    `d_c`:         dimension of the condition\n
    `n_c`:         number of the condition of one airfoil should the dataset give out\n
    `c_mtd`:       how to choose the condition should give out\n
    >   `fix` : by the index in `c_map`\n
    >   `random`: ramdomly selected when initialize the dataset\n
    `c_map`: (list)\n
    `shuffle`: (bool) 
    >   if true, give an data with airfoil_index=index/n_c, condition_index=index%n_c\n
    >   if false, give a series of data of an airfoil\n
    `test`: (int) the number of data not involved\n
    `data_base`: (str)    the fold path of data\n
    `ref`: (bool) 
    >   if True, give delta aoa for every output, and give reference flowfield\n
    >   if false, the aoa is real, and no reference flowfield in getitem

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

    def __init__(self, file_name, d_c=1, n_c=5, c_mtd='fix', c_map=None, c_no=0, test=98, data_base='data/', ref=False):

        super().__init__()

        self.fname = file_name
        self.data_base = data_base

        self.all_data = np.load(data_base + file_name + 'data.npy')
        self.all_index = np.load(data_base + file_name + 'index.npy')

        self.condis_dim = d_c
        self.airfoil_num = int(self.all_index[-1][0]) + 1   #   amount of airfoils in dataset
        # print(self.all_index[-1][0])
        self.condis_all_num = np.zeros((self.airfoil_num,), dtype=np.int)       #   amount of conditions for each airfoil, a array of (N_airfoil, )
        self.condis_st      = np.zeros((self.airfoil_num,), dtype=np.int)       #   the start index of each airfoil in the serial dataset
        self.ref_index      = np.zeros((self.airfoil_num,), dtype=np.int)       #   the index of reference flowfield for each airfoil in the serial dataset
        self.ref_condis     = np.zeros((self.airfoil_num, self.condis_dim), dtype=np.float)     #   the aoa of the reference flowfield 
        self.condis_num = n_c                               #   amount of conditions used in training for each airfoil
        self.shuffle = False
        self.isref   = ref
        self.data = None            # flowfield data selected from all data, size: (N_airfoil * N_c, C, H, W)
        self.cond = None            # condition data (aoa) selected, size: (N_airfoil * N_c, )
        self.refr = None            # reference data, size: (N_airfoil, C, H, W)
        self.dataset_size = 0
        
        self._check_index()
        self._select_index(c_mtd=c_mtd, c_map=c_map, test=test, no=c_no)

        print("dataset %s of size %d loaded, shape:" % (file_name, len(self)), self.data.shape)

    
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

    def _select_index(self, c_mtd, c_map, test, no):
        '''
        select among the conditions of each airfoil for training

        para:
        ===
        `c_mtd`     (str) method to choose conditions for each airfoil
            - `fix`:    give the index of conditions same for every airfoil in `c_map`
            - `random`: ramdomly select the conditions of the amount of `self.condis_num` (reference include)
            - `load`:   load the index list from dataindex.txt
                if not exist, use random
            - `all`:    use all the conditions to training
        '''

        self.data_idx = []

        print('# selecting data from data.npy #')

        if c_mtd == 'fix':
            # check if c_map has correct size
            if c_map is None or len(c_map) != self.condis_num:
                raise Exception()
        elif c_mtd == 'load':
            fname = self.data_base + self.fname + '_%ddataindex.txt' % no
            if not os.path.exists(fname):
                print(' *** WARNING *** Data index file \'%s\' not exist, use random instead!' % fname)
                c_mtd = 'random'
        elif c_mtd in ['random', 'all']:
            pass

        else:
            raise KeyError()

        if c_mtd in ['fix', 'random']:
            for i in range(self.airfoil_num - test):
                if c_mtd == 'random':
                    # print(self.condis_st[i], self.condis_num)
                    c_map = random.sample(range(self.condis_all_num[i]), self.condis_num)
                elif c_mtd == 'all':
                    c_map = range(self.condis_num)
                
                for j in range(self.condis_num):
                    self.data_idx.append(c_map[j] + self.condis_st[i])
        else:
            self.data_idx = np.loadtxt(fname, dtype=np.int)

        self.data = torch.from_numpy(np.take(self.all_data, self.data_idx, axis=0)).float()
        self.cond = torch.from_numpy(np.take(self.all_index[:, 3:3+self.condis_dim], self.data_idx, axis=0)).float() 
        self.refr = torch.from_numpy(np.take(self.all_data, self.ref_index, axis=0)).float()
        self.dataset_size = len(self.data)

        print(' *** number of conditions:  %d, size of data:' % self.condis_num, self.data.size())

    def save_data_idx(self, no):
        np.savetxt(self.data_base + self.fname + '_%ddataindex.txt' % no, self.data_idx, fmt='%d')

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        
        # if self.shuffle:
        op_cod = idx % self.condis_num
        op_idx = int(idx / self.condis_num)
        # print(idx, cod)
        flowfield   = self.data[idx]
        condis      = self.cond[idx]
        refence     = self.refr[op_idx]
        ref_cond    = self.ref_condis[op_idx]

        # else:
        #     flowfield   = self.data[idx*self.condis_num: (idx+1)*self.condis_num]
        #     condis      = self.cond[idx*self.condis_num: (idx+1)*self.condis_num]
        #     refence     = self.refr[idx]
        #     op_idx = idx
        #     op_cod = None
        # refence[1] = refence[1]  * int(self.isref)
        sample = {'flowfields': flowfield, 'condis': condis, 
                  'index': op_idx, 'code_index': op_cod,
                  'ref': refence, 'ref_aoa': ref_cond}  # all the reference of the flowfield is transfered, the airfoil geometry (y) is also.
        # print(sample)
        # input()
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

    
    
    
    def get_index_info(self, i_f, i_c, i_idx):
        return self.all_index[int(self.condis_st[i_f] + i_c), i_idx]

    
    def get_buffet(self, idx):

        return self.get_index_info(idx, self.all_index[self.condis_st[idx], 8], 3), self.get_index_info(idx, self.all_index[self.condis_st[idx], 8], 6)