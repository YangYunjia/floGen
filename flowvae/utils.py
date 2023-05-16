from collections import Counter
import numpy as np
import os
from torch.utils.data import Subset
import torch


class MDCounter(dict):
    '''
    Remark: if a number added to counter is zero, then it won't be added to it. So we re-write it
    '''
    def __init__(self, *args, **kwargs):
        super(MDCounter, self).__init__(*args, **kwargs)
        self._cktensor()
        
    def __add__(self, other):
        if not isinstance(other, MDCounter):
            return NotImplemented
        result = MDCounter()
        for elem, count in self.items():
            result[elem] = count + other[elem]

        for elem, count in other.items():
            if elem not in self:
                result[elem] = count
        return result

    def __sub__(self, other):
        if not isinstance(other, MDCounter):
            return NotImplemented
        result = MDCounter()
        for elem, count in self.items():
            result[elem] = count - other[elem]

        for elem, count in other.items():
            if elem not in self:
                result[elem] = -count
        return result

    def __mul__(self, number):
        for k in self.keys():
            self[k] *= number
        return self
    
    def __truediv__(self, number):
        for k in self.keys():
            self[k] /= number
        return self

    def append(self, item):
        for k, v in item.items():
            if k in self.keys(): 
                self[k].append(v)
            else:
                self[k] = [v]
        return self
    
    def _cktensor(self):
        for k in self.keys():
            if torch.is_tensor(self[k]):
                self[k] = self[k].item()


def warmup_lr(epoch):

    if epoch < 20:
        lr =  1 + 0.5 * epoch
    else:
        lr =  10 * 0.95**(epoch - 20)
    
    print(epoch, 'lr set to ', lr)

    return lr
