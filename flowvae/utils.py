
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

class warmup_lr():
    
    def __init__(self, epoch_ratio=1, base=0.95):
        self.epoch_ratio = epoch_ratio
        self.base = base
    
    def __call__(self, epoch):
        epoch /= self.epoch_ratio
        if epoch < 20:
            lr =  1 + 0.5 * epoch
        else:
            lr =  10 * self.base**(epoch - 20)
        return lr

# def warmup_lr(epoch, epoch_ratio=1):
#     epoch /= epoch_ratio
#     if epoch < 20:
#         lr =  1 + 0.5 * epoch
#     else:
#         lr =  10 * 0.95**(epoch - 20)
#     return lr


def warmup_lr_4(epoch):
    epoch /= 4
    if epoch < 20:
        lr =  1 + 0.5 * epoch
    else:
        lr =  10 * 0.95**(epoch - 20)
    return lr

def warmup_plt_lr(epoch):

    if epoch < 20:
        lr =  1 + 0.5 * epoch
    elif epoch < 50:
        lr = 10
    elif epoch < 100:
        lr = 1
    elif epoch < 150:
        lr = 0.1
    elif epoch < 200:
        lr = 0.01
    elif epoch < 250:
        lr = 0.001
    else:
        lr = 0.0001
    return lr

def device_select(device: str) -> str:
    if device == 'default':
        if torch.cuda.is_available():
            _device = 'cuda:0'
        elif torch.backends.mps.is_available():
            _device = 'mps'
        else:
            _device = 'cpu'

    else:
        _device = device

    return _device