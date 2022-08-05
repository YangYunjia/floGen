from collections import Counter
import numpy as np
import os
from torch.utils.data import Subset
import torch


# def seq_split(dataset, lengths):
#     r"""
#     Randomly split a dataset into non-overlapping new datasets of given lengths.

#     Arguments:
#         dataset (Dataset): Dataset to be split
#         lengths (sequence): lengths of splits to be produced
#     """
#     if sum(lengths) != len(dataset):
#         raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

#     indices = randperm(sum(lengths)).tolist()
#     return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]


class MDCounter(Counter):
    def __init__(self, *args, **kwargs):
        super(MDCounter, self).__init__(*args, **kwargs)
        self._cktensor()
        

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

class Tecploter:
    def __init__(self, size = None, 
                       varlist = ["p"], 
                       contourlevel = [0,1],
                       mesh_output = False,
                       contour_output = True,
                       **kwargs):
        self.paras = kwargs
        self.paras['size'] = size
        self.paras['varlist'] = ["X","Y","Z","I","J","K"] + varlist
        self.paras['contourlevel'] = contourlevel
        self.paras['mesh_output'] = mesh_output
        self.paras['contour_output'] = contour_output
    
    def _wrt_mcr(self, dat_name):
        with open('draw_contour.mcr', 'w') as f:
            f.write("#!MC 1410\n")
            f.write("$!ReadDataSet  \'\".\\%s\" \'\n" % (dat_name + '.dat'))
            f.write("  ReadDataOption = New\n  ResetStyle = Yes\n  VarLoadMode = ByName\n  AssignStrandIDs = Yes\n")
            f.write("VarNameList = \'")
            for var in self.paras['varlist']:
                f.write("\"%s\"" % var)
            f.write("\'\n")
            f.write("$!FieldLayers ShowContour = Yes\n$!SetContourVar \n  Var = %d\n  ContourGroup = 1\n" % len(self.paras['varlist']))
            f.write("$!ContourLevels New\n  ContourGroup = 1\n  RawData\n")
            f.write("%d\n" % len(self.paras['contourlevel']))
            for lv in self.paras['contourlevel']:
                f.write("%f\n" % lv)
            if self.paras['size'] is not None:
                f.write("$!TwoDAxis AxisMode = Independent\n")
                f.write("$!TwoDAxis XDetail{RangeMin = %f}\n" % self.size[0][0])
                f.write("$!TwoDAxis XDetail{RangeMax = %f}\n" % self.size[0][1])
                f.write("$!TwoDAxis YDetail{RangeMin = %f}\n" % self.size[1][0])
                f.write("$!TwoDAxis YDetail{RangeMax = %f}\n" % self.size[1][1])
            if self.paras['contour_output']:
                self._wrt_output_mcr(f, dat_name + '_contour.png')
            if self.paras['mesh_output']:
                f.write("$!FieldLayers ShowMesh = Yes")
                self._wrt_output_mcr(f, dat_name + '_mesh.png')

    
    def _wrt_output_mcr(self, f, name):
        f.write("$!ExportSetup ImageWidth = 970\n")
        f.write("$!ExportSetup ExportFName = \'.\\%s\'\n" % name)
        f.write("$!Export\n")
        f.write("  ExportRegion = CurrentFrame\n")

    def _wrt_data_file(self, x, y, field, name):
        # 如果是二维则转换为三维的格式
        if len(np.shape(x)) == 2:
            x = x.reshape(np.shape(x) + (1,))
            y = y.reshape(np.shape(y) + (1,))
            field = field.reshape(np.shape(field) + (1,))
        nx, ny, nz = np.shape(x)
        
        with open("{}.dat".format(name), 'w') as f:
            f.write("variables = ")
            for var in self.paras['varlist']:
                f.write("%s " % var)
            
            f.write("\nzone i= {}  j= {}  k= {} F=POINT\n".format(nz, nx, ny))
            x = x.T
            y = y.T
            field = field.T
            for ind, val in np.ndenumerate(x):
                k, i, j = ind
                f.write("{} {} {} {} {} {} {} \n".format(val, y[ind], 0.5, k+1, j+1, i+1, field[ind]))

    def plot_dat(self, dat_name):
        self._wrt_mcr(dat_name)
        os.system("tec360 -b {}".format("draw_contour.mcr"))
    
    def plot(self, x, y, field, name):
        self._wrt_data_file(x, y, field, name)
        self.plot_dat(name)

def warmup_lr(epoch):

    if epoch < 20:
        lr =  1 + 0.5 * epoch
    else:
        lr =  10 * 0.95**(epoch - 20)
    
    print(epoch, 'lr set to ', lr)

    return lr
