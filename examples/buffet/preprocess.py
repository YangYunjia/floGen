'''

several thing done in preprocess section:

1. the index of data is reorganized, the index information is rewrited to the `index.npy`
    the size is : (27514, 9), for each flowfield
    0           1           2           3      4          5       6    7    8
    foil idx    condi idx   ref idx     aoa    ref aoa    ref cl  cl   cd   buffet idx       

2. extract wall pressure from the flowfield data

3. interpolate thickness and pressure to the given x position
'''


from cfdpost.cfdresult import cfl3d
from cst_modeling.section import clustcos

import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator as pchip

import os
import numpy as np


if __name__ == '__main__':


    file_path = 'D:\\DeepLearning\\202205SFYdata\\dataset\\dataset_b01\\flowfield'

    data = []
    idx  = []

    nn = 201
    xx = [clustcos(i, nn) for i in range(nn)]

    for root, dirs, files in os.walk(file_path):

        for file_name in files:

            #* 1. read indexs
            foil_idx = int(file_name[6:11])
            aoa_idx = int(file_name[16:18]) - 1

            with open('D:\\DeepLearning\\202205SFYdata\\dataset\\dataset_b01\\performance\\p%s.dat' % file_name[1:15], 'r') as f:
                lines = f.readlines()
                i_buffet = int(lines[1].split()[0]) - 1
                i_ref    = int(lines[3].split()[0]) - 1

                aoa      = float(lines[aoa_idx + 5].split()[1])
                cl       = float(lines[aoa_idx + 5].split()[2])
                cd       = float(lines[aoa_idx + 5].split()[3])

                aoa_ref  = float(lines[i_ref + 5].split()[1])
                cl_ref   = float(lines[i_ref + 5].split()[2])

            idx.append([foil_idx, aoa_idx, i_ref, aoa, aoa_ref, cl_ref, cl, cd,  i_buffet])

            #* 2. extract wall pressure

            suc, field, foil = cfl3d.readprt_foil(file_path, j0=1, j1=385, fname=file_name, coordinate='xz')
            print(file_name, foil_idx, aoa_idx, suc)

            #* 3. interpolate

            iLE = np.argmin(foil[0])

            # lower surface
            fy = pchip(foil[0][:iLE+1][::-1], foil[1][:iLE+1][::-1])
            fp = pchip(foil[0][:iLE+1][::-1], foil[2][:iLE+1][::-1])
            y_l = fy(xx)
            p_l = fp(xx)

            # upper surface]
            fy = pchip(foil[0][iLE:-32], foil[1][iLE:-32])
            fp = pchip(foil[0][iLE:-32], foil[2][iLE:-32])
            y_u = fy(xx)
            p_u = fp(xx)

            data.append([np.concatenate((y_l[::-1], y_u[1:]), axis=0), np.concatenate((p_l[::-1], p_u[1:]), axis=0)])
    
    idx_ar = np.array(idx)
    data_ar = np.array(data)

    print(idx_ar.shape, data_ar.shape)

    np.save('D:\\DeepLearning\\202205SFYdata\\index', idx_ar)
    np.save('D:\\DeepLearning\\202205SFYdata\\data', data_ar)

    #* show all files
    n_data = np.load('D:\\DeepLearning\\202205SFYdata\\data.npy')

    for n_foil in n_data:
        plt.plot(np.concatenate((xx[::-1], xx[1:]), axis=0), -n_foil[1], c='grey')
    plt.show()
