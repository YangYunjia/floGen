'''
Aug. 29, 2024 by Yunjia Yang

API for outside quary

Currently use: SLD estimated with model (no need to call VLM codes)
- airfoil prediction model: 0330_25_Run3
- SLD / airfoil-to-wing: modelcl21_1658_Run0 (randomly pick one)

'''


from flowvae.ml_operator import load_model_from_checkpoint
from flowvae.app.wing import models
import matplotlib.pyplot as plt
import torch

import numpy as np
import os, copy
from cfdpost.wing.basic import Wing, KinkWing, plot_compare_2d_wing, plot_frame
from cst_modeling.section import cst_foil

absolute_file_path = os.path.dirname(os.path.abspath(__file__))

def nondim_index_values(index):
    '''
    get non-dimensionalize index values (according to minmax of Dataset No. 17)
    
    paras:
    ===
    - `index`: normal order
    '''
    range_values = np.array([4.98356865e+00, 1.29940258e-01, 3.49481674e+01, 2.99908783e+00, 3.99400411e+00, 7.98910782e-01, 5.99364106e+00, 1.99938913e-01,
                             5.62159090e-02, 2.37829522e-01, 1.71319984e-01, 3.83469554e-01, 7.06469046e-01, 1.10903631e+00, 1.39157365e+00, 1.18536700e+00,
                             7.59821456e-01, 5.03237704e-01, 5.85693591e-01, 7.51407350e-02, 2.25723228e-01, 2.43932506e-01, 4.56790965e-01, 4.37682678e-01,
                             5.66290589e-01, 4.64605103e-01, 4.65208148e-01, 3.67536969e-01, 4.10273588e-01])
    min_values   = np.array([1.00110293e+00, 7.20032034e-01, 7.80648300e-03, 8.60524000e-04, 6.00205655e+00, 2.01073046e-01, 3.73024600e-03, 8.00043103e-01,
                            7.37786170e-02, 3.71495620e-02, 5.57531820e-02,-2.29357000e-04,-3.49942788e-01,-1.09036313e-01,-8.00000000e-01,-1.85366995e-01,
                            -3.00000000e-01,-3.23770400e-03,-4.19250390e-02,-1.73883091e-01,-2.25613883e-01,-2.98053582e-01,-3.45981178e-01,-4.43266647e-01,
                            -3.66290589e-01,-3.13444839e-01,-5.00000000e-01,-1.67536969e-01,-8.25543820e-02])
    
    return (index - min_values) / range_values

class Wing_api():
    
    def __init__(self, saves_folder: str = None, device: str = 'cuda:0') -> None:
        
        self.device = device
        models.device = device
        
        self.sa_type = 0.25 # base on 1/4 chord line
        self.input_ref = 5
        
        if saves_folder is None:
            saves_folder = os.path.join(absolute_file_path, 'saves')
        
        # load airfoil prediction model
        self.model_2d = models.bresnetunetmodel(h_e=[16, 32, 64], h_d1=[64, 128], h_d2=[512, 256, 128, 64], h_out=2, h_in=1, device=device)
        load_model_from_checkpoint(self.model_2d, epoch=299, folder=os.path.join(saves_folder, '0330_25_Run3'), device=device)
        
        # load SLD prediction model
        self.model_sld = models.triinput_simplemodel1(h_e1=[32, 32], h_e2=[16, 16], h_e3=[16], h_d1=[64, 101], nt=101)
        load_model_from_checkpoint(self.model_sld, epoch=899, folder=os.path.join(saves_folder, 'modelcl21_1658_Run0_cl'), device=device)
        
        # load airfoil-to-wing model
        self.model_3d = models.ounetbedmodel(h_e=[32, 64, 64, 128, 128, 256], h_e1=None, h_e2=None, h_d=[258, 128, 128, 64, 64, 32, 32],
                                  h_in=self.input_ref, h_out=3, de_type='cat', coder_type ='onlycond', coder_kernel=3, device=device)
        load_model_from_checkpoint(self.model_3d, epoch=299, folder=os.path.join(saves_folder, 'modelcl21_1658_Run0'), device=device)
    
    @staticmethod
    def display_sectional_airfoil(inputs: np.ndarray, write_to_file: str = None):
        '''
        show the airfoil profile given parameters

        - `inputs`: (`np.ndarray`) shape = (21, )
        
        >>>  0               1-10  11-20
        >>>  root_thickness, cstu, cstl
        
        
        '''
        nx = 501
        xx, yu, yl, _, rLE = cst_foil(nx, inputs[1:11], inputs[11:], x=None, t=inputs[0], tail=0.004)

        plt.figure(figsize=(5, 2), dpi=100)
        plt.plot(xx, yu, c='k')
        plt.plot(xx, yl, c='k')
        plt.savefig(write_to_file)

        return xx, yu, yl, rLE
    
    @staticmethod
    def display_wing_frame(inputs: np.ndarray, write_to_file: str = None):
        '''
        show the airfoil profile given parameters

        - `inputs`: (`np.ndarray`) shape = (27, )
        
        >>>  Wing planform parameters
        >>>  0            1               2             3             4                5
        >>>  swept_angle, dihedral_angle, aspect_ratio, tapper_ratio, tip_twist_angle, tip2root_thickness_ratio
        >>>  sectional airfoil parameters
        >>>  6               7-16  17-26
        >>>  root_thickness, cstu, cstl
        
        '''

        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(projection='3d')
        ax = plot_frame(ax, *inputs[:7], cst_u=inputs[7:17], cst_l=inputs[17:])
        ax.set_aspect('equal')
        ax.view_init(elev=40, azim=30)
        # ax.set_zlim(0, 1)
        plt.savefig(write_to_file)
        plt.show()
        
    def predict(self, inputs: np.ndarray) -> Wing:
        '''
        predict wing surface values from input
        
        para:
        ===
        
        - `inputs`: (`np.ndarray`)  shape = (29, )

        >>>  Operating conditions
        >>>  0    1     
        >>>  AoA, Mach
        >>>  Wing planform parameters
        >>>  2            3               4             5             6                7
        >>>  swept_angle, dihedral_angle, aspect_ratio, tapper_ratio, tip_twist_angle, tip2root_thickness_ratio
        >>>  sectional airfoil parameters
        >>>  8               9-18  19-28
        >>>  root_thickness, cstu, cstl
        
        '''
        wg = Wing()
        wg.read_formatted_geometry(inputs, ftype=1)
        wg.reconstruct_surface_grids(nx=161, nzs=[101])
        deltaindex, nondimgeom = wg.get_normalized_sectional_geom()
        nondim_inputs = nondim_index_values(inputs)
        
        # SLD prediction
        wing_paras = torch.from_numpy(nondim_inputs).unsqueeze(0).float().to(self.device)
        sld_3d = self.model_sld(wing_paras)[:, 0]
        
        # swept theory transfer of OC and Geom
        ma3d = inputs[1]
        re3d = 6.429
        
        if self.sa_type > 0.:
            sa4 = wg.swept_angle(self.sa_type) / 180*np.pi
        else:
            sa4 = 0.
            
        sld_2d = (sld_3d / np.cos(sa4)**2)
        re2ds = torch.from_numpy(re3d * np.cos(sa4) * wg.sectional_chord_eta(np.linspace(0, 1, 101))).float().to(self.device)
        # print(sld_2d.size(), re2ds.size())
        auxs2d = torch.hstack(((torch.ones_like(sld_2d, device=self.device) * ma3d * np.cos(sa4)).reshape(-1, 1), 
                                sld_2d.reshape(-1, 1), 
                                re2ds.reshape(-1, 1)))
        geoms2d = torch.from_numpy(nondimgeom[:, 1:2] / np.cos(sa4)).float().to(self.device)
        
        # 2D surface value prediction and correction
        output_2d = self.model_2d(geoms2d, code=auxs2d)[0]
        output_2d_corr = (output_2d * np.cos(sa4)**2).transpose(0, 1).unsqueeze(0)
        # shape: nv, nz, nx
        
        # airfoil to wing
        geoms = torch.from_numpy(wg.geometry.transpose((2, 0, 1))).unsqueeze(0).float().to(self.device)
        prior_field = torch.concatenate((geoms, output_2d_corr), dim=1)
        output_3d = self.model_3d(prior_field[:, : self.input_ref], code=wing_paras[:, :2])[0]
        output_3d[:, :2] += output_2d_corr
        
        wg.read_formatted_surface(geometry=None, data=output_3d[0].detach().cpu().numpy(), isnormed=True)
        wg.lift_distribution()
        
        self.info = {'2d_corr': output_2d_corr, '3d_sld': sld_3d}
        
        return wg
        