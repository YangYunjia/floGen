'''
Aug. 29, 2024 by Yunjia Yang

API for outside quary

Currently use: SLD estimated with model (no need to call VLM codes)
- airfoil prediction model: 0330_25_Run3
- SLD / airfoil-to-wing: modelcl21_1658_Run0 (randomly pick one)

'''


from flowvae.ml_operator import load_model_from_checkpoint
from flowvae.app.wing import models
from flowvae.sim.cfl3d import AirfoilSimulator
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import torch

import numpy as np
import os, copy
from cfdpost.wing.basic import Wing, KinkWing, plot_compare_2d_wing, plot_frame
from cst_modeling.section import cst_foil, cst_foil_fit, clustcos
from typing import Optional

absolute_file_path = os.path.dirname(os.path.abspath(__file__))

def nondim_index_values(model_version: str = 'aoa'):
    '''
    get non-dimensionalize index values (according to minmax of Dataset No. 17)
    return the range_value and min_value
    
    paras:
    ===
    - `model_version`: 
        - `aoa`: first two index are: aoa, mach
        - `cl`: first two index are: mach, cl
    '''
    range_values = np.array([4.98356865e+00, 1.29940258e-01, 3.49481674e+01, 2.99908783e+00, 3.99400411e+00, 7.98910782e-01, 5.99364106e+00, 1.99938913e-01,
                             5.62159090e-02, 2.37829522e-01, 1.71319984e-01, 3.83469554e-01, 7.06469046e-01, 1.10903631e+00, 1.39157365e+00, 1.18536700e+00,
                             7.59821456e-01, 5.03237704e-01, 5.85693591e-01, 7.51407350e-02, 2.25723228e-01, 2.43932506e-01, 4.56790965e-01, 4.37682678e-01,
                             5.66290589e-01, 4.64605103e-01, 4.65208148e-01, 3.67536969e-01, 4.10273588e-01])
    min_values   = np.array([1.00110293e+00, 7.20032034e-01, 7.80648300e-03, 8.60524000e-04, 6.00205655e+00, 2.01073046e-01, 3.73024600e-03, 8.00043103e-01,
                            7.37786170e-02, 3.71495620e-02, 5.57531820e-02,-2.29357000e-04,-3.49942788e-01,-1.09036313e-01,-8.00000000e-01,-1.85366995e-01,
                            -3.00000000e-01,-3.23770400e-03,-4.19250390e-02,-1.73883091e-01,-2.25613883e-01,-2.98053582e-01,-3.45981178e-01,-4.43266647e-01,
                            -3.66290589e-01,-3.13444839e-01,-5.00000000e-01,-1.67536969e-01,-8.25543820e-02])
    
    if model_version in ['cl']:
        range_values[0:2] = [1.29940258e-01, 9.78320385e-01]
        min_values[0:2]   = [7.20032034e-01, -4.35158583e-02]
    
    return range_values, min_values

def linear_interpolation(data: torch.Tensor, x: torch.Tensor, x_new: torch.Tensor) -> torch.Tensor:
    
    indices = torch.searchsorted(x, x_new, right=True) - 1
    indices = indices.clamp(0, len(x) - 2)
    x0, x1 = x[indices], x[indices + 1]
    y0, y1 = data[indices], data[indices + 1]
    alpha  = (x_new - x0) / (x1 - x0)
    data_new = y0 + alpha.unsqueeze(-1).unsqueeze(-1) * (y1 - y0)
    
    return data_new


'''
api to transfer learning model

'''
class Wing_api():
    
    version_folder = {
        'aoa':  ['0330_25_Run3', 'modelcl21_1658_Run0_cl', 'modelcl21_1658_Run0'],  
        'cl':   ['0330_25_Run3', '0420_1082_Run2', '0328_589_Run1']
    }
    
    def __init__(self, saves_folder: Optional[str] = None, model_version: str = 'aoa', device: str = 'default') -> None:
        '''
        
        paras:
        ===
        - `model_version`: 
            - `aoa`: use aoa as input first two index are: aoa, mach
            - `cl`: use cl as input first two index are: mach, cl
        '''
        if device == 'default':
            if torch.cuda.is_available():
                _device = 'cuda:0'
            elif torch.backends.mps.is_available():
                _device = 'mps'
            else:
                _device = 'cpu'

        print(f'Pytorch Backend device is set to {_device}')

        self.device = _device
        models.device = _device
        
        self.sa_type = 0.25 # base on 1/4 chord line
        self.input_ref = 5 # x, y, z plus cp_2d, cf_2d
        
        if saves_folder is None:
            saves_folder = os.path.join(absolute_file_path, 'saves')
        
        # load airfoil prediction model
        self.model_2d = models.bresnetunetmodel1(h_e=[16, 32, 64], h_d1=[64, 128], h_d2=[512, 256, 128, 64], h_out=2, h_in=1, device=_device)
        load_model_from_checkpoint(self.model_2d, epoch=299, folder=os.path.join(saves_folder, self.version_folder[model_version][0]), device=_device)
        
        # load SLD prediction model
        self.model_sld = models.triinput_simplemodel1(h_e1=[32, 32], h_e2=[16, 16], h_e3=[16], h_d1=[64, 101], nt=101)
        load_model_from_checkpoint(self.model_sld, epoch=899, folder=os.path.join(saves_folder, self.version_folder[model_version][1]), device=_device)
        
        # load airfoil-to-wing model
        self.model_3d = models.ounetbedmodel(h_e=[32, 64, 64, 128, 128, 256], h_e1=None, h_e2=None, h_d=[258, 128, 128, 64, 64, 32, 32],
                                  h_in=self.input_ref, h_out=3, de_type='cat', coder_type ='onlycond', coder_kernel=3, device=_device)
        load_model_from_checkpoint(self.model_3d, epoch=299, folder=os.path.join(saves_folder, self.version_folder[model_version][2]), device=_device)
        
        self.info = {}
        self.nondimension_coefs = nondim_index_values(model_version=model_version)
        self.model_version = model_version
    
    @staticmethod
    def display_sectional_airfoil(ax, inputs: np.ndarray, write_to_file: Optional[str] = None):
        '''
        show the airfoil profile given parameters

        - `inputs`: (`np.ndarray`) shape = (21, )
        
        >>>  0               1-10  11-20
        >>>  root_thickness, cstu, cstl
        
        
        '''
        nx = 501
        xx, yu, yl, _, rLE = cst_foil(nx, inputs[1:11], inputs[11:], x=None, t=inputs[0], tail=0.004)

        ax.plot(xx, yu, c='k')
        ax.plot(xx, yl, c='k')
        ax.set_ylim(-0.07, 0.07)

        return ax
    
    @staticmethod
    def display_wing_frame(ax, inputs: np.ndarray, write_to_file: Optional[str] = None):
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

        ax = plot_frame(ax, *inputs[:7], cst_u=inputs[7:17], cst_l=inputs[17:])
        # ax.set_aspect('equal')
        ax.view_init(elev=40, azim=30)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 7)
        ax.set_zlim(0, 1)
    
    def predict(self, inputs: np.ndarray, real_sld: Optional[np.ndarray] = None, 
                real_2dmodel: int = 0, airsim: Optional[AirfoilSimulator] = None) -> Wing:
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
        
        return:
        ===
        
        `Wing` object
        
        `self.info`: dict
            '2d_surf': n_span (101) x n_var (2, Cp/Cf) x n_i (321)
            '2d_geom': n_span (101) x n_var (1, y) x n_i (321)
            '2d_auxs':n_span (101) x n_var (3, ma/cl/re)
            '3d_sld': n_var (1, cleta) x n_span (101)
        
        '''
        wg = Wing(aoa=inputs[0])
        wg.read_formatted_geometry(inputs, ftype=1)
        wg.reconstruct_surface_grids(nx=161, nzs=[101])
        deltaindex, nondimgeom = wg.get_normalized_sectional_geom()
        nondim_inputs = (inputs - self.nondimension_coefs[1]) / self.nondimension_coefs[0]
        wing_paras = torch.from_numpy(nondim_inputs).unsqueeze(0).float().to(self.device)
        
        if real_sld is None:
            # SLD prediction
            sld_3d = self.model_sld(wing_paras)[:, 0]
        else:
            # use prescribed SLD
            sld_3d = torch.from_numpy(real_sld).unsqueeze(0).float().to(self.device)
        
        # swept theory transfer of OC and Geom
        ma3d = inputs[0 if self.model_version in ['cl'] else 1]
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
        
        if real_2dmodel > 0 and airsim is not None:
            # use cfd simulation to correct model predicted 2d models
            output_2d = self._sim_2d(real_2dmodel, airsim)
            # return output_2d
        
        else:    
            # 2D surface value prediction with pretrained model
            output_2d = self.model_2d(geoms2d, code=auxs2d)[0]
            
        self.info['2d_surf'] = output_2d
        self.info['2d_geom'] = geoms2d.detach().cpu().numpy()
        self.info['2d_auxs'] = auxs2d.detach().cpu().numpy() # ma, cl, re
        self.info['3d_sld'] = sld_3d[0].detach().cpu().numpy()
            
        output_2d_corr = (output_2d * np.cos(sa4)**2).transpose(0, 1).unsqueeze(0)
        # shape: nv, nz, nx
        
        # airfoil to wing
        geoms = torch.from_numpy(wg.geometry.transpose((2, 0, 1))).unsqueeze(0).float().to(self.device)
        prior_field = torch.concatenate((geoms, output_2d_corr), dim=1)
        output_3d = self.model_3d(prior_field[:, : self.input_ref], code=wing_paras[:, :2])[0]
        output_3d[:, :2] += output_2d_corr
        
        wg.read_formatted_surface(geometry=None, data=output_3d[0].detach().cpu().numpy(), isnormed=True)
        # wg.lift_distribution()
        
        if self.model_version in ['cl']:
            # when input is lift coefficient, the surface distributions are first generated by the model
            # to get the angle of attack, it is searched in -2 to 16 deg., a CL is calculated from the 
            # surface distribution and AOA, to match the given CL
            
            aoa_range = [-2, 16]
            aoa0 = aoa_range[0]
            aoa1 = aoa_range[1]
            daoa = (aoa1 - aoa0) / 2.
            cltarg = inputs[1]
            while daoa > 0.01:
                aoas = np.linspace(aoa0, aoa1, 3)
                cl_left = -10.
                for i in range(len(aoas)):
                    wg.aoa = aoas[i]
                    wg.lift_distribution()
                    cl_right = wg.cl[0]
                    if i > 0 and (cl_left - cltarg) * (cl_right - cltarg) < 0:
                        aoa0 = aoas[i-1]
                        aoa1 = aoas[i]
                        daoa = (aoa1 - aoa0) / 2.
                        break
                    cl_left = cl_right
                else:
                    raise RuntimeError('not found aoa in -2.0 ~ 16.0')

            # another method is use the total force and given lift to calculate drag
            # from triangle relation. It has a little larger error.
            
            # wg.aoa = 0
            # wg.lift_distribution()
            # print(wg.cl)
            # calc_f = (wg.cl[0]**2 + wg.cl[1]**2)**0.5
            # wg.aoa = (np.arctan2(wg.cl[1], wg.cl[0]) - np.arccos(inputs[1] / calc_f)) / np.pi * 180
            # wg.lift_distribution()
        
        return wg
    
    def _sim_2d(self, n_cfd: int, airsim: AirfoilSimulator) -> torch.Tensor:
        '''
        simulate the typical airfoils of a wing to get the 2D results
        
        paras:
        ===
        - `n_cfd`:  amount of cross-sections for CFD simulation (evenly distributed spanwise)
        - `airsim`: airfoil simulator defined in `sim.cfl3d`
        
        return:
        ===
        `torch.Tensor` of size nz x nv (cp/cf) x ni (101 x 2 x 321)

        
        '''
        nn = 161
        xx = np.array([clustcos(i, nn) for i in range(nn)])
        # all_x = xx[::-1] + xx[1:]

        # decide the cross-section index to be simulated
        idxs = []
        izs = [int(iz) for iz in np.linspace(0, 1, n_cfd) * 100]
        
        # submit simulation tasks
        for iz in izs:
            # reconstruct 
            cst_u, cst_l = cst_foil_fit(xu=xx, yu=self.info['2d_geom'][iz, 0, 160:], xl=xx, yl=self.info['2d_geom'][iz, 0, :161][::-1], n_cst=7)

            i = airsim.submit({
                'igroup':   0,
                'icond':    iz,
                'ma':       self.info['2d_auxs'][iz][0],
                're':       self.info['2d_auxs'][iz][2],
                'cl':       self.info['2d_auxs'][iz][1],
                'cstu':     cst_u,
                'cstl':     cst_l
            })
            
            # i = airsim.submit({
            #     'igroup':   0,
            #     'icond':    iz,
            #     'ma':       self.info['2d_auxs'][iz][0],
            #     're':       self.info['2d_auxs'][iz][2],
            #     'cl':       self.info['2d_auxs'][iz][1],
            #     'xx':       xx,
            #     'yu':       self.info['2d_geom'][iz, 0, 160:],
            #     'yl':       self.info['2d_geom'][iz, 0, :161][::-1]
            # })
            
            idxs.append(i)
        
        surfaces = airsim.wait_and_get_surface(idxs)[:, 1:]

        # read sectional airfoil results and interpolate to entire spanwise
        izs_t = torch.tensor(izs).to('cuda:0')
        dsurfaces = torch.from_numpy(surfaces).float().to('cuda:0') - self.info['2d_surf'][izs_t]
        corr_2d = self.info['2d_surf'] + linear_interpolation(dsurfaces, izs_t, torch.tensor(range(101)).to('cuda:0'))
        # corr_2d = torch.transpose(corr_2d, 0, 1)
        self.info['2d_sec_surf'] = surfaces
 
        return corr_2d
        
        