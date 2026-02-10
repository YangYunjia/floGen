'''
Aug. 29, 2024 by Yunjia Yang

API for outside quary

Currently use: SLD estimated with model (no need to call VLM codes)
- airfoil prediction model: 0330_25_Run3
- SLD / airfoil-to-wing: modelcl21_1658_Run0 (randomly pick one)

'''


from flowvae.ml_operator.operator import load_model_weights
from flowvae.ml_operator.config import ModelConfig
from flowvae.ml_operator.hf import download_model_from_hf
from flowvae.sim.cfl3d import AirfoilSimulator
from flowvae.utils import device_select
from flowvae.post import rotate_input, intergal_output
import torch

import numpy as np
import os, copy
import json
from cfdpost.wing.single_section_wing import Wing, plot_frame
from cfdpost.wing.multi_section_wing import MultiSecWing
from cst_modeling.section import cst_foil, cst_foil_fit, clustcos
from cst_modeling.io import read_plot3d, output_plot3d_concat
from typing import Optional, Tuple
from abc import abstractmethod

absolute_file_path = os.path.dirname(os.path.abspath(__file__))

def linear_interpolation(data: torch.Tensor, x: torch.Tensor, x_new: torch.Tensor) -> torch.Tensor:
    
    indices = torch.searchsorted(x, x_new, right=True) - 1
    indices = indices.clamp(0, len(x) - 2)
    x0, x1 = x[indices], x[indices + 1]
    y0, y1 = data[indices], data[indices + 1]
    alpha  = (x_new - x0) / (x1 - x0)
    data_new = y0 + alpha.unsqueeze(-1).unsqueeze(-1) * (y1 - y0)
    
    return data_new

def surface_meshing_ML(wingCoefs: dict, config=None, save_surface=False):
    '''
    only for mdolab API use
    
    '''
    
    #* surface meshing
    block = MultiSecWing._reconstruct_surface_grids(wingCoefs, 129, [129], zaxis=0.25, lower_cst_constraints=True, twists0=6.7166)
    
    global mesh_generation_index
    if save_surface:
        output_dict = {}
        for k in wingCoefs.keys():
            if isinstance(wingCoefs[k], np.ndarray):
                output_dict[k] = wingCoefs[k].tolist()
            else:
                output_dict[k] = wingCoefs[k]
        with open(f'output/input_{mesh_generation_index:d}.json', 'w') as f:
            json.dump(output_dict, f)
            
        output_plot3d_concat([copy.deepcopy(block)[None, ...].transpose(2, 3, 0, 1)], fname=f'output/wing_{mesh_generation_index:d}.xyz', order='ij', verbose=0)
        mesh_generation_index += 1


    return block

'''
api to transfer learning model

'''

class WingAPI():

    models = []
    model_save_names = []
    hf_repo_id = ''
    version_folder = {}

    def __init__(self, saves_folder: Optional[str] = None, model_version: str = 'default', device: str = 'default') -> None:
        '''
        
        paras:
        ===
        - `model_version`: 
            - `aoa`: use aoa as input first two index are: aoa, mach
            - `cl`: use cl as input first two index are: mach, cl
        '''

        _device = device_select(device)
        self.device = _device
        self.model_version = model_version
        
        self.saves_folder = saves_folder if saves_folder is not None else os.path.join('..', 'save', self.__class__.__name__)   
        self.load_model()

    @staticmethod
    def _average_outputs(outputs):
        # print('averaging ensemble outputs... from ', len(outputs), ' members')
        if len(outputs) == 0:
            raise ValueError('No outputs to average.')
        ref = outputs[0]
        if torch.is_tensor(ref):
            return torch.stack(outputs, dim=0).mean(dim=0)
        if isinstance(ref, (list, tuple)):
            avg_items = [WingAPI._average_outputs([o[i] for o in outputs]) for i in range(len(ref))]
            return type(ref)(avg_items)
        if isinstance(ref, dict):
            return {k: WingAPI._average_outputs([o[k] for o in outputs]) for k in ref.keys()}
        raise TypeError(f'Unsupported output type for ensemble averaging: {type(ref)}')

    def _wrap_ensemble(self, models):
        if len(models) == 1:
            return models[0]

        class _EnsembleWrapper:
            def __init__(self, members, avg_fn):
                self.members = members
                self.avg_fn = avg_fn

            def __call__(self, *args, **kwargs):
                outputs = [m(*args, **kwargs) for m in self.members]
                return self.avg_fn(outputs)

        return _EnsembleWrapper(models, self._average_outputs)
        
    def load_model(self) -> None:

        self.loaded_models = []

        print('loading model...')
        needed_models = []
        needed_models_save_names = []
        # select models to load according to version
        if self.model_version in self.version_folder:
            for md in self.version_folder[self.model_version]:
                assert md in self.models, f'model version {self.model_version} requires model {md}, which is not in available models for {self.__class__.__name__}'
                needed_models.append(md)
                needed_models_save_names.append(self.model_save_names[self.models.index(md)])
        else:
            raise RuntimeError(f'model version {self.model_version} not supported for {self.__class__.__name__}')


        # check and download models if needed
        for model_name, save_name in zip(needed_models, needed_models_save_names):
            if not os.path.exists(os.path.join(self.saves_folder, model_name, 'model_config')):
                print(f'downloading model {model_name} from huggingface...')
                os.makedirs(self.saves_folder, exist_ok=True)
                download_model_from_hf(repo_id=self.hf_repo_id, local_folder=self.saves_folder, model_name=model_name)

            folders = [d for d in os.listdir(os.path.join(self.saves_folder, model_name)) if os.path.isdir(os.path.join(self.saves_folder, model_name, d))]
            if len(folders) > 0:
                print(f'loading ensemble model {model_name}... with {len(folders)} members')
                # this model is ensemble
                ensemble_members = []

                for subfolder in folders:
                    model_fram = ModelConfig(os.path.join(self.saves_folder, model_name, 'model_config')).create()
                    load_model_weights(model_fram, os.path.join(self.saves_folder, model_name, subfolder, save_name), device=self.device)
                    ensemble_members.append(model_fram)
                self.loaded_models.append(self._wrap_ensemble(ensemble_members))

            else:
                print(f'loading single model {model_name}...')
                # single model
                model_fram = ModelConfig(os.path.join(self.saves_folder, model_name, 'model_config')).create()
                load_model_weights(model_fram, os.path.join(self.saves_folder, model_name, save_name), device=self.device)
                self.loaded_models.append(model_fram)

    @abstractmethod
    def end2end_predict(self, shape_paras: dict) -> dict:
        
        return {
            "geom": [],
            "value": [],
            "cl_array": []
        }

    @abstractmethod
    def predict(self, inp: torch.Tensor, cnd: torch.Tensor) -> torch.Tensor:
        pass

    def setAP(self, ap) -> None:
        '''
        set the AeroProblem for the model if needed (containing operating conditions etc.)
        
        :param ap: AeroProblem object
        :type ap: AeroProblem
        '''
        self.ap = ap
    

class SimpleWingAPI(WingAPI):

    models = ['model_2d', 'model_sld', 'model_3d']
    model_save_names = [f'checkpoint_epoch_{epoch}_weights' for epoch in [299, 899, 299]]
    hf_repo_id = 'yunplus/PI_Trans_Wings'
    
    version_folder = {
        'aoa':  ['model_2d', 'model_sld', 'model_3d'],  
        'cl':   ['model_2d', '0420_1082_Run2', '0328_589_Run1']
    }
    
    def __init__(self, saves_folder: Optional[str] = None, model_version: str = 'aoa', device: str = 'default') -> None:
        '''
        
        paras:
        ===
        - `model_version`: 
            - `aoa`: use aoa as input first two index are: aoa, mach
            - `cl`: use cl as input first two index are: mach, cl
        '''

        super().__init__(saves_folder, device)
        
        self.sa_type = 0.25 # base on 1/4 chord line
        self.input_ref = 5 # x, y, z plus cp_2d, cf_2d

        self.info = {}
        self.nondimension_coefs = self.nondim_index_values(model_version=model_version)
        self.model_version = model_version
    
    @staticmethod
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
            sld_3d = self.loaded_models[1](wing_paras)[:, 0]
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
            output_2d = self.loaded_models[0](geoms2d, code=auxs2d)[0]
            
        self.info['2d_surf'] = output_2d
        self.info['2d_geom'] = geoms2d.detach().cpu().numpy()
        self.info['2d_auxs'] = auxs2d.detach().cpu().numpy() # ma, cl, re
        self.info['3d_sld'] = sld_3d[0].detach().cpu().numpy()
            
        output_2d_corr = (output_2d * np.cos(sa4)**2).transpose(0, 1).unsqueeze(0)
        # shape: nv, nz, nx
        
        # airfoil to wing
        geoms = torch.from_numpy(wg.geom.transpose((2, 0, 1))).unsqueeze(0).float().to(self.device)
        prior_field = torch.concatenate((geoms, output_2d_corr), dim=1)
        output_3d = self.loaded_models[2](prior_field[:, : self.input_ref], code=wing_paras[:, :2])[0]
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
    
    def end2end_predict(self, data: dict) -> dict:
        
        inputs = data['condition'] + data['planform'] + data['secpara'][0] + data['csts'][0][0] + data['csts'][0][1]
        wg = self.predict(inputs)
        wg.aero_force()
        cl_array = wg.coefficients
        surfaceField = wg.get_formatted_surface().transpose(2, 0, 1)
        return {
            "geom": surfaceField[:3].tolist(),
            "value": surfaceField[3:].tolist(),
            "cl_array": cl_array.tolist()
        }
    
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

class SuperWingcoefAPI(WingAPI):
    models = ['ATcoef_L']
    model_save_names = ['best_model_weights' for _ in range(len(models))]
    hf_repo_id = 'yunplus/AeroTransformer'

    version_folder = {
        'default': ['ATcoef_L_v1'],
        # 'finetune': ['ATcoef_L_v1_FT']
    }

    def __init__(self, saves_folder = None, model_version = 'default', device = 'default'):
        super().__init__(saves_folder, model_version, device)

    def _predict(self, inp, cnd):
        return self.loaded_models[0](inp, code=cnd)[0]

    def predict(self, inp: torch.Tensor, cnd: torch.Tensor) -> torch.Tensor:
        '''
        :param inp: input wing coef points
        :type inp: torch.Tensor B x 29
        :param cnd: condition parameters
        :type cnd: torch.Tensor B x 2 (AoA, Mach)
        :return: forces (lift, drag, moment)
        :rtype: torch.Tensor B x 3

        '''
        inp, cnd = rotate_input(inp, cnd)
        inp_cen = 0.25 * (inp[..., 1:,1:] + inp[..., 1:,:-1] + inp[..., :-1,1:] + inp[..., :-1,:-1])
        outputs = self._predict(inp_cen, cnd)  # B, 3
        self.last_outputs = None

        return outputs # B, 3

class SuperWingAPI(WingAPI):

    models = ['ATsurf_L', 'ATsurf_L_v1', 
              'ATsurf_L_v1_FT', 'ATsurf_L_v1_FT_ENS', 
              'ATsurf_L_v1_FT20', 'ATsurf_L_v1_FT20_ENS', 
              'ATsurf_L_v1_SC']
    model_save_names = ['best_model_weights' for _ in range(7)]
    hf_repo_id = 'yunplus/AeroTransformer'

    version_folder = {
        'default': ['ATsurf_L_v1'],
        'finetune': ['ATsurf_L_v1_FT'],
        'ensemble': ['ATsurf_L_v1_FT_ENS'],
        'finetune20': ['ATsurf_L_v1_FT20'],
        'ensemble20': ['ATsurf_L_v1_FT20_ENS'],
        'scratch': ['ATsurf_L_v1_SC'],
    }

    def __init__(self, saves_folder = None, model_version = 'default', device = 'default'):
        super().__init__(saves_folder, model_version, device)

    def _predict(self, inp, cnd):
        return self.loaded_models[0](inp, code=cnd)[0]

    def end2end_predict(self, shape_paras: dict) -> dict:

        '''
        end2end predict from shape parameters to surface field
        
        :param shape_paras: WebWing format shape parameters
        :type shape_paras: dict
        :return: Description
        :rtype: dict
        '''

        geom_dict = {
            'SA': shape_paras['planform'][0],
            'AR': shape_paras['planform'][1],
            'TR': shape_paras['planform'][2],
            'kink': shape_paras['planform'][3],
            'rootadj': shape_paras['planform'][4],
            'tmaxs': shape_paras['secpara'][0],
            'DAs': shape_paras['secpara'][1],
            'twists': shape_paras['secpara'][2],
            'cst_u': np.array([cst[0] for cst in shape_paras['csts']]),
            'cst_l': np.array([cst[1] for cst in shape_paras['csts']])
        }

        wg = MultiSecWing(geom_dict, aoa=shape_paras['condition'][0], iscentric=True)
        wg.reconstruct_surface_grids(nx=129, nzs=[129])
        origeom, geom = wg.get_all_geometries()
        # print(origeom.shape, geom.shape)
        inputs = torch.from_numpy(geom).float().to(self.device).unsqueeze(0)
        auxs   = torch.tensor(shape_paras['condition']).float().to(self.device).unsqueeze(0)
        output = self._predict(inputs, auxs).cpu().detach().numpy()
        # print(output.shape)

        wg.read_formatted_surface(data=output[0], isinitg=False, isnormed=True)
        wg.aero_force()

        formated = wg.get_formatted_surface()
        print(formated[0].shape, formated[1].shape)
        # formated[0][..., 2] = -formated[0][..., 2]

        return {
            "geom": formated[0].transpose(2, 0, 1).tolist(),
            "value": formated[1].transpose(2, 0, 1).tolist(),
            "cl_array": wg.coefficients.tolist()
        }

    def predict(self, inp: torch.Tensor, cnd: torch.Tensor) -> torch.Tensor:
        '''
        :param inp: input surface grid points
        :type inp: torch.Tensor B x 3 x spanwise x W
        :param cnd: condition parameters
        :type cnd: torch.Tensor B x 2 (AoA, Mach)
        :return: forces (lift, drag, moment)
        :rtype: torch.Tensor B x 3

        '''
        inp, cnd = rotate_input(inp, cnd)
        inp_cen = 0.25 * (inp[..., 1:,1:] + inp[..., 1:,:-1] + inp[..., :-1,1:] + inp[..., :-1,:-1])

        outputs = self._predict(inp_cen, cnd)  # B, 3, H-1, W-1
        self.last_surface_outputs = outputs.detach().cpu().numpy()

        forces = intergal_output(inp, outputs, cnd[:, 0], 
                                 s=self.ap.areaRef, c=self.ap.chordRef, xref=self.ap.xRef, yref=self.ap.yRef)
        # forces = outputs[:, 0, 0, :2]

        return forces # B, 3
