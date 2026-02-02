
from flowvae.ml_operator import load_model_from_checkpoint
from flowvae.post import _xy_2_cl_tc, get_cellinfo_1d_t, get_cellinfo_2d_t, get_force_2d_t, get_moment_2d_t
from flowvae.app.mdolab import MLModel
import matplotlib.pyplot as plt
from flowvae.app.wing import models
from cfdpost.wing.multi_section_wing import MultiSecWing
from cst_modeling.io import read_plot3d, output_plot3d_concat

from baseclasses import AeroProblem

import os
import torch
import numpy as np
import json
import copy

from typing import Tuple

mesh_generation_index = 0

# j57e_Run1_Tranpp30
def rotate_input(inp: torch.Tensor, cnd: torch.Tensor) -> Tuple[torch.Tensor]:
    B, C, H, W = inp.shape
    
    # rotate to without baseline twist ( w.r.t centerline LE (0,0,0))
    inp = torch.cat([
        _xy_2_cl_tc(inp[:, :2].permute(0, 2, 3, 1).reshape(-1, 2), -6.7166 * torch.ones((B*H*W,))).reshape(B, H, W, 2).permute(0, 3, 1, 2),
        inp[:, 2:]
    ], dim = 1) 
    
    cnd = torch.cat([
        cnd[:, :1] + 6.7166,
        cnd[:, 1:]
    ], dim = 1)

    return inp, cnd

def intergal_output(geom: torch.Tensor, outputs: torch.Tensor, aoa: torch.Tensor,
                    s: float, c: float, xref: float, yref: float) -> torch.Tensor:
    '''
    geom: B, I, J, 3
    outputs: B, 3, I-1, J-1
    
    '''

    cp = outputs[:, 0]
    tangens, normals2d = get_cellinfo_1d_t(geom[:, :2].permute(0, 2, 3, 1))                    
    tangens = 0.5 * (tangens[:, 1:] + tangens[:, :-1])    # transfer to cell centre at spanwise direction

    cf = torch.concatenate((outputs[:, [1]].permute(0, 2, 3, 1) * tangens / 150, outputs[:, [2]].permute(0, 2, 3, 1) / 300), axis=-1)
    forces = get_force_2d_t(geom.permute(0, 2, 3, 1), aoa=aoa, cp=cp, cf=cf)[:, [1, 0]] / s
    moment = get_moment_2d_t(geom.permute(0, 2, 3, 1), cp=cp, cf=cf, 
                             ref_point=torch.Tensor([xref, yref, 0.]))[:, [2]] / s / c

    return torch.cat((forces, moment), dim=-1)

def surface_meshing_ML(wingCoefs: dict, config=None, save_surface=False):
    
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



class CoefPredictModel(MLModel):

    def __init__(self, model_run: str = '1f2e', 
                 model_subrun: int = 1,
                 step: int = 1600,
                 device = None):
        
        super().__init__(device)
        self.file_name = '%s_%s_Run%d' % ('0524', model_run, model_subrun)
        self.model = models.WingPDETransformer(patch_size=(4,4), fun_dim=3, out_dim=3, mlp_ratio=4, n_layers=5, n_hidden=16, device=device, 
                                              type_cond='inj', output_type='attn_pool')
        self.step = step
        # WingPDETransformer(patch_size=(4,4), fun_dim=3, out_dim=3, mlp_ratio=4, n_layers=5, n_hidden=16, 
        #                               type_cond='inj', output_type=None)
    
    def loading(self) -> None:
        last_error = load_model_from_checkpoint(self.model, epoch=self.step-1, folder=os.path.join('..', 'saves', self.file_name), device=self.device)


    def predict(self, inp: torch.Tensor, cnd: torch.Tensor) -> torch.Tensor:
        '''
        inp: B x 3 x H x W
        cnd: B x Nc

        return: B x 3
        
        '''
        inp, cnd = rotate_input(inp, cnd)
        inp_cen = 0.25 * (inp[..., 1:,1:] + inp[..., 1:,:-1] + inp[..., :-1,1:] + inp[..., :-1,:-1])

        outputs = self.model(inp_cen, code=cnd)[0]

        return outputs # B, 3

class SurfacePredictionModel(MLModel):
    
    def __init__(self, model_run: str = 'j57e', 
                 model_subrun: int = 0,
                 transfer_run: str = 'pp30',
                 transfer_subrun: int = 0,
                 step: int = 400,
                 ap: AeroProblem = None,
                 device = None):
        
        super().__init__(device)
        self.file_name = '%s_%s_Run%d_Tran%s_Run%d' % ('0524', model_run, model_subrun, transfer_run, transfer_subrun)
        self.model = models.WingPDETransformer(patch_size=(4,4), fun_dim=3, out_dim=3, mlp_ratio=4, n_layers=5, n_hidden=16, 
                                      type_cond='inj', output_type=None)
        self.step = step
        self.ap = ap
    
    def loading(self) -> None:
        last_error = load_model_from_checkpoint(self.model, epoch=self.step-1, folder=os.path.join('..', 'saves', self.file_name), device=self.device)

    def predict(self, inp, cnd):
        '''
        inp: B x 3 x spanwise x W
        cnd: B x Nc

        return: B x 3
        
        '''
        inp, cnd = rotate_input(inp, cnd)
        inp_cen = 0.25 * (inp[..., 1:,1:] + inp[..., 1:,:-1] + inp[..., :-1,1:] + inp[..., :-1,:-1])

        outputs = self.model(inp_cen, code=cnd)[0]
        self.last_surface_outputs = outputs.detach().cpu().numpy()

        forces = intergal_output(inp, outputs, cnd[:, 0], 
                                 s=self.ap.areaRef, c=self.ap.chordRef, xref=self.ap.xRef, yref=self.ap.yRef)
        # forces = outputs[:, 0, 0, :2]

        return forces # B, 3

    @staticmethod
    def show_slices(geom, outputs):
        for iz in [0, 60, 120]:
            plt.plot(geom[0, iz], outputs[0, iz])
        plt.show()
