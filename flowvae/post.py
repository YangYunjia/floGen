import numpy as np
import torch

from sklearn.linear_model import LinearRegression
from scipy.interpolate import PchipInterpolator as pchip
# from .vae import VanillaVAE
# from .ml_operator import AEOperator
# from .dataset import FlowDataset, CondiFlowDataset
# from .utils import warmup_lr



WORKCOD = {'Tinf':460.0,'Minf':0.76,'Re':5e6,'AoA':0.0,'gamma':1.4, 'x_mc':0.25, 'y_mc':0.0}

metrix_nt2xy = torch.Tensor([[[1.0,0], [0,1.0]], [[0,-1.0], [1.0,0]]])

def cos1(theta):
    return torch.cos(theta * np.pi / 180)

def sin1(theta):
    return torch.sin(theta * np.pi / 180)



# def get_aoa_o(U, V):

#     u_inlet_avg = torch.mean(U[3: -3, -5:-2])
#     v_inlet_avg = torch.mean(V[3: -3, -5:-2])

#     return torch.atan(v_inlet_avg / u_inlet_avg) / 3.14 * 180

def get_aoa(vel):

    # inlet_avg = torch.mean(vel[:, 3: -3, -5:-2], dim=(1, 2))
    inlet_avg = torch.mean(vel[:, 100: -100, -5:-2], dim=(1, 2))
    # inlet_avg = torch.mean(vel[:, 3: -3, -1], dim=1)
    # v_inlet_avg = torch.sum(V[3: -3, -1]) / 325

    return torch.atan(inlet_avg[1] / inlet_avg[0]) / 3.14 * 180

def get_p_line(X, P, i0=15, i1=316):
    p_cen = []
    for j in range(i0, i1):
        p_cen.append(-0.25 * (P[j, 0] + P[j, 1] + P[j+1, 0] + P[j+1, 1]))
    return X[i0: i1, 0], p_cen

def _get_vector(X, Y, j0, j1):

    _vec_sl = torch.zeros((j1-j0-1, 2,))
    _veclen = torch.zeros(j1-j0-1) 
    _area   = torch.zeros(j1-j0-1) 
    # _sl_cen = np.zeros((j1-j0-1, 2)) 

    for idx, j in enumerate(range(j0, j1-1)):
            
        point1 = torch.Tensor([X[j, 0], Y[j, 0], 0])        # coordinate of surface point j
        point2 = torch.Tensor([X[j, 1], Y[j, 1], 0]) 
        point3 = torch.Tensor([X[j + 1, 0], Y[j + 1, 0], 0])
        point4 = torch.Tensor([X[j + 1, 1], Y[j + 1, 1], 0])
                    
        vec_sl = point3 - point1                    # surface vector sl
        _veclen[idx] = torch.sqrt((vec_sl * vec_sl).sum())   # length of surface vector sl
        _vec_sl[idx] = (vec_sl / _veclen[idx])[:2]
        ddiag = torch.cross(point4 - point1, point3 - point2)
        _area[idx] = 0.5 * torch.sqrt((ddiag * ddiag).sum())
        
        # _sl_cen[idx] = 0.5 * (point1 + point3)
    
    return _vec_sl, _veclen, _area

def _get_force_xy(vec_sl, veclen, area, UV, T, P, j0: int, j1: int, paras, ptype='Cp', dev='cpu'):
    # cxy = torch.Tensor([0.0, 0.0]).to(dev)

    p_cen = 0.25 * (P[j0:j1-1, 0] + P[j0:j1-1, 1] + P[j0+1:j1, 0] + P[j0+1:j1, 1])
    t_cen = 0.25 * (T[j0:j1-1, 0] + T[j0:j1-1, 1] + T[j0+1:j1, 0] + T[j0+1:j1, 1])
    uv_cen = 0.5 * (UV[:, j0:j1-1, 1] + UV[:, j0+1:j1, 1])
    # print(p_cen, veclen)
    
    dfp_n = 1.43 / (paras['gamma'] * paras['Minf']**2) * (paras['gamma'] * p_cen - 1) * veclen
    mu = t_cen**1.5 * (1 + 198.6 / paras['Tinf']) / (t_cen + 198.6 / paras['Tinf'])
    dfv_t = 0.063 / (paras['Minf'] * paras['Re']) * mu * torch.einsum('kj,jk->j', uv_cen, vec_sl) * veclen**2 / area

    # cx, cy
    dfp = torch.einsum('lj,lpk,jk->p', torch.cat((dfv_t.unsqueeze(0), -dfp_n.unsqueeze(0)),dim=0), metrix_nt2xy, vec_sl)

    return dfp

def _xy_2_cl(dfp, aoa, dev):
    aoa = torch.FloatTensor([aoa])
    fld = torch.einsum('p,prs,s->r', dfp, metrix_nt2xy.to(dev), torch.Tensor([cos1(aoa), -sin1(aoa)]).to(dev))
    return fld

def _xy_2_clc(dfp, aoa, dev):
    metrix_aoa = torch.cat((cos1(aoa).unsqueeze(1), -sin1(aoa).unsqueeze(1)), dim=1)
    # metrix_nt2xy = metrix_nt2xy.to(dev)
    metrix_nt2xy = torch.Tensor([[[1.0,0], [0,1.0]], [[0,-1.0], [1.0,0]]]).to(dev)
    fld = torch.einsum('bp,prs,bs->br', dfp, metrix_nt2xy, metrix_aoa)
    return fld

def _get_force_cl(vec_sl, veclen, area, UV, T, P, j0: int, j1: int, paras, ptype='Cp', dev='cpu'):

    dfp = _get_force_xy(vec_sl, veclen, area, UV, T, P, j0, j1, paras, ptype, dev)
    fld = _xy_2_cl(dfp, paras['AoA'], dev)
    return fld

def _get_force_aoa(vec_sl, veclen, area, UV, T, P, j0: int, j1: int, paras, ptype='Cp', dev='cpu'):
    dfp = _get_force_xy(vec_sl, veclen, area, UV, T, P, j0, j1, paras, ptype, dev)
    print(dfp)
    cd = np.sqrt(torch.sum(dfp**2).cpu().numpy() - paras['cl']**2)
    aoa = (torch.atan(dfp[0]/dfp[1]).cpu().numpy() - np.arctan(cd/paras['cl'])) / 3.14 * 180
    return cd, aoa

def _get_force_o(_vec_sl, _veclen, _area, U, V, T, P, j0: int, j1: int, paras, ptype='Cp', dev='cpu'):
    cxy = torch.Tensor([0.0, 0.0]).to(dev)

    for idx, j in enumerate(range(j0, j1-1)):
         
        vec_sl = _vec_sl[idx]
        veclen = _veclen[idx]
        area = _area[idx]
        p_cen = 0.25 * (P[j, 0] + P[j, 1] + P[j+1, 0] + P[j+1, 1])
        t_cen = 0.25 * (T[j, 0] + T[j, 1] + T[j+1, 0] + T[j+1, 1])

        u_cen = 0.5 * (U[j, 1] + U[j+1, 1])
        v_cen = 0.5 * (V[j, 1] + V[j+1, 1])

        metrix_nt2xy = torch.Tensor([[vec_sl[0], vec_sl[1]],[-vec_sl[1], vec_sl[0]]]).to(dev)

        # pressure part, normal to wall(sl)
        ### P
        if ptype == 'P':
            dfp_n = 1.43 / (paras['gamma'] * paras['Minf']**2) * (paras['gamma'] * p_cen - 1) * veclen
        else:
            dfp_n = p_cen * veclen

        # viscous part, tang to wall(sl)
        mu = t_cen**1.5 * (1 + 198.6 / paras['Tinf']) / (t_cen + 198.6 / paras['Tinf'])
        dfv_t = 0.063 / (paras['Minf'] * paras['Re']) * mu * torch.dot(torch.Tensor([u_cen, v_cen]).to(dev),vec_sl) * veclen**2 / area

        dfp = torch.matmul(torch.Tensor([dfv_t, -dfp_n]).to(dev), metrix_nt2xy)

        cxy += dfp
        # cmz += dfp[1] * (sl_cen[0] - paras['x_mc']) - dfp[0] * (sl_cen[1] - paras['y_mc'])
    
    metrix_xy2ab = torch.Tensor([[cos1(paras['AoA']), -sin1(paras['AoA'])],[sin1(paras['AoA']), cos1(paras['AoA'])]]).to(dev)
    fld = torch.matmul(cxy, metrix_xy2ab)

    return fld
    
def get_force(X, Y, UV, T, P, j0: int, j1: int, paras, ptype='Cp'):
    _vec_sl, _veclen, _area = _get_vector(X, Y, j0, j1)
    return _get_force_cl(_vec_sl, _veclen, _area, UV, T, P, j0, j1, paras, ptype)

# old get_forch function, and have given to Wangjing
def get_force1(X, Y, U, V, T, P, j0: int, j1: int, paras, ptype='Cp'):
    '''
    Calculate cl and cd from field data

    ### Inputs:
    ```text
    Field data: X, Y, U, V, T, P
        - in ndarray (nj,nk) type
        - data should be at nodes, rather than at cell center (cfl3d -> .prt are nodes value)
    j0:     j index of the lower surface TE node
    j1:     j index of the upper surface TE node
    paras:  'gamma'    : self.gamma,
            'Minf'     : self.Minf,
            'Re'       : self.Re,
            'Tinf'     : self.Tinf,
            'AoA'      : self.AoA
    ```

    ### Return:
    ```text
    cx, cy: force coefficient of x,y dir
    cl, cd: lift coef. and drag coef.
    ```

    ### Note:

    ### Filed data (j,k) index
    ```text
    j: 1  - nj  from far field of lower surface TE to far field of upper surface TE
    j: j0 - j1  from lower surface TE to upper surface TE
    k: 1  - nk  from surface to far field (assuming pertenticular to the wall)
    '''

    cx = 0.0
    cy = 0.0
    cmz = 0.0
    # print(self.Minf, self.Re, self.Tinf)

    for j in range(j0, j1-1):
        
        point1 = np.array([X[j, 0], Y[j, 0]])        # coordinate of surface point j
        point2 = np.array([X[j, 1], Y[j, 1]]) 
        point3 = np.array([X[j + 1, 0], Y[j + 1, 0]])
        point4 = np.array([X[j + 1, 1], Y[j + 1, 1]])
        
        p_cen = 0.25 * (P[j, 0] + P[j, 1] + P[j+1, 0] + P[j+1, 1])
        t_cen = 0.25 * (T[j, 0] + T[j, 1] + T[j+1, 0] + T[j+1, 1])
        ### u,v on wall kepp origin
        # u_cen = 0.25 * (U[j, 0] + U[j, 1] + U[j+1, 0] + U[j+1, 1])
        # v_cen = 0.25 * (V[j, 0] + V[j, 1] + V[j+1, 0] + V[j+1, 1])
        ### u,v on wall set to 0
        u_cen = 0.5 * (U[j, 1] + U[j+1, 1])
        v_cen = 0.5 * (V[j, 1] + V[j+1, 1])
        
        vec_sl = point3 - point1                    # surface vector sl
        veclen = np.linalg.norm(vec_sl)   # length of surface vector sl
        vec_sl = vec_sl / veclen
        area = 0.5 * np.linalg.norm(np.cross(point4 - point1, point3 - point2))
        
        metrix_nt2xy = np.array([[vec_sl[0], vec_sl[1]],[-vec_sl[1], vec_sl[0]]])

        # pressure part, normal to wall(sl)
        ### P
        if ptype == 'P':
            dfp_n = 1.43 / (paras['gamma'] * paras['Minf']**2) * (paras['gamma'] * p_cen - 1) * veclen
        else:
            dfp_n = p_cen * veclen

        # viscous part, tang to wall(sl)
        mu = t_cen**1.5 * (1 + 198.6 / paras['Tinf']) / (t_cen + 198.6 / paras['Tinf'])
        dfv_t = 0.063 / (paras['Minf'] * paras['Re']) * mu * np.dot(np.array([u_cen, v_cen]), vec_sl) * veclen**2 / area

        # print(mu, t_cen, p_cen, np.array([u_cen,v_cen]))
        dfp = np.dot(np.array([dfv_t, -dfp_n]), metrix_nt2xy)

        cx += dfp[0]
        cy += dfp[1]

        sl_cen = 0.5 * (point1 + point3)
        cmz += dfp[1] * (sl_cen[0] - paras['x_mc']) - dfp[0] * (sl_cen[1] - paras['y_mc'])


    metrix_xy2ab = np.array([[cos1(paras['AoA']), -sin1(paras['AoA'])],[sin1(paras['AoA']), cos1(paras['AoA'])]])

    fld = np.dot(np.array([cx, cy]), metrix_xy2ab)

    return cx, cy, fld[1], fld[0], cmz

def cal_error(vae_model, fldata, allcondis, paras, pre_path, nnobst=0, nnob=50):
    cll = np.zeros((nnob, len(allcondis)))
    cdd = np.zeros((nnob, len(allcondis)))
    cmm = np.zeros((nnob, len(allcondis)))
    dcl = np.zeros((nnob, len(allcondis)))
    dcd = np.zeros((nnob, len(allcondis)))
    dcm = np.zeros((nnob, len(allcondis)))
    aoaa = np.zeros((nnob, len(allcondis)))
    daoa = np.zeros((nnob, len(allcondis)))
    loss = np.zeros((nnob, len(allcondis)))

    avg_data = torch.load(pre_path, map_location='cuda:0')
    vae_model.new_data['vec_sl'] = avg_data['vec_sl'] 
    vae_model.new_data['veclen'] = avg_data['veclen'] 
    vae_model.new_data['area'] = avg_data['area'] 
    
    # allcondisss = [0.6002, 0.6402, 0.6802, 0.7202, 0.7603, 0.8003, 0.8403, 0.8804, 0.9205, 0.9608, 1.001]
    
    for nob in range(nnob):

        for idx, ff in enumerate(fldata.all_data[nnobst+nob]):

            # result0 = vae_model.generate(fldata[nob]['flowfields'][idx].unsqueeze(0).to("cuda:0"))
            # result0 = result0[0].cpu().squeeze(0).detach().numpy()
            # print(vae_model.sample(num_samples=10, mode='giv', code=allcondis[idx], indx=nnobst+nob).detach().size())
            result2 = torch.from_numpy(ff).to('cuda:0').float()
            aoa = get_aoa(result2[4:6])
            # print(aoaa,allcondis[idx])

            result3 = torch.mean(vae_model.sample(num_samples=1, mode='giv', code=aoa, indx=nnobst+nob).detach(), axis=0)
            # print(result3.size())

            # result0 /= (N+1)
            # codee /= (N+1)
            # result = result


            loss[nob, idx] = 0.5 * torch.mean((result3[:, :, :] - result2[2:, :, :])**2).cpu().numpy()
            # print(loss[nob, idx])

            paras['AoA'] = aoa
            aoaa[nob, idx] = paras['AoA'].cpu().numpy()
            # print(result2.device)
            cd2, cl2= _get_force_cl(vae_model.new_data['vec_sl'][nob], 
                                    vae_model.new_data['veclen'][nob],
                                    vae_model.new_data['area'][nob], result2[4:6], result2[3], result2[2], j0=15, j1=316, paras=paras, ptype='P', dev='cuda:0').cpu().numpy()
            # cd2, cl2 = vae_model.new_data['coef'][nob][icode]

            # paras['AoA'] = 
            daoa[nob, idx] = get_aoa(result3[2:4]).cpu().numpy() - aoaa[nob, idx]
            cd3, cl3 = _get_force_cl(vae_model.new_data['vec_sl'][nob], 
                                    vae_model.new_data['veclen'][nob],
                                    vae_model.new_data['area'][nob], result3[2:4], result3[1], result3[0], j0=15, j1=316, paras=paras, ptype='P', dev='cuda:0').cpu().numpy()
            # paras['cl'] = allcondis[idx]
            # cd2, aoa2 = _get_force_aoa(vae_model.new_data['vec_sl'][nob], 
            #                         vae_model.new_data['veclen'][nob],
            #                         vae_model.new_data['area'][nob], result2[4:6], result2[3], result2[2], j0=15, j1=316, paras=paras, ptype='P', dev='cuda:0')
            # cd3, aoa3 = _get_force_aoa(vae_model.new_data['vec_sl'][nob], 
            #                         vae_model.new_data['veclen'][nob],
            #                         vae_model.new_data['area'][nob], result3[2:4], result3[1], result3[0], j0=15, j1=316, paras=paras, ptype='P', dev='cuda:0')

            # print(aoa2, aoa3, cd2, cd3)
            # cx0, cy0, cl0, cd0, cmz0 = PhysicalSec.get_force(result2[0], result2[1], result0[2], result0[3], result0[1], result0[0], j0=15, j1=316, paras=paras, ptype='P')
            cmz3, cmz2 = 0, 0
            # cl3, cl2 = 0, 0

            cll[nob, idx] = cl2
            cdd[nob, idx] = cd2
            cmm[nob, idx] = cmz2
            # aoaa[nob, idx] = aoa2

            dcl[nob, idx] = cl3 - cl2
            dcd[nob, idx] = cd3 - cd2
            dcm[nob, idx] = cmz3 - cmz2
            # daoa[nob, idx] = aoa3 - aoa2

        if nob % 50 == 0:
            print("{} samples is done".format(nob))

    return cll, cdd, cmm, dcl, dcd, dcm, loss, aoaa, daoa

def cal_aoa_error(vae_model, fldata, allcondis, paras, nnobst=0, nnob=50):

    aoaa = np.zeros((nnob, len(allcondis)))
    daoa = np.zeros((nnob, len(allcondis)))
    
    for nob in range(nnob):

        for idx, cd in enumerate(allcondis):

            result3 = torch.mean(vae_model.sample(num_samples=1, mode='giv', code=allcondis[idx], indx=nnobst+nob).detach(), axis=0)

            result2 = torch.from_numpy(fldata.all_data[nnobst+nob][idx]).to('cuda:0').float()

            aoaa[nob, idx] = get_aoa(result2[4:6]).cpu().numpy()

            daoa[nob, idx] = get_aoa(result3[2:4]).cpu().numpy() - aoaa[nob, idx]

        if nob % 50 == 0:
            print("{} samples is done".format(nob))

    return aoaa, daoa

def _get_xyforce_1dc(geom, profile):
    
    avg_cp  = 0.5 * (profile[:, 1:] + profile[:, :-1])
    dr      = - (geom[:, :, 1:] - geom[:, :, :-1])

    return torch.einsum('bi,bki->bk', avg_cp, dr)

def _get_xyforce_1d(geom, profile):
    
    avg_cp  = 0.5 * (profile[1:] + profile[:-1])
    dr      = - (geom[:, 1:] - geom[:, :-1])

    return torch.einsum('i,ki->k', avg_cp, dr)
    
def get_force_1d(geom, profile, aoa, dev=None):
    # geom = geom.unsquenze
    dfp = _get_xyforce_1d(geom, profile)
    fld = _xy_2_cl(dfp, aoa, dev)
    fld[1] = -fld[1]
    return fld

def get_force_1dc(geom, profile, aoa, dev=None):
    dfp = _get_xyforce_1dc(geom, profile)
    fld = _xy_2_clc(dfp, aoa, dev)
    fld[:, 1] = -fld[:, 1]
    return fld

def get_buffet(aoas, clss, cdss, d_aoa=0.1):
    f_cdcl_aoa_all = pchip(aoas, np.array(clss) / np.array(cdss))
    f_cl_aoa_all = pchip(aoas, np.array(clss))
    aoa_refs = list(np.arange(-2, 4, 0.05))

    max_i = np.argmax(f_cdcl_aoa_all(aoa_refs))
    max_aoa = aoa_refs[max_i]
    # print(max_i, max_aoa)

    linear_aoas = np.arange(max_aoa - 2.0, max_aoa, 0.1)
    reg = LinearRegression().fit(linear_aoas.reshape(-1,1), f_cl_aoa_all(linear_aoas))
    # print(reg.coef_[0], reg.intercept_)
    reg_k = reg.coef_[0]
    reg_b = reg.intercept_

    d_b = - d_aoa * reg_k

    # f_cl_aoa = pchip(aoas[max_i:], clss[max_i:])
    step_aoa = np.arange(max_aoa, aoas[-1], 0.001)

    delta = f_cl_aoa_all(step_aoa) - (reg_k * step_aoa + reg_b + d_b)

    for idx in range(len(delta)-1):
        if delta[idx] * delta[idx+1] < 0:
            aoa_buf = step_aoa[idx] + 0.001 * delta[idx] / (delta[idx] - delta[idx+1])
            cl_buf = f_cl_aoa_all(aoa_buf)
            break
    else:
        print('Warning:  buffet not found')
        aoa_buf = None
        cl_buf = None

    # print(aoa_buf, cl_buf)

    return (aoa_buf, cl_buf)