'''

graph of C-mesh
===
    _____________________   
    / *  *__*_*_|_________|
    | * /                 |
    | * |   ^       i1    |
    | * |   | <======-----|
    | * |   |____   i0    | <- j=0
    | * \_________________|
    \_*___*__*_|__________| <- j=NJ

    * the area used to calculate average velocity field


'''



import numpy as np
import torch



from typing import List, NewType, Tuple
Tensor = NewType('Tensor', torch.tensor)

WORKCOD = {'Tinf':460.0,'Minf':0.76,'Re':5e6,'AoA':0.0,'gamma':1.4, 'x_mc':0.25, 'y_mc':0.0}

# the base rotation matrix:
#  / (1, 0)  (0, 1) \
#  \ (0,-1)  (1, 0) /
# if one want to rotate the vector (x_o, y_o) in origin coordinate (o) to the target coordinate (t), 
# the rotate matrix should be the base matrix dot the origin x unit-vector in the target coordinate.
# for example: transfer force (f_x, f_y) to lift and drag
#   - the target coor.(along the freestream) can be obtained by rotate the origin coor.(along the chord)
#     a angle of AoA c.c.w.
#   - the x unit-vector in target coor. is /  cos(AoA) \
#                                          \ -sin(AoA) /
#
#   - thus, ( Drag, Lift ) = ( f_x, f_y ) .  / (1, 0)  (0, 1) \  .  /  cos(AoA) \
#                                            \ (0,-1)  (1, 0) /     \ -sin(AoA) /
_rot_metrix = torch.Tensor([[[1.0,0], [0,1.0]], [[0,-1.0], [1.0,0]]])


#* function to rotate x-y to aoa
def _aoa_rot(aoa: Tensor):
    '''
    aoa is in size (B, )
    
    '''
    aoa = aoa * np.pi / 180
    return torch.cat((torch.cos(aoa).unsqueeze(1), -torch.sin(aoa).unsqueeze(1)), dim=1).squeeze()

def _xy_2_cl(dfp: Tensor, aoa: float):
    '''
    transfer fx, fy to CD, CL

    param:
    dfp:    (Fx, Fy), Tensor with size (2,)
    aoa:    angle of attack, float

    return:
    ===
    Tensor: (CD, CL)
    '''
    aoa = torch.FloatTensor([aoa])
    # print(dfp.size(), _rot_metrix.size(), _aoa_rot(aoa).size())
    return torch.einsum('p,prs,s->r', dfp, _rot_metrix.to(dfp.device), _aoa_rot(aoa).to(dfp.device))

def _xy_2_clc(dfp: Tensor, aoa: Tensor):
    '''
    batch version of _xy_2_cl
    
    transfer fx, fy to CD, CL

    param:
    dfp:    (Fx, Fy), Tensor with size (B, 2,)
    aoa:    angle of attack, Tensor with size (B, )

    return:
    ===
    Tensor: (CD, CL),  with size (B, 2)
    '''
    return torch.einsum('bp,prs,bs->br', dfp,  _rot_metrix.to(dfp.device), _aoa_rot(aoa).to(dfp.device))

#* function to extract information from 2-D flowfield
def get_aoa(vel):
    '''
    This function is to extract the angle of attack(AoA) from the far-field velocity field

    param:
    ===
    `vel`:   the velocity field, shape: (2 x H x W), the two channels should be U and V (x and y direction velocity)
        only the field at the front and farfield is used to averaged (see comments of post.py)

    return:
    ===
    (torch.Tensor): the angle of attack

    '''

    # inlet_avg = torch.mean(vel[:, 3: -3, -5:-2], dim=(1, 2))
    inlet_avg = torch.mean(vel[:, 100: -100, -5:-2], dim=(1, 2))
    # inlet_avg = torch.mean(vel[:, 3: -3, -1], dim=1)

    return torch.atan(inlet_avg[1] / inlet_avg[0]) / 3.14 * 180

def get_p_line(X, P, i0=15, i1=316):
    '''
    This function is to extract p values at the airfoil surface from the P field

    The surface p value is obtained by averaging the four corner values on each first layer grid

    param:
    ===
    `X`:    The X field, shape: (H x W)

    `P`:    The P field, shape: (H x W)

    `i0` and `i1`:  The position of the start and end grid number of the airfoil surface

    return:
    ===
    Tuple(torch.Tensor, list):  X, P (shape of each: (i1-i0, ))
    '''
    p_cen = []
    for j in range(i0, i1):
        p_cen.append(-0.25 * (P[j, 0] + P[j, 1] + P[j+1, 0] + P[j+1, 1]))
    return X[i0: i1, 0], p_cen

def get_vector(X: Tensor, Y: Tensor, i0: int, i1: int):
    '''
    get the geometry variables on the airfoil surface
    
    remark:
    ===
    ** `should only run once at the begining, since is very slow` **

    param:
    ===
    `X`:    The X field, shape: (H x W)

    `Y`:    The Y field, shape: (H x W)

    `i0` and `i1`:  The position of the start and end grid number of the airfoil surface

    return:
    ===
    Tuple(torch.Tensor):  `_vec_sl`, `_veclen`, `_area`

    `_vec_sl`:  shape : (i1-i0-1, 2), the surface section vector (x2-x1, y2-y1)

    `_veclen`:  shape : (i1-i0-1, ), the length of the surface section vector

    `area`:     shape : (i1-i0-1, ), the area of the first layer grid (used to calculate tau)
    '''
    _vec_sl = torch.zeros((i1-i0-1, 2,))
    _veclen = torch.zeros(i1-i0-1) 
    _area   = torch.zeros(i1-i0-1) 
    # _sl_cen = np.zeros((i1-i0-1, 2)) 

    for idx, j in enumerate(range(i0, i1-1)):
            
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

def get_force_xy(vec_sl: Tensor, veclen: Tensor, area: Tensor,
                  vel: Tensor, T: Tensor, P: Tensor, 
                  i0: int, i1: int, paras: dict, ptype: str = 'Cp'):
    '''
    integrate the force on x and y direction

    param:
    `_vec_sl`, `_veclen`, `_area`: obtained by _get_vector
    
    `vel`:   the velocity field, shape: (2 x H x W), the two channels should be U and V (x and y direction velocity)

    `T`:    The temperature field, shape: (H x W)
    
    `P`:    The pressure field, shape: (H x W); should be non_dimensional pressure field by CFL3D

    `i0` and `i1`:  The position of the start and end grid number of the airfoil surface

    `paras`:    the work condtion to non-dimensionalize; should include the key of (`gamma`, `Minf`, `Tinf`, `Re`)

    return:
    ===
    Tensor: (Fx, Fy)
    '''

    p_cen = 0.25 * (P[i0:i1-1, 0] + P[i0:i1-1, 1] + P[i0+1:i1, 0] + P[i0+1:i1, 1])
    t_cen = 0.25 * (T[i0:i1-1, 0] + T[i0:i1-1, 1] + T[i0+1:i1, 0] + T[i0+1:i1, 1])
    uv_cen = 0.5 * (vel[:, i0:i1-1, 1] + vel[:, i0+1:i1, 1])

    # if ptype == 'P':
    #     dfp_n = 1.43 / (paras['gamma'] * paras['Minf']**2) * (paras['gamma'] * p_cen - 1) * veclen
    # else:
    #     dfp_n = p_cen * veclen
    dfp_n = 1.43 / (paras['gamma'] * paras['Minf']**2) * (paras['gamma'] * p_cen - 1) * veclen
    mu = t_cen**1.5 * (1 + 198.6 / paras['Tinf']) / (t_cen + 198.6 / paras['Tinf'])
    dfv_t = 0.063 / (paras['Minf'] * paras['Re']) * mu * torch.einsum('kj,jk->j', uv_cen, vec_sl) * veclen**2 / area

    # cx, cy
    dfp = torch.einsum('lj,lpk,jk->p', torch.cat((dfv_t.unsqueeze(0), -dfp_n.unsqueeze(0)),dim=0), _rot_metrix.to(dfv_t.device), vec_sl)

    return dfp

def get_force_cl(aoa: float, **kwargs):
    '''
    get the lift and drag

    param:
    `aoa`:  angle of attack

    `_vec_sl`, `_veclen`, `_area`: obtained by _get_vector
    
    `vel`:   the velocity field, shape: (2 x H x W), the two channels should be U and V (x and y direction velocity)

    `T`:    The temperature field, shape: (H x W)
    
    `P`:    The pressure field, shape: (H x W); should be non_dimensional pressure field by CFL3D

    `i0` and `i1`:  The position of the start and end grid number of the airfoil surface

    `paras`:    the work condtion to non-dimensionalize; should include the key of (`gamma`, `Minf`, `Tinf`, `Re`)

    return:
    ===
    Tensor: (CD, CL)
    '''
    dfp = get_force_xy(**kwargs)
    fld = _xy_2_cl(dfp, aoa)
    return fld

#* function to extract pressure force from 1-d pressure profile
def get_xyforce_1d(geom: Tensor, profile: Tensor):
    '''
    integrate the force on x and y direction

    param:
    ===
    `geom`:    The geometry (x, y), shape: (2, N)
    
    `profile`: The pressure profile, shape: (N, ); should be non_dimensional pressure profile by freestream condtion

        Cp = (p - p_inf) / 0.5 * rho * U^2

    return:
    ===
    Tensor: (Fx, Fy)
    '''
    dfp_n  = 0.5 * (profile[1:] + profile[:-1]).unsqueeze(0)
    dfv_t  = torch.zeros_like(dfp_n)
    dr      = (geom[:, 1:] - geom[:, :-1]).permute(1, 0)

    return torch.einsum('lj,lpk,jk->p', torch.cat((dfv_t, -dfp_n), dim=0), _rot_metrix.to(dfv_t.device), dr)

def get_xyforce_1dc(geom: Tensor, profile: Tensor):
    '''
    batch version of integrate the force on x and y direction

    param:
    ===
    `geom`:    The geometry (x, y), shape: (B, 2, N)
    
    `profile`: The pressure profile, shape: (B, N); should be non_dimensional pressure profile by freestream condtion

        Cp = (p - p_inf) / 0.5 * rho * U^2

    return:
    ===
    Tensor: (Fx, Fy)
    '''    
    dfp_n  = 0.5 * (profile[:, 1:] + profile[:, :-1]).unsqueeze(1)
    dfv_t  = torch.zeros_like(dfp_n)
    dr     = (geom[:, :, 1:] - geom[:, :, :-1]).permute(0, 2, 1)

    return torch.einsum('blj,lpk,bjk->bp', torch.cat((dfv_t, -dfp_n), dim=1), _rot_metrix.to(dfv_t.device), dr)

def get_force_1d(geom: Tensor, profile: Tensor, aoa: float):
    '''
    integrate the lift and drag

    param:
    ===
    `geom`:    The geometry (x, y), shape: (2, N)
    
    `profile`: The pressure profile, shape: (N, ); should be non_dimensional pressure profile by freestream condtion

        Cp = (p - p_inf) / 0.5 * rho * U^2
    
    `aoa`:  angle of attack

    return:
    ===
    Tensor: (CD, CL)
    '''
    dfp = get_xyforce_1d(geom, profile)
    return _xy_2_cl(dfp, aoa)

def get_force_1dc(geom: Tensor, profile: Tensor, aoa: Tensor):
    '''
    batch version of integrate the lift and drag

    param:
    ===
    `geom`:    The geometry (x, y), shape: (B, 2, N)
    
    `profile`: The pressure profile, shape: (B, N); should be non_dimensional pressure profile by freestream condtion

        Cp = (p - p_inf) / 0.5 * rho * U^2
    
    `aoa`:  angle of attack, shape: (B, )

    return:
    ===
    Tensor: (CD, CL),  with size (B, 2)
    '''
    dfp = get_xyforce_1dc(geom, profile)
    return _xy_2_clc(dfp, aoa)

# def get_massflux_1d(xcor: Tensor, ycor: Tensor, xvel: Tensor, yvel: Tensor, rho: Tensor):
#     dx      = (xcor[1:] - xcor[:-1])
#     dy      = (ycor[1:] - ycor[:-1])
#     xvel    = 0.5 * (xvel[1:] + xvel[:-1])
#     yvel    = 0.5 * (yvel[1:] + yvel[:-1])
#     rho     = 0.5 * (rho[1:] + rho[:-1])

def get_flux_1d(xcor: Tensor, ycor: Tensor, pressure: Tensor, xvel: Tensor, yvel: Tensor, rho: Tensor):
    dx      = (xcor[1:] - xcor[:-1])
    dy      = (ycor[1:] - ycor[:-1])
    pressure = 0.5 * (pressure[1:] + pressure[:-1])
    xvel    = 0.5 * (xvel[1:] + xvel[:-1])
    yvel    = 0.5 * (yvel[1:] + yvel[:-1])
    rho     = 0.5 * (rho[1:] + rho[:-1])

    phixx = rho * xvel**2 + pressure
    phixy = rho * xvel * yvel
    phiyy = rho * yvel**2 + pressure

    mass_flux   = torch.sum(rho * xvel * dy - rho * yvel * dx)
    moment_flux = torch.zeros((2,))
    moment_flux[0] = torch.sum(phixx * dy - phixy * dx)
    moment_flux[1] = torch.sum(phixy * dy - phiyy * dx)

    return mass_flux, moment_flux

