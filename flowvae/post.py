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

from typing import List, NewType, Tuple, Dict, Union
Tensor = NewType('Tensor', torch.Tensor)

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

#* here collect the Tensor version
#  original numpy version can be found in cfdpost.utils

_rot_metrix = torch.Tensor([[[1.0,0], [0,1.0]], [[0,-1.0], [1.0,0]]])

#* function to rotate x-y to aoa

def _aoa_rot_t(aoa: Tensor) -> Tensor:
    '''
    aoa is in size (B, )
    
    '''
    aoa = aoa * np.pi / 180
    return torch.cat((torch.cos(aoa).unsqueeze(1), -torch.sin(aoa).unsqueeze(1)), dim=1)#.squeeze(-1)

def _xy_2_cl_t(dfp: Tensor, aoa: float) -> Tensor:
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
    # print(dfp.size(), _rot_metrix.size(), _aoa_rot_t(aoa).size())
    return torch.einsum('p,prs,s->r', dfp, _rot_metrix.to(dfp.device), _aoa_rot_t(aoa).squeeze().to(dfp.device))

def _xy_2_cl_tc(dfp: Tensor, aoa: Tensor) -> Tensor:
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
    # print(dfp.shape, _rot_metrix.shape, aoa.shape, _aoa_rot_t(aoa).shape)
    return torch.einsum('bp,prs,bs->br', dfp,  _rot_metrix.to(dfp.device), _aoa_rot_t(aoa).to(dfp.device))

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
                  i0: int, i1: int, paras: Dict, ptype: str = 'Cp'):
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
# numpy.ndarray version in `cfdpost.utils`
def get_dxyforce_1d_t(geom: Tensor, cp: Tensor, cf: Tensor=None) -> Tensor:
    '''
    integrate the force on each surface grid cell, batch data
    
    paras:
    ---
    - `geom`  Tensor  (B, N, 2) -> (x, y)
    - `cp`    Tensor  (B, N)
    - `cf`    Tensor  (B, N), default is `None`

    ### retrun
    Tensor (B, N-1, 2) -> (dFx, dFy)
    
    '''
    
    dfp_n  = (0.5 * (cp[:, 1:] + cp[:, :-1])).unsqueeze(1)
    if cf is None:
        dfv_t  = torch.zeros_like(dfp_n)
    else:
        dfv_t = (0.5 * (cf[:, 1:] + cf[:, :-1])).unsqueeze(1)

    dr     = (geom[:, 1:] - geom[:, :-1])
    # print(torch.cat((dfv_t, -dfp_n), dim=1).shape, dr.shape)
    return torch.einsum('blj,lpk,bjk->bjp', torch.cat((dfv_t, -dfp_n), dim=1), _rot_metrix.to(dfv_t.device), dr)

def get_xyforce_1d_t(geom: Tensor, cp: Tensor, cf: Tensor=None) -> Tensor:
    '''
    integrate the force on x and y direction

    param:
    ===
    - `geom`  Tensor  (B, N, 2) -> (x, y)
    - `cp`    Tensor  (B, N)
        The pressure profile; should be non_dimensional pressure profile by freestream condtion
        
        `Cp = (p - p_inf) / 0.5 * rho * U^2`
        
    - `cf`    Tensor  (B, N), default is `None`
        The friction profile; should be non_dimensional pressure profile by freestream condtion
        
        `Cf = tau / 0.5 * rho * U^2`

    return:
    ===
    Tensor: (B, 2) -> (Fx, Fy)
    '''

    dr_tail = geom[:, 0] - geom[:, -1]
    dfp_n_tail = 0.5 * (cp[:, 0] + cp[:, -1]).unsqueeze(1)
    dfv_t_tail = torch.zeros_like(dfp_n_tail)
    
    force_surface = torch.sum(get_dxyforce_1d_t(geom, cp, cf), dim=1)
    force_tail = torch.einsum('bl,lpk,bk->bp', torch.cat((dfv_t_tail, -dfp_n_tail), dim=1), _rot_metrix.to(dfp_n_tail.device), dr_tail)
    
    return force_surface + force_tail

def get_force_1d_t(geom: Tensor, aoa: Tensor, cp: Tensor, cf: Tensor=None) -> Tensor:
    '''
    batch version of integrate the lift and drag

    param:
    ===
    - `geom`  Tensor  (B, N, 2) -> (x, y)
    - `cp`    Tensor  (B, N)
        The pressure profile; should be non_dimensional pressure profile by freestream condtion
        
        `Cp = (p - p_inf) / 0.5 * rho * U^2`
        
    - `cf`    Tensor  (B, N), default is `None`
        The friction profile; should be non_dimensional pressure profile by freestream condtion
        
        `Cf = tau / 0.5 * rho * U^2`
        
    - `aoa`   Tensor (B,), in angle degree

    return:
    ===
    Tensor: (B, 2) -> (CD, CL)
    '''
    
    dfp = get_xyforce_1d_t(geom, cp, cf)
    return _xy_2_cl_tc(dfp, aoa)

def get_flux_1d_t(geom: Tensor, pressure: Tensor, xvel: Tensor, yvel: Tensor, rho: Tensor) -> Tensor:
    '''
    obtain the mass and momentum flux through a line

    param:
    ===
    `geom`:    The geometry (x, y), shape: (2, N)
    
    `pressure`: The pressure on every line points, shape: (N, ); should be dimensional pressure profile
    
    `xvel`: x-direction velocity on every line points, shape: (N, )

    `yvel`: y-direction velocity on every line points, shape: (N, )

    `rho`: density on every line points, shape: (N, )

    return:
    ===
    Tensor: (mass_flux, moment_flux)
    '''
    
    dx      = (geom[0, 1:] - geom[0, :-1])
    dy      = (geom[1, 1:] - geom[1, :-1])
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

#* functions to get force from 2-D surfaces

def get_cellinfo_2d_t(geom: Tensor) -> Tuple[Tensor]:
    
    '''
    get the normal vector and area of each surface grid cell
    :param geom: The geometry (x, y, z)
    :type geom: torch.Tensor (..., I, J, 3)
    :return: normals and areas
    :rtype: Tuple(torch.Tensor, torch.Tensor), shape (..., I-1, J-1, 3), (..., I-1, J-1)
    '''
    
    # get corner points（p0, p1, p2, p3）
    p0 = geom[..., :-1, :-1, :] # SW
    p1 = geom[..., :-1, 1:, :]  # SE
    p2 = geom[..., 1:, 1:, :]   # NW
    p3 = geom[..., 1:, :-1, :]  # NE

    # calculate two groups of normal vector and average
    normals = torch.cross(p2 - p0, p3 - p1, dim=-1)
    areas   = 0.5 * (torch.linalg.norm(torch.cross(p1 - p0, p2 - p0, dim=-1), dim=-1) + torch.linalg.norm(torch.cross(p2 - p0, p3 - p0, dim=-1), dim=-1))

    # normalization
    normals = normals / (torch.linalg.norm(normals, dim=-1, keepdim=True) + 1e-20)
    # print(np.sum(normals * areas[..., np.newaxis], axis=(0,1)))
    return normals, areas    

def get_dxyforce_2d_t(geom: Union[Tensor, List[Tensor]], cp: Tensor, cf: Tensor=None) -> Tensor:
    '''
    integrate forces from 2D surface data on every surface grid cell
    
    :param geom: The geometry (x, y, z)
    :type geom: torch.Tensor (..., I, J, 3)
    :param cp: pressure coefficients Cp = (p - p_inf) / 0.5 * rho * U_\infty^2
    :type cp: torch.Tensor (..., I-1, J-1)
    :param cf: friction coefficients Cf = (tau @ n) / 0.5 * rho * U_\infty^2
    :type cf: torch.Tensor (..., I-1, J-1, 3)
        
    :return: coefficients of forces in x, y, z directions
    :rtype: torch.Tensor, (dCx, dCy, dCz), shape (..., I-1, J-1, 3)
    
    '''
    # calculate normal vector
    if isinstance(geom, list):
        n, a = geom
    else:
        n, a = get_cellinfo_2d_t(geom)
    dfp = cp[..., None] * n * a[..., None]
    
    if not (cf is None or len(cf) == 0):
        shear = (cf - torch.sum(cf * n, dim=-1, keepdim=True) * n) * a[..., None]
        dfp = dfp + shear
    
    return dfp

def get_xyforce_2d_t(geom: Union[Tensor, List[Tensor]], cp: Tensor, cf: Tensor=None) -> Tensor:
    '''
    integrate forces from 2D surface data
    
    :param geom: The geometry (x, y, z)
    :type geom: torch.Tensor (..., I, J, 3)
    :param cp: pressure coefficients Cp = (p - p_inf) / 0.5 * rho * U_\infty^2
    :type cp: torch.Tensor (..., I-1, J-1)
    :param cf: friction coefficients Cf = (tau @ n) / 0.5 * rho * U_\infty^2
    :type cf: torch.Tensor (..., I-1, J-1, 3)
        
    :return: coefficients of forces in x, y, z directions
    :rtype: torch.Tensor (CX, CY, CZ)
    '''
    return torch.sum(get_dxyforce_2d_t(geom, cp, cf), dim=(-3,-2))

def get_force_2d_t(geom: Union[Tensor, List[Tensor]], aoa: Tensor, cp: Tensor, cf: Tensor=None) -> Tensor:
    '''
    integrate lift and drag from 2D surface data
    
    :param geom: The geometry (x, y, z)
    :type geom: torch.Tensor (..., I, J, 3)
    :param aoa: angle of attack in Degree
    :type aoa: torch.Tensor (..., )
    :param cp: pressure coefficients Cp = (p - p_inf) / 0.5 * rho * U_\infty^2
    :type cp: torch.Tensor (..., I-1, J-1)
    :param cf: friction coefficients Cf = (tau @ n) / 0.5 * rho * U_\infty^2
    :type cf: torch.Tensor (..., I-1, J-1, 3)
        
    :return: coefficients of drag, lift, and side force
    :rtype: torch.Tensor (CD, CL, CZ)
    '''
    dfp = get_xyforce_2d_t(geom, cp, cf)
    dfp_xy = _xy_2_cl_tc(dfp[..., :2], aoa)
    dfp = torch.concatenate((dfp_xy, dfp[..., 2:]), axis=-1)
    return dfp

def get_moment_2d_t(geom: torch.Tensor, cp: torch.Tensor, cf: torch.Tensor=None, ref_point: torch.Tensor=np.array([0.25, 0, 0])) -> torch.Tensor:
    '''
    :param geom: The geometry (x, y, z)
    :type geom: torch.Tensor (..., I, J, 3)
    :param cp: pressure coefficients Cp = (p - p_inf) / 0.5 * rho * U_\infty^2
    :type cp: torch.Tensor (..., I-1, J-1)
    :param cf: friction coefficients Cf = (tau @ n) / 0.5 * rho * U_\infty^2
    :type cf: torch.Tensor (..., I-1, J-1, 3)
    :param ref_point: ref point for moment calculation
    :type ref_point: torch.Tensor (..., 3)
        
    :return: moment around z-axis
    :rtype: torch.Tensor (CMx, CMy, CMz)
    '''
    
    dxyforce = get_dxyforce_2d_t(geom, cp, cf)
    r = 0.25 * (geom[..., :-1, :-1, :] + geom[..., :-1, 1:, :] + geom[..., 1:, 1:, :] + geom[..., 1:, :-1, :]) - ref_point.to(geom.device)
    
    return torch.sum(torch.cross(r, dxyforce, dim=-1), dim=(-3, -2))

def get_cellinfo_1d_t(geom: torch.Tensor) -> Tuple[torch.Tensor]:
    '''
    :param geom: The geometry (x, y)
    :type geom: torch.Tensor (..., I, 2)

    :return: tangens, normals
    :rtype: Tuple[torch.Tensor] (..., I-1, 2), (..., I-1, 2)
    '''
    
    # grid centric
    tangens = geom[..., 1:, :] - geom[..., :-1, :]
    tangens = tangens / (torch.linalg.norm(tangens, dim=-1, keepdim=True) + 1e-20)
    normals = torch.concatenate((-tangens[..., [1]], tangens[..., [0]]), axis=-1)
    
    return tangens, normals

#* functions for wings

def rotate_input(inp: torch.Tensor, cnd: torch.Tensor, root_twist: float = 6.7166) -> Tuple[torch.Tensor]:
    '''
    rotate the input and condition to remove the baseline twist effect
    
    :param inp: geometric mesh input
    :type inp: torch.Tensor (B, C, H, W)
    :param cnd: operating condition (AoA, Mach)
    :type cnd: torch.Tensor (B, 2)
    :param root_twist: The root twist value to be removed
    :type root_twist: float
    :return: inp, cnd
    :rtype: Tuple[Tensor]
    '''

    B, C, H, W = inp.shape
    
    # rotate to without baseline twist ( w.r.t centerline LE (0,0,0))
    inp = torch.cat([
        _xy_2_cl_tc(inp[:, :2].permute(0, 2, 3, 1).reshape(-1, 2), -root_twist * torch.ones((B*H*W,))).reshape(B, H, W, 2).permute(0, 3, 1, 2),
        inp[:, 2:]
    ], dim = 1) 
    
    cnd = torch.cat([
        cnd[:, :1] + root_twist,
        cnd[:, 1:]
    ], dim = 1)

    return inp, cnd

def intergal_output(geom: torch.Tensor, outputs: torch.Tensor, aoa: torch.Tensor,
                    s: float, c: float, xref: float, yref: float) -> torch.Tensor:
    '''
    torch version intergal_output from cell-centric outputs to forces/moments
    
    :param geom: geometric
    :type geom: torch.Tensor (B, 3, I, J)
    :param outputs: pressure and friction coefficients (cp, cf_tau, cf_z)
    :type outputs: torch.Tensor (B, 3, I-1, J-1)
    :param aoa: angle of attacks
    :type aoa: torch.Tensor (B, )
    :param s: reference area
    :type s: float
    :param c: reference chord
    :type c: float
    :param xref: x reference point
    :type xref: float
    :param yref: y reference point
    :type yref: float
    :return: lift, drag, moment_z
    :rtype: torch.Tensor (B, 3)  
    '''

    cp = outputs[:, 0]
    tangens, normals2d = get_cellinfo_1d_t(geom[:, :2].permute(0, 2, 3, 1))                    
    tangens = 0.5 * (tangens[:, 1:] + tangens[:, :-1])    # transfer to cell centre at spanwise direction

    cf = torch.concatenate((outputs[:, [1]].permute(0, 2, 3, 1) * tangens / 150, outputs[:, [2]].permute(0, 2, 3, 1) / 300), axis=-1)
    forces = get_force_2d_t(geom.permute(0, 2, 3, 1), aoa=aoa, cp=cp, cf=cf)[:, [1, 0]] / s
    moment = get_moment_2d_t(geom.permute(0, 2, 3, 1), cp=cp, cf=cf, 
                             ref_point=torch.Tensor([xref, yref, 0.]))[:, [2]] / s / c

    return torch.cat((forces, moment), dim=-1)

def _get_xz_cf_t(geom: torch.Tensor, cf: torch.Tensor):
    '''
    params:
    ===
    `geom`:  The geometry (x, y), shape: (..., Z, I, 3)
    `cf`:  The geometry (cft, cfz), shape: (..., Z, I, 2)

    returns:
    ===
    `cfxyz`: shape: (..., I, J, 3)
    '''

    tangens, normals = get_cellinfo_1d_t(geom[..., [0,1]])
    tangens = 0.5 * (tangens[..., 1:, :, :] + tangens[..., :-1, :, :])    # transfer to cell centre at spanwise direction
    # normals = 0.5 * (normals[1:] + normals[:-1])
    # cfn = np.zeros_like(cf[..., 0])
    # print(cf[..., [0]].shape, tangens.shape)
    cfxyz = torch.concatenate((cf[..., [0]] * tangens, cf[..., [1]]), axis=-1)

    return cfxyz
