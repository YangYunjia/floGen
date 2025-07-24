
'''
calling cfl3d for outside simulation

'''


import os
import numpy as np
import time
# import pyDOE
# import random
import shutil
from cst_modeling.section import cst_foil, clustcos, dist_clustcos
from cfdpost.cfdresult import cfl3d
from cfdpost.section.physical import PhysicalSec
from scipy.interpolate import PchipInterpolator as pchip

from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed

import time

# from flowvae.post import get_buffet

class AirfoilSimulator():
   
    def __init__(self, n_job: int, base_folder: str = '.', case_folder: str = None, info: str = '') -> None:
        
        current_time = time.time()
        str_time = time.strftime(f'Airfoil-%Y-%m-%d-%H-%M-%S', time.localtime(current_time))
        
        if case_folder is None:
            # create a new folder in simulation path
            case_folder = str_time
        
        self.folder = os.path.join(base_folder, case_folder)
        
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        print(f'Calculation files in {self.folder}')
        
        with open(os.path.join(self.folder, 'info.txt'), 'w') as f:
            f.write(str_time + '\n')
            f.write(info)
            
        # start multi-processor
        self.executor =  ProcessPoolExecutor(max_workers=n_job)
        
        self.counter = 0
        self.futures = []
        
    def __del__(self):
        
        self.executor.shutdown()
    
    def submit(self, parameters: dict) -> int:
        '''
        parameters: 
        - igroup, icond
        - ma, t, re, cl
        - cstu, cstl
        
        '''        
        cal_folder = os.path.join(self.folder, str(self.counter))
        future = self.executor.submit(sim_airfoil_cfl3d, cal_folder, parameters)
        self.counter += 1
        self.futures.append(future)
        return self.counter - 1
    
    def submit_group(self, parameters: dict) -> list:
        '''
        parameters: 
        - igroup
        - ma, t, re, cl
        - cstu (N x NCST), cstl
        '''
        counters = []
        for i in range(len(parameters['ma'])):
            para0 = {'icond': i}
            for k in parameters.keys():
                if isinstance(parameters[k], list) or isinstance(parameters[k], np.ndarray):
                    para0[k] = parameters[k][i]
                else:
                    para0[k] = parameters[k]
        
            cal_folder = os.path.join(self.folder, str(self.counter))
            future = self.executor.submit(sim_airfoil_cfl3d, cal_folder, parameters)
            counters.append(self.counter)
            self.futures.append(future)
            self.counter += 1
            
        return counters
        
        
    def check_state(self, idx: int) -> str:
        '''
        RUNNING, PENDING, FINISHED, CANCELLED
        
        '''
        return self.futures[idx]._state
    
    def get_result(self, idx: int) -> dict:
        if self.futures[idx]._state in ['FINISHED']:
            return self.futures[idx].result() 
        else:
            raise Exception()
        
    def get_surface(self, idx: int) -> np.ndarray:
        '''
        return:
        ===
        
        `np.ndarray`:   N_var (3, y/Cp/Cf) x N_i
        
        '''
        if self.futures[idx]._state in ['FINISHED']:
            return get_surface(os.path.join(self.folder, str(idx)), 40, 361) 
        else:
            raise Exception()
    
    def wait_and_get_surface(self, idxs: list) -> np.ndarray:
        '''
        return:
        ===
        
        `np.ndarray`:   N_idxs x N_var (3, y/Cp/Cf) x N_i
        
        '''
        surfaces = []
        
        while len(idxs) > 0:
            
            for idx in idxs:
                if self.futures[idx]._state in ['FINISHED']:
                    surfaces.append(self.get_surface(idx))
                    idxs.remove(idx)
            time.sleep(1)
        
        return np.array(surfaces)


def log(string: str = '', path: str = '.', init: bool = False) -> None:
    print(string)
    if init:
        with open(path+'/running.log', 'w') as f:
            f.write('\n')
    
    with open(path+'/running.log', 'a') as f:
        f.write('[%s] %s \n'%(time.ctime(), string))

def clean_folder(folder: str) -> None:
    log('clearing the folder...')

    # os.system('del '+folder+'\\*.error > nul')
    os.system('del '+folder+'\\*.xyz > nul')
    os.system('del '+folder+'\\*.out > nul')
    os.system('del '+folder+'\\*.exe > nul')
    os.system('del '+folder+'\\*.p3d > nul')
    os.system('del '+folder+'\\aesurf.dat > nul')
    os.system('del '+folder+'\\blockforce.dat > nul')
    os.system('del '+folder+'\\cfl3d.2out > nul')
    os.system('del '+folder+'\\cfl3d.blomax > nul')
    os.system('del '+folder+'\\cfl3d.dynamic_patch > nul')
    os.system('del '+folder+'\\cfl3d.press > nul')
    os.system('del '+folder+'\\cfl3d.restart > nul')
    os.system('del '+folder+'\\cfl3d.subit_res > nul')
    os.system('del '+folder+'\\cfl3d.subit_turres > nul')
    os.system('del '+folder+'\\cfl3d.turres > nul')
    os.system('del '+folder+'\\genforce.dat > nul')
    os.system('del '+folder+'\\ovrlp.bin > nul')
    os.system('del '+folder+'\\patch.bin > nul')
    # os.system('del '+folder+'\\plot3d_grid.xyz > nul')
    # os.system('del '+folder+'\\plot3d_sol.bin > nul')

def is_calculated(folder: str) -> bool:
    # for 6.7
    return os.path.exists(os.path.join(folder, 'cfl3d.prt')) and os.path.getsize(os.path.join(folder, 'cfl3d.prt')) > 7200000

def post_process(name: str, folder: str, j0: int, j1: int) -> tuple:
    
    # succeed1, Minf, AoA, Re, _ = cfl3d.readinput(folder)
    converged, step, vals = cfl3d.readCoef1(folder, n=100)
    # succeed3, AoA = cfl3d.readAoA(folder, n=30)
    succeed, field, foil, fs = cfl3d.readprt_foil(folder, j0=j0, j1=j1, fname='cfl3d.prt', coordinate='xy')

    if succeed:
        #* Extract features
        (CL, CD, Cm, CDp, CDf) = vals
        (X,Y,U,V,P,T,Ma,Cp,vi) = field
        (x, y, Cp, _, Cf) = foil
        (Minf, AoA, _, Re, _, _) = fs
        Hi, Hc, info = PhysicalSec.getHi(X,Y,U,V,T,j0=j0,j1=j1,nHi=40)
        (Tw, dudy) = info
        
        fF = PhysicalSec(Minf, AoA, Re*1e6)
        fF.setdata(x, y, Cp, Tw, Hi, Hc, dudy)

        with open(folder + '\\Mw.dat', 'w') as f:
            nn = fF.x.shape[0]
            # f.write('zone T= "CL%3.3d" i= %d \n'%(int_CL, nn))
            f.write('VARIABLES = x y Mw Cp dudy Cp1 Cf1\n')
            f.write('zone T= "%s" i= %d \n'%(name, nn))
            for i in range(nn):
                f.write('% 20.10f  %20.10f  %20.10f  %20.10f  %20.10f  %20.10f  %20.10f\n'%(fF.x[i], fF.y[i], fF.Mw[i], fF.Cp[i], fF.dudy[i], Cp[i], Cf[i]))
            f.write('\n')

        with open(folder + '\\output.txt', 'w') as f:
            f.write('nConverged  %d \n'%(converged))
            f.write('nStep       %d \n'%(step))   
            f.write('AOA   %.9f\n' % AoA)
            f.write('CL    %.9f\n' % CL)
            f.write('CD    %.9f\n' % CD)
            f.write('CDp   %.9f\n' % CDp)
            f.write('Cm    %.9f\n' % Cm)
            f.write('K     %.9f\n' % (CL / CD))
        
    else:
        with open(folder + '\\output.txt', 'w') as f:
            f.write('nConverged  %d \n'%(converged))
            f.write('nStep       %d \n'%(step))  

    return converged, step, AoA, CL, CD

def get_surface(folder: str, j0: int, j1: int, nn: int = 161) -> np.ndarray:
    '''
    return:
    ===
    
    `np.ndarray`:   N_var (3, y/Cp/Cf) x N_i
    
    '''
    
    succeed, field, foil, fs = cfl3d.readprt_foil(folder, j0=j0, j1=j1, fname='cfl3d.prt', coordinate='xy')
    
    xx = dist_clustcos(nn)
    iLE = np.argmin(foil[0])
    
    
    yss = []

    for iv in [1, 2, 4]:    # y, cp, cf
        # lower surface
        fy = pchip(foil[0][:iLE+1][::-1], foil[iv][:iLE+1][::-1])
        y_l = fy(xx)

        # upper surface
        fy = pchip(foil[0][iLE:], foil[iv][iLE:])
        y_u = fy(xx)

        ys = np.concatenate((y_l[::-1], y_u[1:]), axis=0)
        yss.append(ys)
        
    return preprocess(xx, np.array(yss))

def preprocess(xx: np.ndarray, datas: np.ndarray) -> np.ndarray:
    
    nn = len(xx)
    # cpt = 1. + datas[i_f, 1] * (0.5 * indexs[i_f, 2]**2 * 1.4)
    cpt = datas[1]
    cft = datas[2] * 200
    
    hn = 6
    for i_p in range(0, hn):
        cft[i_p] = cft[hn] - (cft[hn+1] - cft[hn]) / (xx[hn+1] - xx[hn]) * (xx[hn] - xx[i_p])
        
    # hn = 40
    # # for i_p in range(0, hn):
    # #     cpt[-i_p-1] = cpt[-hn-1] + (cpt[-i_p-1] - cpt[-hn-1]) * (0.6 + i_p / hn * 0.4)
    # for i_p in range(hn, 0, -1):
    #     localgra = (cpt[-i_p-1] - cpt[-i_p]) / (xx[-i_p-1] - xx[-i_p])
    #     # print(i_p, localgra)
    #     if abs(localgra) > 2:
    #         for i_pp in range(i_p, 0, -1):
    #             cpt[-i_pp] = cpt[-i_p-1] - localgra * (xx[-i_p-1] - xx[-i_pp])
    #         break
    # if i_p > 1:
    #     i_fs.append(i_f)
    # print(cft[158:162])
    cft[:nn-1] = - cft[:nn-1]
    
    # plt.plot(all_xn, ndata[0])
    # plt.plot(all_xn, ndata[1])
    # plt.show()
    datas[1] = cpt
    datas[2] = cft
    
    return datas

def sim_airfoil_cfl3d(folder, parameters: dict) -> dict:
    '''
    parameters: 
    - igroup, icond
    - ma, re, cl
    - t, cstu, cstl (or `xx`, `yu`, `yl`)
    
    '''
    fname_solver = 'cfl3d_seq'
    
    # print(parameters)

    # print(os.path.getsize(os.path.join(folder, 'cfl3d.prt')))

    if not is_calculated(folder):
        
        calname = '%d - %d' % (parameters['igroup'], parameters['icond'])
        
        os.mkdir(folder)
        
        src_folder = os.path.join(os.path.dirname(__file__), 'cpsrc')
        shutil.copy(os.path.join(src_folder, 'cfl3d_Seq.exe'), folder)
        
        if 'cstu' in parameters.keys():
            n_cst = len(parameters['cstu'])
            nn = 1001
        else:
            n_cst = 0   # this ceases the output of cst numbers in input.txt
        
        with open(os.path.join(folder, 'input.txt'), 'w') as f:
            f.write('igroup %d\n' % parameters['igroup'])
            f.write('icond  %d\n' % parameters['icond'])
            f.write('Ma     %18.9f\n' % parameters['ma'])
            f.write('Re     %18.9f\n' % parameters['re'])
            f.write('cl     %18.9f\n' % parameters['cl'])
            
            if 't' in parameters.keys():
                f.write('t_     %18.9f\n' % parameters['t'])
            for icst in range(n_cst):
                f.write('U%d     %18.9f\n' % (icst, parameters['cstu'][icst]))
            for icst in range(n_cst):
                f.write('L%d     %18.9f\n' % (icst, parameters['cstl'][icst]))
        
        if n_cst == 0:  
            # write airfoil geometry to foil.dat
            with open(os.path.join(folder, 'foil.dat'), 'w') as f:
                xx = parameters['xx']
                yu = parameters['yu']
                yl = parameters['yl']
                f.write('VARIABLES = x yu yl\n')
                f.write('zone T= "%s" i= %d \n'%('foil-%s' % calname, len(parameters['xx'])))
                for ixx, iyu, iyl in zip(xx, yu, yl):
                    f.write('% 20.10f  %20.10f  %20.10f\n' % (ixx, iyu, iyl))
        
        else:
            #* Reconstruct airfoil geometry from cst coefficients
            #  if tmax is prescribed, then extend the airfoil to given tmax
            #* Read CST coefficients
            cst_u = parameters['cstu']
            cst_l = parameters['cstl']
            if 't' in parameters.keys():
                t_max = parameters['t']
            else:
                t_max = None
            tail = 0.

            #* Build airfoil
            xx, yu, yl, t_max, R0 = cst_foil(nn, cst_u, cst_l, t=t_max, tail=tail)
            log(f'>>> [sample {calname}]  airfoil done (tmax {t_max:.4f} / R0 {R0:.4f} / ttail {tail:.4f})')

        #* Build grid
        from cgrid.foil import CGrid
        cg = CGrid(G_Foil=161, G_Wake=41, G_Grow=81, yp_cri=0.9, rout=20.0, Uinf=300.0, rou=1.1, Re=2E7)
        # cg = FoilGrid(G_Foil=161, G_Wake=41, G_Grow=81, G_Tail=17,  yp_cri=0.9, rout=20.0, Uinf=300.0, rou=1.1, Re=2E7)

        cg.gen_grid(xx, yu, xx, yl, output=False)
        cg.output_grid(name=os.path.join(folder, 'cfl3d'), output_inp=False)
        log(f'>>> [sample {calname}]  mesh done')
        # input()
    
        with open(os.path.join(src_folder, 'cfl3d0.inp'), 'r') as f:
            cfl3d_inp_lines = f.readlines()

        with open(os.path.join(folder, 'cfl3d.inp'), 'w') as f:
            for i in range(15):
                f.write(cfl3d_inp_lines[i])
            f.write('cltarg %.4f\n' % (parameters['cl']))
            for i in range(16,19):
                f.write(cfl3d_inp_lines[i])                
            f.write('  %.4f  %.4f  0.00  %.4f  %.1f  1  0 \n' % 
                    (parameters['ma'], 0.0, parameters['re'], 580))
            for i in range(len(cfl3d_inp_lines)-20):
                f.write(cfl3d_inp_lines[i+20])

        
        log(f'>>> [sample {calname}]  start running... ')
        # os.system('chmod 777 %s' % fname_solver)
        os.system('cd %s && start /wait /min cfl3d_Seq.exe' % folder)
        # os.system('mpiexec --allow-run-as-root -np %d %s' % (core_number, fname_solver))
        # os.system('cd %s && ./%s' % (folder, fname_solver))
        clean_folder(folder)

    conv, step, AoA, Cl, Cd = post_process(str(i), folder, 40, 361)
    log(f'>>> [sample {calname}]  Calculation done (Code {conv:d} / Step {step:d} / AoA {AoA:.4f} / CL {Cl:.4f} / CD {Cd:.4f})')
    return {'iconv':    conv, 
            'nstep':    step, 
            'aoa':      AoA,
            'cl':       Cl,
            'cd':       Cd,
            'folder':   folder}

