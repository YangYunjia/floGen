
'''
calling cfl3d for outside simulation

'''


import os
import numpy as np
import time
import pyDOE
import random
import shutil
from cst_modeling.section import cst_foil
from cfdpost.cfdresult import cfl3d
from cfdpost.section.physical import PhysicalSec
from cgrid.foil import CGrid

from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed

import time

# from flowvae.post import get_buffet

class AirfoilSimulator():
    
    def __init__(self, n_job, base_folder = '.', info = '') -> None:
        
        # create a new folder in simulation path
        current_time = time.time()
        str_time = time.strftime(f'Airfoil-%Y-%m-%d-%H-%M-%S', time.localtime(current_time))
        
        self.folder = os.path.join(base_folder, str_time)
        
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
    
    def submit(self, parameters):
        '''
        parameters: 
        - igroup, icond
        - ma, t, re, cl
        - cstu, cstl
        
        '''
        future = self.executor.submit(sim_airfoil_cfl3d, parameters)
        self.counter += 1
        self.futures.append(future)
        return self.counter
        
    def check_state(self, idx):
        '''
        RUNNING, PENDING, FINISHED, CANCELLED
        
        '''
        return self.futures[idx]._state
    
    def get_result(self, idx):
        if self.futures[idx]._state in ['FINISHED']:
            return self.futures[idx].result() 
        else:
            raise Exception()   
    


def log(string: str = '', path: str = '.', init=False):
    print(string)
    if init:
        with open(path+'/running.log', 'w') as f:
            f.write('\n')
    
    with open(path+'/running.log', 'a') as f:
        f.write('[%s] %s \n'%(time.ctime(), string))

def clean_folder(folder: str):
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

def is_calculated(folder: str):
    # for 6.7
    return os.path.exists(os.path.join(folder, 'cfl3d.prt')) and os.path.getsize(os.path.join(folder, 'cfl3d.prt')) > 7200000

def post_process(name: str, folder: str, j0: int, j1: int):
    
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

def sim_airfoil_cfl3d(folder, parameters):
    '''
    parameters: 
    - igroup, icond
    - ma, t, re, cl
    - cstu, cstl
    
    '''
    fname_solver = 'cfl3d_seq'

    # print(os.path.getsize(os.path.join(folder, 'cfl3d.prt')))

    if not is_calculated(folder):
        
        src_folder = os.path.join(os.path.dirname(__file__), 'cpsrc')
        shutil.copy(os.path.join(src_folder, 'cfl3d_Seq.exe'), folder)
        
        n_cst = len(parameters['cstu'])
        nn = 1001
        with open(os.path.join(folder, 'input.txt'), 'w') as f:
            f.write('igroup %d\n' % parameters['igroup'])
            f.write('icond  %d\n' % parameters['icond'])
            f.write('Ma     %18.9f\n' % parameters['ma'])
            f.write('t_     %18.9f\n' % parameters['t'])
            f.write('Re     %18.9f\n' % parameters['re'])
            f.write('cl     %18.9f\n' % parameters['cl'])
            for icst in range(n_cst):
                f.write('U%d     %18.9f\n' % (icst, parameters['cstu'][icst]))
            for icst in range(n_cst):
                f.write('L%d     %18.9f\n' % (icst, parameters['cstl'][icst]))

        calname = f'{parameters['igroup']:d} - {parameters['icond']:d}'
        #* ============================================
        #* input.txt  =>   cfl3d.xyz
        #* ============================================
        #* Read CST coefficients
        cst_u = parameters['cstu']
        cst_l = parameters['cstl']
        t_max = parameters['t']
        tail = 0.

        #* Build airfoil
        xx, yu, yl, _, R0 = cst_foil(nn, cst_u, cst_l, t=t_max, tail=tail)
        log(f'>>> [sample {calname}]  airfoil done (tmax {t_max:.4f} / R0 {R0:.4f} / ttail {tail:.4f})')

        #* Build grid
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
    return i, conv, step


def sim_airfoil_cfl3d():

    # os.chdir(os.path.dirname(__file__))
    nn     = 1001
    n_cst  = 10

    rf_folder = '.'
    n_job = 14
    n_airfoil = 995
    n_samples = 24
    n_all_samples = 64

    base_folder = os.path.join(rf_folder, 'Calculation')

    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # log('Start running calculation airfoils', init=True)

    log('reading parameters')
    
    parameters = np.loadtxt('foilparameters.dat')
    print(parameters.shape)
    
    log('start parallel calculating with job number = %d' % n_job)

    futures = []
    with ProcessPoolExecutor(max_workers=n_job) as executor:

        for i_airfoil in range(781, 995):

            for i_sample in range(n_samples):

                cal_base_dir = os.path.join(base_folder, str(i_airfoil), str(i_sample))
                ev_idx = i_airfoil * n_all_samples + i_sample

                if is_calculated(cal_base_dir):
                    # clean_folder(cal_base_dir)
                    continue

                os.makedirs(cal_base_dir)
                
                # func_mc(ev_idx, cal_base_dir)
                log('>>> [sample %4d - %4d] (id: %4d) in line' %  (i_airfoil, i_sample, ev_idx))
                futures.append(executor.submit(func_mc, ev_idx, cal_base_dir))


    
    