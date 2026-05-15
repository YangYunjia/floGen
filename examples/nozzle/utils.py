import math
from functools import reduce
from scipy.optimize import fsolve
import numpy as np

import copy

ga = 1.4
R = 287

coef8 = math.sqrt(ga / R * (2.0 / (ga + 1))**((ga + 1) / (ga - 1)))

def std_atomsphere(h, tref=288.15, rref=1.225, pref=101325, muref=1.72e-5):
    '''
    docin.com/p-70811798.html

    '''
    if h <= 11:
        w = 1 - h / 44.3308
        t = tref * w
        r = rref * w**4.2559
        p = pref * w**5.2559
    elif h <= 20:
        w = math.exp((14.9647 - h) / 6.3416)
        t = 216.65
        p = pref * 0.11953 * w
        r = rref *  1.5898 * w
    elif h <= 32:
        w = 1 + (h - 24.9021) / 221.552
        t = 221.55
        p = pref * 0.025158 * w**-34.1629
        r = rref * 0.032722 * w**-35.1629
    else:
        raise ValueError("h < 0 or h > 32")
    
    mu = muref * (t / 273.11)**1.5 * 383.67 / (t + 110.56)
    a = 20.0468 * t**0.5

    return {'temperature': t, 'pressure': p, 'density': r, 'viscosity': mu, 'soundspeed': a}

def ideal_mfr(pt8, tt8, A8, ma8=1.0):
    
    coef8 = math.sqrt(ga / R * (1 + 0.5 * (ga - 1) * ma8**2)**(-(ga + 1) / (ga - 1)))

    return coef8 * ma8 *  pt8 / tt8**0.5 * A8

def npr2ma(npr):
    return math.sqrt(2.0 / (ga-1) * (npr**((ga-1)/ga) - 1))

def tr2ma(tr):
    return math.sqrt(2.0 / (ga-1) * (tr - 1))

def ideal_thrust(pt7, tt7, p9, m8, fluid=None):
    npr = pt7 / p9
    if npr < 1:
        return 0.0, 0.0
    
    if fluid is None:
        v9id = math.sqrt(2 * ga * R / (ga-1) * tt7 * (1 - npr**(-(ga-1)/ga)))
    else:
        v9id = u9(fluid, tt7, t9(fluid, npr, tt7))
    
    thrustid = m8 * v9id
    return thrustid, v9id


class Fluid():
    
    def __init__(self, name, fix_thermo=False):
        self.name = name
        self.is_fix_thermo = fix_thermo
        if name == 'Air':
            self._gamma = 1.4
            self._cp_R = 3.5
            self._cp_R_para =  [[3.69736457,-1.5817846e-03, 3.89632940e-06,-2.5147111e-09, 4.89905696e-13],
                                [3.01061392, 1.4262348e-03,-5.41285431e-07, 9.8318138e-11,-6.77753328e-15]]
        else:
            raise KeyError()

    def cp(self, T):
        return self.cp_R(T) * R
    
    def cv(self, T):
        return (self.cp(T) - 1) * R

    def gamma(self, T):
        return self.cp(T) / self.cv(T)

    def cp_R(self, T=None):

        if T is None or self.is_fix_thermo:
            return self._cp_R
        else:
            return self.cal_cp_R(T)

    def cal_cp_R(self, T, intergal=0):
        
        para = copy.deepcopy(self._cp_R_para)

        multi = 1.0

        if intergal == 1:
            for i in range(1, len(para[0])):
                para[0][i] /= i
                para[1][i] /= i
            para[0][0] *= math.log(T)
            para[1][0] *= math.log(T)

        if intergal == 2:
            for i in range(0, len(para[0])):
                para[0][i] /= (i + intergal - 1)
                para[1][i] /= (i + intergal - 1)
            multi = T**(intergal - 1)
        # print(para[1][0]* math.log(T))
        
        if T < 100.0:
            return self.cal_cp_R(100.0, intergal=intergal)
        elif 100.0 <= T and T < 1000.0:
            return multi * (reduce(lambda x,i: (x + para[0][-i-1]) * T, range(len(para[0])-1),0) + para[0][0])
        elif 1000.0 <= T and T < 5000.0:
            return multi * (reduce(lambda x,i: (x + para[1][-i-1]) * T, range(len(para[1])-1),0) + para[1][0])
        elif 5000.0 <= T:
            return self.cal_cp_R(5000.0, intergal=intergal)

    def cal_cp_R_intergal(self, T, T0, intergal=1):

        if T < 1000.0 and 1000.0 <= T0:
            delta = - self.cal_cp_R(1000 - 1e-7,intergal=intergal) + self.cal_cp_R(1000,intergal=intergal)
        elif T0 < 1000.0 and 1000.0 <= T:
            delta = self.cal_cp_R(1000 - 1e-7,intergal=intergal) - self.cal_cp_R(1000,intergal=intergal)
        else:
            delta = 0.0
        
        return self.cal_cp_R(T,intergal=intergal) - self.cal_cp_R(T0,intergal=intergal) + delta


def t9(fluid, npr, tt7, fix_thermo=False):
    # approx with fix thermo
    if fluid.cp_R(500) == 0:
        print('!')
    _t9_fix = tt7 * npr**(-1 / 3.5)
    # print(npr, _t9_fix)
    if fix_thermo:return _t9_fix

    return fsolve(lambda x: math.log(npr) + fluid.cal_cp_R_intergal(x, tt7), _t9_fix)[0]

def u9(fluid, tt7, t9):
    if t9 >= tt7:
        return 0.0
    # return math.sqrt(2 * R * (fluid.cp_R(tt7)*tt7 - fluid.cp_R(t9)*t9))
    return math.sqrt(2 * R * fluid.cal_cp_R_intergal(tt7, t9, intergal=2))

def avg_2d_data(xx, yy, zz, var):
    length = ((xx[1:] - xx[:-1])**2 + (yy[1:] - yy[:-1])**2 + (zz[1:] - zz[:-1])**2)**0.5
    avgvar = (var[1:] + var[:-1]) * 0.5

    # for l, v in zip(length, avgvar):
    #     print('%.6f %.6f' %(l, v))
    # print(np.sum(length))
    # print(length**2 * avgvar)
    return np.sum(length * avgvar) / np.sum(length)


def mfr_2d(xx, vv, rho):
    # rot = np.repeat(np.array([[[0, 1], [-1, 0]]]), len(xx), axis=0)
    # print(rot.shape)
    nn = np.einsum('ij,jk->ik', np.array([[0, 1], [-1, 0]]), xx[:, 1:] - xx[:, :-1])
    avgvv = (vv[:, 1:] + vv[:, :-1]) * 0.5
    avgrho = (rho[1:] + rho[:-1]) * 0.5
    mfr = np.einsum('j,ij,ij->j', avgrho, avgvv, nn)

    return np.sum(mfr)

def mfravg_2d_data(xx, vv, rho, var):
    
    nn = np.einsum('ij,jk->ik', np.array([[0, 1], [-1, 0]]), xx[:, 1:] - xx[:, :-1])
    avgvv = (vv[:, 1:] + vv[:, :-1]) * 0.5
    avgrho = (rho[1:] + rho[:-1]) * 0.5
    mfr = np.einsum('j,ij,ij->j', avgrho, avgvv, nn)
    avgvar = (var[1:] + var[:-1]) * 0.5
    
    return np.sum(mfr * avgvar) / np.sum(mfr)