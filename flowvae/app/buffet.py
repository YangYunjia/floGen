from sklearn.linear_model import LinearRegression
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import interp1d
from scipy.stats import theilslopes


import numpy as np
import math
import copy

import matplotlib.pyplot as plt

from typing import Callable, Dict, List, Tuple

class IntersectNotFound(Exception):
    pass

def sort_by_aoa(_aoas, _clss, _cdss):
    #* sort aoa datas
    aoas = []
    clss = []
    cdss = []
    for aoa, cl, cd in zip(_aoas, _clss, _cdss):
        if len(aoas) == 0 or aoa < aoas[0]:
            aoas.insert(0, aoa)
            clss.insert(0, cl)
            cdss.insert(0, cd)
        else:
            for idx in range(len(aoas) - 1):
                if aoa > aoas[idx] and aoa < aoas[idx + 1]:
                    aoas.insert(idx + 1, aoa)
                    clss.insert(idx + 1, cl)
                    cdss.insert(idx + 1, cd)
                    break
            else:
                aoas.append(aoa)
                clss.append(cl)
                cdss.append(cd)

    # print(_aoas, _clss, _cdss)
    # print(aoas, clss, cdss)
    # raise
    return aoas, clss, cdss

def first_argmin(l):
    for i in range(1, len(l)-1):
        if l[i] <= l[i-1] and l[i] <= l[i+1]:
            return i
    return -1

class Ploter():

    def __init__(self, name: str = None, symbol: str = 'o', color: str = 'r', ref_aoa: float = 0.0, instant_plot: bool = False) -> None:
        self.name = name
        self.symbol = symbol
        self.color = color
        self.ref_aoa = ref_aoa
        self.instant_plot = instant_plot


def disturb(base_series, n_samples, sigma, variables=['Cl']):

    random_number = np.random.normal(loc=0., scale=sigma, size=(n_samples, len(variables), len(base_series)))
    disturbed_serires = []
    for i in range(n_samples):
        new_series = Series(copy_from=base_series)
        for ik, k in enumerate(variables):
            new_series.series[k] = new_series.series[k] * (1 + random_number[i, ik])
        disturbed_serires.append(new_series)
    return disturbed_serires
    

class Series():

    def __init__(self, varlist: List = None, datas: Dict = None, copy_from=None) -> None:
        
        self.series = {}
        self._length = 0
        if varlist is not None:
            for var in varlist:
                self.series[var] = []
        elif datas is not None:
            for var in datas.keys():
                self.series[var] = datas[var]
                if self._length > 0 and self._length != len(self.series[var]):
                    raise RuntimeError('length not match')
                else:
                    self._length = len(self.series[var])
        elif copy_from is not None:
            self.copy_from(copy_from)
        else:
            raise ReferenceError('At least varlist and datas should not be none')
    
    def copy_from(self, x):
        self._length = x._length
        for k in x.series.keys():
            self.series[k] = copy.deepcopy(x.series[k])
    
    def __len__(self):
        return self._length

    def __getattr__(self, __name: str) -> List:
        if __name not in self.series.keys():
            raise KeyError('%s is not in the series' % __name)
        return self.series[__name]

    def add(self, x: Dict):

        if self.series.keys() != x.keys():
            raise Exception(self.series.keys(), x.keys())
        
        if x['AoA'] in self.series['AoA']:
            return

        for k in x.keys():
            self.series[k].append(x[k])

        self.sort()
        self._length += 1

    def sort(self):
        
        ii_sort = np.argsort(self.series['AoA']).tolist()
        for k in self.series.keys():
            self.series[k]= [self.series[k][i] for i in ii_sort]

    def print_tecplot(self, name='series.dat'):

        lines = []
        line = 'Variables='
        for key in self.series.keys():
            line += ' %s' % key
        line += '\n'
        lines.append(line)
        lines.append('zone i= %d\n' % len(self))

        for i in range(len(self)):
            line = ''
            for key in self.series.keys():
                line += ' %20.10f' % self.series[key][i]
            line += '\n'
            lines.append(line)

        with open(name, 'w') as f:
            f.writelines(lines)

    def down_sampling(self, x: np.array, method: str):
        for var in self.series.keys(): 
            if var == 'AoA': continue
            # print(var, len(self.series['AoA']), len(self.series[var])) 
            if method == 'pchip':
                f = PchipInterpolator(self.series['AoA'], self.series[var])
            elif method == '1d':
                f = interp1d(self.series['AoA'], self.series[var], bounds_error=False, fill_value='extrapolate')
            self.series[var] = f(x)
        self.series['AoA'] = copy.deepcopy(x)
        self._length = len(x)

    def delete_duplicate(self, therhold):
        new_idxs = [0]
        for idx in range(1, len(self.AoA)):
            if (self.AoA[idx] - self.AoA[new_idxs[-1]]) >= therhold:
                new_idxs.append(idx)
        for var in self.series.keys(): 
            self.series[var] = np.take(self.series[var], new_idxs)

class Buffet():

    def __init__(self, method='lift_curve_break',logtype=0, **kwargs) -> None:

        '''
        buffet judger from aerodynamic curves

        ## buffer onset criteria

        ### adaptive lift curve break 
        This method is put forward by Renze Li and improved by Yunjia Yang.

        #### parameters for decide linear section of lift curve
        - `AoA_linupp`..... 1.5 ..... upper bound for AoA when searching increasingly
        - `   

        
        
        '''

        self.logtype = logtype
        self.method = method
        self.paras = {}
        self.aoadelta = 0.001
        self.process_values = []

        for key in kwargs.keys():
            self.paras[key] = kwargs[key]

        #* deal with parameters
        if 'sep_check'   not in kwargs.keys():       self.paras['sep_check'] = False
        if 'AoA_linupp'   not in kwargs.keys():       self.paras['AoA_linupp'] = 1.5
        if 'AoA_min'      not in kwargs.keys():       self.paras['AoA_min']    = -3.0
        if 'AoA_max'      not in kwargs.keys():       self.paras['AoA_max']    = 6.0
        if 'dAoAs'        not in kwargs.keys():       self.paras['dAoAs']      = [0.5, 0.2, 0.02] 
        if 'AoA_1sep'     not in kwargs.keys():       self.paras['AoA_1sep']   = 3.0 
        if 'intp'   not in kwargs.keys():       self.paras['intp'] = 'pchip'
        if 'lsumtd' not in kwargs.keys():       self.paras['lsumtd'] = 'error'  # method to decider linear section upper bound
                                                                                # `error` for 
        if 'lslmtd' not in kwargs.keys():       self.paras['lslmtd'] = 'shock'  # method to decider linear section upper bound
        if 'lsthod' not in kwargs.keys():
            if self.paras['lsumtd'] in ['error']:       self.paras['lsthod']   = 0.02
            elif self.paras['lsumtd'] in ['cruise', 'maxLD']:   self.paras['lsthod'] = 0.9999
        if 'lstsa'  not in kwargs.keys():       self.paras['lstsa']  = 0.95
        if 'lsuth1' not in kwargs.keys():       self.paras['lsuth1'] = 0.0      # linear segment upper bound therhold upper
                                                                    
        if 'lsuth2' not in kwargs.keys():       self.paras['lsuth2'] = 1.0         # old 1.2
                                                                                # linear segment upper bound therhold lower
                                                                                # the u.b. is searched down from start point not less than this value
                                                                                # to let r2 > 0.9999 (for LRZ is 1.0, for SFY is 0.01)
        if 'lslt' not in kwargs.keys():         self.paras['lslt'] = 3.0        # linear segment length (for LRZ is 3.0, for SFY is 1.0)
        if 'daoa' not in kwargs.keys():         self.paras['daoa'] = 0.1
        if 'srst' not in kwargs.keys():         self.paras['srst'] = 'sep'
        # if 'linear' not in kwargs.keys():       self.paras['linear'] = 'cruise'
        
        if 'diff' not in kwargs.keys():         self.paras['diff'] = 0.1
        if 'dfcri' not in kwargs.keys():         self.paras['dfcri'] = 'Cl'
        if 'crodr' not in kwargs.keys():        self.paras['crodr'] = 2
        if 'crmod' not in kwargs.keys():        self.paras['crmod'] = 'max'

    def log(self, s: str):

        if self.logtype > 0:
            print(s)

    def buffet_onset(self, seri, **kwargs):

        if self.method in ['lift_curve_break']: results = self.lift_curve_break(seri, **kwargs)
        elif self.method in ['adaptive_lift_curve_break']: results = self.adaptive_lift_curve_break(seri, **kwargs)
        elif self.method in ['curve_slope_diff']: results = self.curve_slope_diff(seri, **kwargs)
        elif self.method in ['cruve_critical_value']: results = self.cruve_critical_value(seri, **kwargs)
        elif self.method in ['shock_to_maxcuv']: results = self.shock_to_maxcuv(seri, **kwargs)

        if self.paras['sep_check']:
            if self.paras['intp'] == 'pchip':
                f_cl_aoa_all = PchipInterpolator(seri.AoA, np.array(seri.Cl))
            elif self.paras['intp'] == '1d':
                f_cl_aoa_all = interp1d(seri.AoA, np.array(seri.Cl), bounds_error=False, fill_value='extrapolate')
            
            AoA_sep = self.incipient_separation(seri)
            for i in range(3):
                if results[i, 0] < AoA_sep:
                    print('AoA buffet onset change from %.4f to %.4f' % (results[i, 0], AoA_sep))
                    results[i, 0] = AoA_sep
                    results[i, 1] = f_cl_aoa_all(AoA_sep)
                    
        return results

    @staticmethod
    def __zero(x):
        return 0.0

    @staticmethod
    def __delete_zero(xs, ys, both=False, track_index=False):
        '''
        Delete zeros of y (and x)
        '''
        x = np.array(xs).copy()
        y = np.array(ys).copy()
        ii = [i for i in range(len(xs))]

        n0 = len(xs)
        i0 = 0  # index in xs, ys
        k  = 0  # index in x, y
        while i0<n0:
            if abs(y[k])<1e-6:
                x = np.delete(x, k, axis=0)
                y = np.delete(y, k, axis=0)
                ii.pop(k)
            else:
                k += 1
            i0 += 1

        if both:
            n0 = x.shape[0]
            i0 = 0
            k  = 0
            while i0<n0:
                if abs(x[k])<1e-6:
                    x = np.delete(x, k, axis=0)
                    y = np.delete(y, k, axis=0)
                    ii.pop(k)
                else:
                    k += 1
                i0 += 1

        if track_index:
            return x, y, ii
        else:
            return x, y

    @staticmethod
    def interpolate(xs, ys, kind=0, delete_zero=True) -> Tuple[Callable, float, float]:
        '''
        >>> func, x_low, x_upp = interpolate(xs, ys, kind=0)

        ### Inputs:
        ```text
        xs: ndarray [n]
        ys: ndarray [n]
        ```
        '''
        if delete_zero:
            x, y = Buffet.__delete_zero(xs, ys, both=False)
        else:
            x = xs
            y = ys
        
        if x.shape[0] <= 1:
            print(x)
            return None, 0.0, 0.0
        elif x.shape[0] <= 3:
            return interp1d(x, y, kind='linear', fill_value='extrapolate'), x[0], x[-1]
        else:
            if kind in [0, '1d']:
                # piecewise cubic spline
                return interp1d(x, y, kind='cubic', fill_value='extrapolate'), x[0], x[-1]
            elif kind in [1, 'pchip']:
                # piecewise monotonic cubic Hermite interpolation (PCHIP)
                return PchipInterpolator(x, y, extrapolate=True), x[0], x[-1]
            else:
                raise KeyError('interpolate method (`intp`) should be 1d or pchip')
    
    @staticmethod
    def interpolate_variable(AoAs, Vs, a: float, kind=1):
        '''
        Interpolate the value of V at a
        '''
        fv, A_low, A_upp = Buffet.interpolate(np.array(AoAs), np.array(Vs), kind=kind)

        va = 0.0
        if a<A_low or a>A_upp:
            return va

        return fv(a)

    @staticmethod
    def intersection(x, y1, y2, mode=0):
        '''
        mode > 0, d[i+1] > d[i]
        '''
        delta = np.array(y2) - np.array(y1)
        for i in range(len(x) - 1):
            if delta[i] * delta[i+1] < 0. and (delta[i+1] - delta[i]) * mode >= 0:
                xs = x[i] + (x[i+1] - x[i]) * delta[i] / (delta[i] - delta[i+1])
                ys = y1[i] + (y1[i+1] - y1[i]) / (x[i+1] - x[i]) * (xs - x[i])
                break
        else:
            raise IntersectNotFound('intersection not found')
        return xs, ys

    @staticmethod
    def localmax(x1, y1, x2, y2, x3, y3):
        '''
        find the maximum location of the quadratic function that pass the three points (x1, y1), (x2, y2), and (x3, y3)
        '''
        # l = (x1 - x2) * (x2 - x3) * (x1 - x3)
        # a = (y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2)) / l
        # b = -(y1 * (x2**2 - x3**2) + y2 * (x3**2 - x1**2) + y3 * (x1**2 - x2**2)) / l
        # c = (x2 * x3 * y1 * (x2 - x3) + x1 * x3 * y2 * (x3 - x1) + x1 * x2 * y3 * (x1 - x2)) / l
        return 0.5 * (y1 * (x2**2 - x3**2) + y2 * (x3**2 - x1**2) + y3 * (x1**2 - x2**2)) / (y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2))

    @staticmethod
    def curvature(x1, y1, x2, y2, x3, y3):
        '''
        calculat the local curvature at (x2, y2) with three points (x1, y1), (x2, y2), and (x3, y3)
        '''
        localdCLdA = (y3 - y1) / (x3 - x1)
        locald2CLdA2 = ((y3 - y2) / (x3 - x2) - (y2 - y1) / (x2 - x1)) / (0.5 * (x3 - x1))
        return - locald2CLdA2 / (1 + localdCLdA**2)**1.5

    def _decide_outer_point(self, _AoA, new_AoA, AoAs, dAoA, lAoA):

        _AoA = float('%.3f'%(int(_AoA / dAoA) * dAoA))
        lAoA = float('%.3f'%(int(lAoA / dAoA) * dAoA))

        if new_AoA <= self.paras['AoA_min'] or new_AoA >= AoAs[-2]: # Not enough points on the right
            
            _AoA += 2 * dAoA
            
            if _AoA in AoAs:
                _AoA = AoAs[-1] + 2 * dAoA
        
        elif new_AoA < lAoA:  # Smaller than first try
            
            _AoA = new_AoA

            if _AoA in AoAs:
                _AoA = lAoA - dAoA
        
        else:

            pass # no need for additional points
        
        _AoA = float('%.3f'%(int(_AoA / dAoA) * dAoA))
        
        return _AoA

    def _predict_buffet(self, AoAs, CLs, Cfs):

        linear_portion, bfpoints, AoA_sep = self._adjust_linear_portion(AoAs, CLs, Cfs)

        ii_et = np.argmin(np.abs(np.array(AoAs) - bfpoints[1, 0]))
        ii_e1 = min(ii_et + 1, len(AoAs) - 1)
        
        # Set to AoA_sep <= AoA_lb <= AoA_et <= AoA_ub
        bfpoints[0, 0] = max(bfpoints[0, 0], AoA_sep)
        bfpoints[1, 0] = max(bfpoints[1, 0], bfpoints[0, 0])
        bfpoints[2, 0] = max(bfpoints[2, 0], bfpoints[1, 0])

        # log('')
        # info = '  Coarse: incipient separation AoA = %.3f; buffet AoA = %.3f (dAoA=%.3f)'%(AoA_sep, AoA_et, dAoA[1]); log(info)
        # info = '          Buffet AoA range = [%.3f, %.3f]'%(AoA_lb, AoA_ub); log(info)
        # info = '          CL error of CFD =         %.1E'%(Errs[ii_et][0]); log(info)
        # info = '          CL error due to slope =   %.1E'%(fCl(AoA_ub)-fCl(AoA_lb)); log(info)
        # info = '          CL delta between points = %.1E'%(0.5*(CLs[ii_e1]-CLs[ii_et-1])); log(info)
        # info = '          CFD calculations = %d'%(len(AoAs)); log(info)
        # info = '          Time = %.1f min'%((t2-t1)/60.0); log(info)

        return linear_portion, bfpoints, AoA_sep

    def _estimate_linear_lift_curve(self, AoAs: np.array, CLs: np.array, CDs: np.array = None, X1s: np.array = None, cl_c: float = 0.8, add_right: int = 0) -> dict:
        '''
        Estimate linear portion of the lift curve
        
        >>> linear_portion = estimate_linear_lift_curve(AoAs, CLs)
        
        linear_portion['slope']: slope
        linear_portion['intercept']: intercept
        linear_portion['slope_lb']: slope lower bound
        linear_portion['slope_ub']: slope upper bound
        linear_portion['AoA_ub']: AoA upper bound of linear region
        '''
        ds0 = 0
        x, y = Buffet.__delete_zero(np.array(AoAs), np.array(CLs), both=False)

        #* decide upper bound for linear section

        if isinstance(self.paras['lslmtd'], int):
            i_lb = self.paras['lslmtd']
        else:
            if self.paras['lslmtd'] in ['shock']:
                i_lb = 0
                while not (X1s[i_lb] > 0. and X1s[i_lb + 1] > 0.) and AoAs[i_lb] <= 0.0:
                    i_lb += 1
                self.log('The single shock starts at i = %d' % i_lb)
            else:
                raise KeyError()

        if self.paras['lsumtd'] in ['error']:
        
            for i in range(i_lb, x.shape[0]-2):
                
                if x[i+2] > self.paras['AoA_linupp']:
                    i_ub = i+1
                    break
            
                slope, intercept, low_slope, high_slope = theilslopes(y[i_lb: i+3], x[i_lb: i+3], alpha=self.paras['lstsa'])
                ds = high_slope-low_slope
                
                if i == i_lb:
                    ds0 = ds
                    i_ub = i+2
                    # print(i+2, x[i+2], ds, ds0)

                elif ds > self.paras['lsthod'] or ds > ds0 * 3: # primary ds > 0.02, ds > ds0 * 5
                    i_ub = i+1
                    # print(i+2, x[i+2], ds, ds0, 'break')
                    break

                else:
                    # print(i+2, x[i+2], ds, ds0)
                    pass

            else:
                i_ub = i_lb + 3
                print('!!not found')

            i_ub = min(i_ub+add_right, x.shape[0]-1)
            AoA_ub = x[i_ub]
        else:
            if self.paras['intp'] == 'pchip':
                f_cl_aoa_all = PchipInterpolator(AoAs, np.array(CLs))
            elif self.paras['intp'] == '1d':
                f_cl_aoa_all = interp1d(AoAs, np.array(CLs), bounds_error=False, fill_value='extrapolate')

            aoa_refs = list(np.arange(-2, 4, 0.01))
            if self.paras['lsumtd'] == 'cruise':
                max_i = first_argmin(abs(f_cl_aoa_all(aoa_refs) - cl_c))
            elif self.paras['lsumtd'] == 'maxLD':
                f_cdcl_aoa_all = PchipInterpolator(AoAs, np.array(CLs) / np.array(CDs))
                max_i = first_argmin(-f_cdcl_aoa_all(aoa_refs))
            

            # #* this part of code is to find the better `lsthod`,
            # #* it plot the regression score of different length of linear section
            # min_aoa = AoAs[i_lb]
            # max_aoas = np.arange(min_aoa+0.3, min_aoa+4, 0.1)
            # reg_scores = []
            # for i_max_aoa in max_aoas:
            #     linear_aoas = np.arange(min_aoa, i_max_aoa, 0.1).reshape(-1,1)    # modified 2023.5.14, change the upper bound to max_aoa - 0.5
            #     f_cl_aoa_linear = f_cl_aoa_all(linear_aoas)
            #     reg = LinearRegression().fit(linear_aoas, f_cl_aoa_linear)
            #     reg_scores.append(reg.score(linear_aoas, f_cl_aoa_linear))

            # plt.plot(max_aoas, reg_scores, 'gray', lw=0.1)
            
            max_aoa = aoa_refs[max_i] + self.paras['lsuth1'] + 0.1
            min_aoa = AoAs[i_lb]

            while max_aoa > aoa_refs[max_i] + self.paras['lsuth1'] - self.paras['lsuth2'] and max_aoa > AoAs[i_lb + 2]:
                max_aoa -= 0.1
                linear_aoas = np.arange(min_aoa, max_aoa, 0.1).reshape(-1,1)    # modified 2023.5.14, change the upper bound to max_aoa - 0.5
                f_cl_aoa_linear = f_cl_aoa_all(linear_aoas)
                reg = LinearRegression().fit(linear_aoas, f_cl_aoa_linear)
                if reg.score(linear_aoas, f_cl_aoa_linear) > self.paras['lsthod']:
                    break

            while i_lb > 0 and max_aoa - AoAs[i_lb] < 1.0:
                # print(min_aoa, max_aoa, i_lb)
                i_lb = max(0, i_lb - 1)

            # plt.plot([max_aoa], [reg.score(linear_aoas, f_cl_aoa_linear)], '+', c='k')
            
            i_ub = max(i_lb + 2, np.argmin(np.abs(AoAs - max_aoa)) + 1)
            AoA_ub = max_aoa
        # print(x, y, i_lb, i_ub)
        slope, intercept, low_slope, high_slope = theilslopes(y[i_lb:i_ub+1], x[i_lb:i_ub+1], alpha=self.paras['lstsa'])
            
        linear_portion = {
            'slope':        slope,
            'intercept':    intercept,
            'slope_lb':     low_slope,
            'slope_ub':     high_slope,
            'AoA_ub':       AoA_ub,
            'i_lb':         i_lb,
            'i_ub':         i_ub
        }

        self.log(' > Predict linear section slope: %.4f [%.4f, %.4f]; intercept: %.4f' % (slope, low_slope, high_slope, intercept))
        self.log(' > Linear section range: [%d, %d]' % (i_lb, i_ub))

        return linear_portion

    def _estimate_buffet_onset(self, AoAs, CLs, linear_portion: dict, kind: int = 1):
        '''
        Buffet onset estimation with "lift curve break method"
        
        delta alpha = 0.1 deg
        
        Notation
            lb: lower bound of slope
            et: slope by Theil-Sen estimator
            ub: upper bound of slope
        
        >>> AoA_et, AoA_lb, AoA_ub, fCl = estimate_buffet_onset(AoAs, CLs, linear_portion, dAoA, kind=1)
        
        '''
        
        fCl, A_low, A_upp = Buffet.interpolate(np.array(AoAs), np.array(CLs), kind=kind)
        
        aa = np.arange(A_low, A_upp + 1e-6, 0.001)
        cl = fCl(aa)

        bfpoints = np.ones((3, 2)) * -10.0
        
        #* Auxiliary line that offsets the linear portion of the lift curve
        for idx, key in enumerate(['slope_lb', 'slope', 'slope_ub']):
            try:
                bfpoints[2-idx, 0], bfpoints[2-idx, 1] = Buffet.intersection(aa, 
                                                        linear_portion['intercept'] + linear_portion[key] * (aa - self.paras['daoa']), cl)
            except IntersectNotFound:
                print('buffet not found for %s' % key)
        # AoA_lb = float('%.3f'%(int(AoA_lb/dAoA)*dAoA))
        # AoA_et = float('%.3f'%(int(AoA_et/dAoA)*dAoA))
        # AoA_ub = float('%.3f'%(int(AoA_ub/dAoA)*dAoA))        
        # AoA_lb = min(AoA_lb, AoA_et)
        # AoA_ub = max(AoA_ub, AoA_et)

        self.log(' > Predict buffet onset: AoA = %.4f [%.4f, %.4f]; CL = %.4f' % (bfpoints[1,0], bfpoints[0,0], bfpoints[2,0], bfpoints[1,1]))

        return bfpoints, fCl

    def _estimate_incipient_separation(self, AoAs, mUys, kind=1):
        '''
        Estimation of incipient separation
        
        >>> AoA_uy = estimate_incipient_separation(AoAs, mUys, dAoA, kind=1)
        
        '''
        # print(AoAs, mUys)
        fUy, A_low, A_upp = Buffet.interpolate(np.array(AoAs), np.array(mUys), kind=kind)
        
        AoA_uy = AoAs[0]

        if fUy is not None:
        
            if fUy(A_upp)>=0:
                # no separation at the largest input angle of attacks
                for i in range(100):
                    if fUy(A_upp + i * 0.001)<0:
                        AoA_uy = A_upp + i * 0.001
                        break
            
            else:
            
                aa = np.arange(A_low, A_upp+0.001, 0.001)
                uy = fUy(aa)
                try:
                    AoA_uy, _ = Buffet.intersection(aa, np.zeros_like(aa), uy, mode=-1)
                except IntersectNotFound:
                    AoA_uy = A_upp

        # AoA_uy = float('%.3f'%(int(AoA_uy/0.02)*0.02))
            
        self.log(' > Predict separation: AoA = %.4f ' % (AoA_uy))

        return AoA_uy

    def _adjust_linear_portion(self, AoAs, CLs, mUys):
        '''
        Buffet only happens after incipient separation
        '''
        add_right = 0
        
        while True:
            
            self.log(' >> Adjust linear portion: add_right = %d' % (add_right))
        
            linear_portion = self._estimate_linear_lift_curve(AoAs, CLs, add_right=add_right)

            AoA_sep = self._estimate_incipient_separation(AoAs, mUys, kind=1)
            
            bfpoints, _ = self._estimate_buffet_onset(AoAs, CLs, linear_portion, kind=0)

            add_right += 1
            
            if AoA_sep <= bfpoints[1, 0] or add_right>=10:
                break
        
        return linear_portion, bfpoints, AoA_sep

    def adaptive_lift_curve_break(self, cfdfunc: Callable, cfdfuncparas: List = None,
                    plot_name: str = None):
        '''
        adaptive obtain linear section and buffet onset, adapted from Runze Li.
        
        '''

        seri = Series(['Minf', 'AoA', 'Cl', 'Cd', 'Cm', 'Cf', 'M1', 'X1'])

        def _external_cfd(a):
            
            suc, results = cfdfunc(a, *cfdfuncparas)
            seri.add(results)
            self.log(' > Calculation AoA = %.4f (CL = %.4f; CD = %.4f; Cf = %.4f)' % (results['AoA'], results['Cl'], results['Cd'], results['Cf']))
            
            return suc


        AoA_min = self.paras['AoA_min']
        dAoA    = self.paras['dAoAs']

        #* Initial three points
        for a in [AoA_min, AoA_min + dAoA[0]*2, AoA_min + dAoA[0]*4]:
            suc = _external_cfd(a)
        
        #* Adding more points
        _AoA = AoA_min + dAoA[0]*6  # first try
        n = 0
        while True:
            
            _AoA = float('%.3f'%(int(_AoA/dAoA[0])*dAoA[0]))
            suc = _external_cfd(_AoA)
            n += 1

            if not suc:
                _AoA -= dAoA[0]
            else:
                linear_portion = self._estimate_linear_lift_curve(seri.AoA, seri.Cl)
                AoA_ub = linear_portion['AoA_ub']
                
            if AoA_ub == seri.AoA[-1]:
                
                if _AoA + dAoA[0]*2 > self.paras['AoA_linupp']:
                    _AoA += dAoA[0]
                else:
                    _AoA += dAoA[0]*2
                
            else:
                
                _AoA = AoA_ub - dAoA[0]
            
            if _AoA in seri.AoA or _AoA > self.paras['AoA_linupp'] or _AoA < AoA_min or n >= 10:
                break

        linear_portion = self._estimate_linear_lift_curve(seri.AoA, seri.Cl)

        #* ---------------------------------------
        self.log('')
        self.log(' > Coarse points [incipient separation]')

        if True:

            _AoA = self.paras['AoA_1sep']  # first try
            n = 0
            while True:
                
                _AoA = float('%.3f'%(int(_AoA/dAoA[0])*dAoA[0]))
                suc = _external_cfd(_AoA)
                n += 1
                
                AoA_sep = self._estimate_incipient_separation(seri.AoA, seri.Cf, kind=1)

                if not suc:
                    _AoA -= dAoA[0]
                else:
                    _AoA = self._decide_outer_point(_AoA, AoA_sep, seri.AoA, dAoA[0], lAoA=self.paras['AoA_1sep'])
                
                if _AoA in seri.AoA or _AoA < AoA_min or _AoA > self.paras['AoA_max'] or n >= 10:
                    break

        #* ---------------------------------------
        self.log('')
        self.log(' > Coarse points [buffet region]')

        if True:
            
            _AoA = AoA_sep + dAoA[0]
            n = 0
            while True:
                
                _AoA = float('%.3f'%(int(_AoA/dAoA[0])*dAoA[0]))
                suc = _external_cfd(_AoA)
                
                if not suc:
                    _AoA -= dAoA[0]
                
                else:
                    
                    linear_portion, bfpoints, AoA_sep = self._adjust_linear_portion(seri.AoA, seri.Cl, seri.Cf)
                    _AoA = self._decide_outer_point(_AoA, bfpoints[1, 0], seri.AoA, dAoA[0], AoA_sep)
                    
                if _AoA in seri.AoA or _AoA < AoA_min or _AoA > self.paras['AoA_max'] or n >= 10:
                    break
            
            _, bfpoints, _ = self._predict_buffet(seri.AoA, seri.Cl, seri.Cf)

        #* ---------------------------------------
        self.log('') 
        self.log(' > Add Coarse points (from lower bound to upper bound)')
        if (bfpoints[2, 0] - bfpoints[0, 0]) // dAoA[0] >= 2:
            
            low = float('%.3f'%(int(bfpoints[0, 0]/dAoA[0])*dAoA[0]))
            upp = float('%.3f'%(math.ceil(bfpoints[2, 0]/dAoA[0])*dAoA[0]))
            calc_As = np.arange(low, upp+1e-6, dAoA[0]).tolist()
            
            for a in calc_As:
                if a not in seri.AoA:
                    suc = _external_cfd(a)
            
            _, bfpoints, _ = self._predict_buffet(seri.AoA, seri.Cl, seri.Cf)

        #* ---------------------------------------
        self.log('')
        self.log(' > Medium points (from lower bound to upper bound)')
        if True:
            
            low = bfpoints[0, 0] - dAoA[1]
            upp = bfpoints[2, 0] + dAoA[1]
            calc_As = np.arange(low, upp+1e-6, dAoA[1]).tolist()
            for a in calc_As:
                a = float('%.3f'%(int(a/dAoA[1])*dAoA[1]))
                if a not in seri.AoA:
                    suc = _external_cfd(a)

            _, bfpoints, _ = self._predict_buffet(seri.AoA, seri.Cl, seri.Cf)

        #* ---------------------------------------
        self.log('')
        self.log(' > Fine points (locate buffet onset)')

        if True:

            calc_As = [AoA_sep, bfpoints[1, 0]-dAoA[2], bfpoints[1, 0], bfpoints[1, 0]+dAoA[2]]
            
            for a in calc_As:
                a = float('%.3f'%(int(a/dAoA[2])*dAoA[2]))
                if a not in seri.AoA:
                    suc = _external_cfd(a)

            linear_portion, bfpoints, AoA_sep = self._predict_buffet(seri.AoA, seri.Cl, seri.Cf)

        if plot_name is not None:

            aa = np.arange(seri.AoA[0], seri.AoA[-1], 0.01)
            c0 = linear_portion['intercept'] + linear_portion['slope']*(aa)
            c1 = linear_portion['intercept'] + linear_portion['slope_lb']*(aa)
            c2 = linear_portion['intercept'] + linear_portion['slope_ub']*(aa)

            plt.plot(seri.AoA,  seri.Cl,  'kx-')
            plt.plot(aa, c0, 'b--')
            plt.plot(aa, c1, 'g--')
            plt.plot(aa, c2, 'r--')
            plt.plot([seri.AoA[linear_portion['i_lb']], seri.AoA[linear_portion['i_ub']]], 
                        [seri.Cl[linear_portion['i_lb']], seri.Cl[linear_portion['i_ub']]], 'ko')
            plt.show()

        return linear_portion, bfpoints, AoA_sep

    def lift_curve_break(self, seri: Series, cl_c: float = None, p: Ploter = None):
        '''
        calculate buffet point based on aerodynamic curves
        
        '''

        if self.paras['intp'] == 'pchip':
            f_cl_aoa_all = PchipInterpolator(seri.AoA, np.array(seri.Cl))
        elif self.paras['intp'] == '1d':
            f_cl_aoa_all = interp1d(seri.AoA, np.array(seri.Cl), bounds_error=False, fill_value='extrapolate')

        #* -----------------------------------
        #* decide linear segment
        # aoa_refs = list(np.arange(-2, 4, 0.01))
        # if self.paras['linear'] == 'cruise':
        #     max_i = first_argmin(abs(f_cl_aoa_all(aoa_refs) - self.paras['cl_c']))
        # elif self.paras['linear'] == 'maxLD':
        #     f_cdcl_aoa_all = PchipInterpolator(aoas, np.array(clss) / np.array(cdss))
        #     max_i = first_argmin(-f_cdcl_aoa_all(aoa_refs))
        
        # max_aoa = aoa_refs[max_i] + self.paras['lsuth1']

        # while max_aoa > aoa_refs[max_i] + self.paras['lsuth1'] - self.paras['lsuth2']:
        #     min_aoa = max(aoas[0], max_aoa - self.paras['lslt'])
        #     linear_aoas = np.arange(min_aoa, max_aoa, 0.1).reshape(-1,1)    # modified 2023.5.14, change the upper bound to max_aoa - 0.5
        #     f_cl_aoa_linear = f_cl_aoa_all(linear_aoas)
        #     reg = LinearRegression().fit(linear_aoas, f_cl_aoa_linear)
        #     if reg.score(linear_aoas, f_cl_aoa_linear) > 0.9999:
        #         break
        #     max_aoa -= 0.01
        
        # reg_k = reg.coef_[0]
        # reg_b = reg.intercept_

        # self.log(' >> linear section')
        # self.log('     reference:  AoA = %.4f (with %s method)' % (aoa_refs[max_i], self.paras['linear']))
        # self.log('     Minimal:    AoA = %.4f, Cl = %.4f' % (min_aoa, f_cl_aoa_all(min_aoa)))
        # self.log('     Maximal:    AoA = %.4f, Cl = %.4f' % (max_aoa, f_cl_aoa_all(max_aoa)))
        # self.log('     slope:      %.4f' % (reg_k))
        # self.log('     intercept:  %.4f' % (reg_b))
        # self.log('     error:      %.4f' % (reg.score(linear_aoas, f_cl_aoa_linear)))
        if 'Cd' in seri.series.keys():
            CDs = seri.Cd
        else:
            CDs = None

        if 'X1' in seri.series.keys():
            X1s = seri.X1
        else:
            X1s = None

        ll = self._estimate_linear_lift_curve(seri.AoA, seri.Cl, CDs=CDs, X1s=X1s, cl_c=cl_c)
        if 'Cf' in seri.series.keys() and self.paras['srst'] in ['sep']:
            AoA_sep = self._estimate_incipient_separation(seri.AoA, seri.Cf)
            # print(seri.Cf, AoA_sep)
        else:
            AoA_sep = ll['AoA_ub'] + 0.5

        #* -----------------------------------
        #* find buffet onset
        # f_cl_aoa = pchip(aoas[max_i:], clss[max_i:])
        
        upp_bound = seri.AoA[-1] + 0.8
        step_aoa = np.arange(max(0.2, AoA_sep), upp_bound, 0.001)

        flags = {}
        bufs = np.ones((3, 2)) * (-10.0)

        for idx, key in enumerate(['slope_ub', 'slope', 'slope_lb']):
            # larger slope means smaller buffet lift coefficient
            try:
                bufs[idx] = Buffet.intersection(step_aoa, 
                        ll[key] * (step_aoa - self.paras['daoa']) + ll['intercept'], f_cl_aoa_all(step_aoa))
                flags[key] = True
            except IntersectNotFound:
                flags[key] = False

        # if bufs[2, 1] < bufs[0, 1]:
        #     plt.plot(seri.AoA, np.array(seri.Cl), 'o')
        #     plt.plot(step_aoa, f_cl_aoa_all(step_aoa), c='k')
        #     plt.plot(step_aoa, ll['slope_ub'] * (step_aoa - self.paras['daoa']) + ll['intercept'], c='C0')
        #     plt.plot(step_aoa, ll['slope'] * (step_aoa - self.paras['daoa']) + ll['intercept'], c='C1')
        #     plt.plot(step_aoa, ll['slope_lb'] * (step_aoa - self.paras['daoa']) + ll['intercept'], c='C2')
        #     print(bufs[:, 1], ll)
        #     plt.show()

        if flags['slope_ub']:
            # lower buffet boundary found
            if not flags['slope']: bufs[1] = bufs[0]
            if not flags['slope_lb']: bufs[2] = bufs[1]
        else:
            # modified 2023.5.14, change the upper bound to aoa[-1]+0.5(pchip is valid for extrapolate)
            # change the lower bound to aoa_cruise + 0.1, if not found, search max_aoa
            step_aoa = np.arange(ll['AoA_ub'] - 0.1, upp_bound, 0.001)
            try:
                bufs[0] = Buffet.intersection(step_aoa, 
                        ll['slope_ub'] * (step_aoa - self.paras['daoa']) + ll['intercept'], f_cl_aoa_all(step_aoa))
                if not flags['slope']:
                    bufs[1] = Buffet.intersection(step_aoa, 
                        ll['slope'] * (step_aoa - self.paras['daoa']) + ll['intercept'], f_cl_aoa_all(step_aoa))
                if not flags['slope_lb']:
                    bufs[2] = Buffet.intersection(step_aoa, 
                        ll['slope_lb'] * (step_aoa - self.paras['daoa']) + ll['intercept'], f_cl_aoa_all(step_aoa))
            except IntersectNotFound:
                self.log(' >> buffet onset  not found')

        # swap lower and upper boundary if values are inversed
        if bufs[1, 1] > bufs[0, 1] and bufs[1, 1] > bufs[2, 1]:
            bufs[0, 1] = min(bufs[0, 1], bufs[2, 1])
            bufs[2, 1] = bufs[1, 1]
        elif bufs[1, 1] < bufs[0, 1] and bufs[1, 1] > bufs[2, 1]:
            buf_t = bufs[0, 1]
            bufs[0, 1] = bufs[2, 1]
            bufs[2, 1] = buf_t

        self.log(' >> buffet onset')
        self.log('     AoA = %.4f ([%.4f, %.4f]), Cl = %.4f ([%.4f, %.4f])' % (bufs[1,0], bufs[0,0], bufs[2,0], bufs[1,1], bufs[0,1], bufs[2,1]))
            
        if p is not None:
            hl_aoa = np.arange(seri.AoA[0], seri.AoA[-1]+0.4, 0.01)
            aoas = np.array(seri.AoA)
            # plt.plot(aoas - p.ref_aoa, seri.Cl, p.symbol, c=p.color, label=p.name) #!
            # plt.plot(aoas - p.ref_aoa, seri.X1, p.symbol, c=p.color, label=p.name)
            # plt.plot(aoas[ll['i_lb']:ll['i_ub']+1] - p.ref_aoa, seri.Cl[ll['i_lb']:ll['i_ub']+1], '-', c=p.color, label=p.name)
            # plt.plot(hl_aoa - p.ref_aoa, f_cl_aoa_all(hl_aoa), '--', c=p.color) #!
            plt.plot([-2 - p.ref_aoa, bufs[1,0] - p.ref_aoa], 
                     [ll['slope'] * (-2 - self.paras['daoa']) + ll['intercept'], ll['slope'] * (bufs[1,0] - self.paras['daoa']) + ll['intercept']], 
                     '-.', c=p.color, lw=0.8)
            # plt.plot([AoA_sep - p.ref_aoa], f_cl_aoa_all(AoA_sep), 's', c=p.color) #!
            plt.plot([bufs[1,0] - p.ref_aoa], [bufs[1,1]], '^', c=p.color)
            if p.instant_plot:
                plt.show()    

        return bufs

    def curve_slope_diff(self, seri: Series, p: Ploter = None):
        dAoA = self.aoadelta
        AoAGrid = np.arange(seri.AoA[0], seri.AoA[-1] - self.aoadelta + 0.5, dAoA)
        fcria = self.interpolate(seri.AoA, seri.series[self.paras['dfcri']], kind=self.paras['intp'])[0]
        fcla  = self.interpolate(seri.AoA, seri.Cl, kind=self.paras['intp'])[0]
        ll = self._estimate_linear_lift_curve(seri.AoA, seri.Cl, seri.Cd, seri.X1)
        coef = ll['slope'] - self.paras['diff']
        flag = False
        
        dYdX = []
        aoa_buf = -10.0
        cl_buf = -10.0

        for i in range(1, AoAGrid.shape[0] -1):

            pointp1 = fcria(AoAGrid[i + 1])
            point1 =  fcria(AoAGrid[i])
            pointm1 = fcria(AoAGrid[i - 1])

            localdCLdA = 0.5 * (pointp1 - pointm1) / dAoA
            
            if localdCLdA < coef and dYdX[-1] > coef:
                aoa_buf = AoAGrid[i]
                cl_buf = fcla(AoAGrid[i])
                flag = True

            dYdX.append(localdCLdA)
        
        if not flag:
            print('buffet not found')

        self.process_values = dYdX
        return (aoa_buf, cl_buf)

    def cruve_critical_value(self, seri: Series, p: Ploter = None):

        if self.paras['crmod'] == 'max':
            md = 1
        elif self.paras['crmod'] == 'min':
            md = -1
        else:
            raise ValueError('The mode (crmod) cruve critical value method should be min or max')

        dAoA = 0.01
        max_i = first_argmin(abs(np.array(seri.AoA)))
        AoAGrid  = np.arange(seri.AoA[max_i], seri.AoA[-1] + 0.5 - dAoA, dAoA)

        fcria = self.interpolate(seri.AoA, seri.series[self.paras['dfcri']], kind=self.paras['intp'])[0]
        fcla  = self.interpolate(seri.AoA, seri.Cl, kind=self.paras['intp'])[0]
        flag = False

        values = [] # need to compare with RCL[-1]
        aoa_buf = -10.0
        cl_buf = -10.0
        for i in range(1, AoAGrid.shape[0] -1):
            
            pointp1 = fcria(AoAGrid[i + 1])
            point1 =  fcria(AoAGrid[i])
            pointm1 = fcria(AoAGrid[i - 1])
            
            if self.paras['crodr'] == 2:
                localvalue = Buffet.curvature(-dAoA, pointm1, 0., point1, dAoA, pointp1)
            elif self.paras['crodr'] == 1:
                localvalue = 0.5 * (pointp1 - pointm1) / dAoA
            elif self.paras['crodr'] == 0:
                localvalue = point1
            else:
                raise ValueError('The cruve critical value method only deal with origin value (crodr = 0), \
                        gradient value (crodr = 1), and curvature (crodr = 2)')
            
            localvalue *= md
            if len(values) >= 2 and localvalue < values[-1] and values[-1] > values[-2]:
                aoa_buf = Buffet.localmax(AoAGrid[i-2], values[-2], AoAGrid[i-1], values[-1], AoAGrid[i], localvalue)
                print(AoAGrid[i-1], aoa_buf)
                cl_buf = fcla(aoa_buf)
                flag = True
                # break

            values.append(localvalue)
        
        values.pop(0) # pop the first element, -1

        if not flag:
            print('buffet not found')
        
        if p is not None:
            plt.plot(seri.AoA, seri.series[self.paras['dfcri']], 'o', c=p.color)
            plt.plot(AoAGrid, fcria(AoAGrid), '-', c='gray')
            plt.plot(AoAGrid[1:len(values)+1], values, '--', c=p.color)
            plt.show()
        
        self.process_values = values
        return (aoa_buf, cl_buf)

    def shock_to_maxcuv(self, seri: Series, maxcuv: float):
        
        try:
            aoa_buf, _ = self.intersection(seri.AoA, seri.X1, [maxcuv for _ in seri.AoA])
            fcla  = self.interpolate(seri.AoA, seri.Cl, kind=self.paras['intp'])[0]
            cl_buf = fcla(aoa_buf)
        except IntersectNotFound:
            aoa_buf = -10.0
            cl_buf  = -10.0
            print('buffet not found')

        return (aoa_buf, cl_buf)

    def incipient_separation(self, seri: Series):

        AoA_sep = self._estimate_incipient_separation(seri.AoA, seri.Cf, kind=self.paras['intp'])
        return AoA_sep

'''
old version of get buffet

'''
def get_buffet(aoas, clss, cdss, d_aoa=0.1, linear='cruise', cl_c=0.8, plot=False, intp='pchip', **kwargs):

    # f_idx = 0
    # for idx, cd in enumerate(cdss):
    #     if cd > 0:
    #         f_idx = idx
    #         break
    # print(f_idx)
    # print(np.array(clss[f_idx:]) / np.array(cdss[f_idx:]))
    # plt.plot(clss, cdss)
    # plt.show()

    if 'lsuth1' not in kwargs.keys():       kwargs['lsuth1'] = 0.0      # linear segment upper bound therhold upper
    if 'lsuth2' not in kwargs.keys():       kwargs['lsuth2'] = 1.2      # linear segment upper bound therhold lower
                                                                            # the u.b. is searched down from start point not less than this value
                                                                            # to let r2 > 0.9999 (for LRZ is 1.0, for SFY is 0.01)
    if 'lslt' not in kwargs.keys():         kwargs['lslt'] = 3.0        # linear segment length (for LRZ is 3.0, for SFY is 1.0)

    if intp == 'pchip':
        f_cl_aoa_all = pchip(aoas, np.array(clss))
    elif intp == '1d':
        f_cl_aoa_all = interp1d(aoas, np.array(clss), bounds_error=False, fill_value='extrapolate')

    aoa_refs = list(np.arange(-2, 4, 0.01))

    if linear == 'cruise':
        max_i = first_argmin(abs(f_cl_aoa_all(aoa_refs) - cl_c))
    elif linear == 'maxLD':
        f_cdcl_aoa_all = pchip(aoas, np.array(clss) / np.array(cdss))
        max_i = first_argmin(-f_cdcl_aoa_all(aoa_refs))

    max_aoa = max_aoa = aoa_refs[max_i] + kwargs['lsuth1']
    # print(max_i, max_aoa)
    # input()

    while max_aoa > aoa_refs[max_i] + kwargs['lsuth1'] - kwargs['lsuth2']:
        min_aoa = max(aoas[0], max_aoa - kwargs['lslt'])
        linear_aoas = np.arange(min_aoa, max_aoa, 0.1).reshape(-1,1)    # modified 2023.5.14, change the upper bound to max_aoa - 0.5
        f_cl_aoa_linear = f_cl_aoa_all(linear_aoas)
        reg = LinearRegression().fit(linear_aoas, f_cl_aoa_linear)
        if reg.score(linear_aoas, f_cl_aoa_linear) > 0.9999:
            break
        max_aoa -= 0.01

    if plot:
        print(max_aoa, aoa_refs[max_i], f_cl_aoa_all(max_aoa), f_cl_aoa_all(aoa_refs[max_i]))
        print(reg.score(linear_aoas, f_cl_aoa_linear))
    # print(reg.coef_[0], reg.intercept_)
    reg_k = reg.coef_[0]
    reg_b = reg.intercept_

    d_b = - d_aoa * reg_k

    # f_cl_aoa = pchip(aoas[max_i:], clss[max_i:])
    upp_bound = aoas[-1]+0.3
    for low_bound in [aoa_refs[max_i]+0.1, max_aoa-0.1]:
        # modified 2023.5.14, change the upper bound to aoa[-1]+0.5(pchip is valid for extrapolate)
        # change the lower bound to aoa_cruise + 0.1, if not found, search max_aoa
        step_aoa = np.arange(low_bound, upp_bound, 0.001)

        delta = f_cl_aoa_all(step_aoa) - (reg_k * step_aoa + reg_b + d_b)

        for idx in range(len(delta)-1):
            if delta[idx] * delta[idx+1] < 0:
                aoa_buf = step_aoa[idx] + 0.001 * delta[idx] / (delta[idx] - delta[idx+1])
                cl_buf = f_cl_aoa_all(aoa_buf)
                break
        else:
            aoa_buf = None
            cl_buf = None
            continue
        break
    if aoa_buf is None:
        print('Warning:  buffet not found')

    if plot:
        plt.plot(linear_aoas, f_cl_aoa_all(linear_aoas), '-', c='C0')
        plt.plot(step_aoa, f_cl_aoa_all(step_aoa), '-', c='C1')
        plt.plot([0, 4.5], [reg_b + d_b, reg_k * 4.5 + reg_b + d_b], '--', c='k')
        # plt.show()

    # print(aoa_buf, cl_buf)

    return (aoa_buf, cl_buf)