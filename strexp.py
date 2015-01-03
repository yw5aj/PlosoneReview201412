# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 11:57:37 2014

@author: Yuxiang Wang
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def main():
    return


def qlv_prony(stretch_array, visco_params, hyper_params, dt=1e-3, sign=1., 
    nu=.5):
    """ 
    Wrapper for the QLV model with Prony series.
    """
    return qlv(stretch_array, visco_params, hyper_params, 
               relax_func=relax_func_prony, dsdt_func=dsdt_func_ogden, dt=dt, 
               sign=sign, nu=nu)


def qlv(stretch_array, visco_params, hyper_params, relax_func, dsdt_func, 
        dt=1e-3, sign=1, nu=.5):
    time_array = np.arange(stretch_array.shape[0]) * dt
    relax_array = relax_func(time_array, *visco_params)
    dsdt_array = dsdt_func(stretch_array, *hyper_params, dt=dt, nu=nu)
    stress_array = np.convolve(relax_array, dsdt_array, mode='full')[
        :time_array.shape[0]] * dt
    return sign * stress_array


def ogden(stretch, *ogden_params, sign=1., nu=.5):
    mu, alpha = ogden_params
    if nu == .5:
        stress = 2. * mu / alpha * (stretch**alpha - stretch**(-.5 * alpha))
    else:
        stress = 4. * mu / 3. / alpha * stretch**((1. - 2. * nu) * (-1. 
            - alpha / 3.)) * (stretch**alpha - stretch**(-nu * alpha)) \
            + 20. * mu * (stretch**(1. - 2. * nu) - 1.)
    return sign * stress


def relax_func_prony(time_array, *visco_params):
    tau1, tau2, g1, g2 = visco_params
    relax_array = g1 * np.exp(-time_array / tau1) \
        + g2 * np.exp(-time_array / tau2) \
        + 1. - g1 - g2
    return relax_array


def relax_func_strexp(time_array, *visco_params):
    beta1, beta2, g1, g2 = visco_params
    relax_array = g1 * np.exp(-time_array**beta1) \
        + g2 * np.exp(-time_array**beta2) \
        + 1. - g1 - g2
    return relax_array


def dsdt_func_ogden(stretch_array, *hyper_params, dt=1e-3, nu=.5):
    dsdt_array = np.diff(ogden(stretch_array, *hyper_params, nu=nu)) / dt
    dsdt_array = np.r_[0., dsdt_array]
    return dsdt_array


def read_stress_stretch(num):
    stress_array, stretch_array = np.genfromtxt(
        '../StressStretchTraces/%dStressStretchForceDisp.csv' % num, 
        delimiter=',').T[:2]
    return stress_array, stretch_array


def get_r2_visco(visco_params, stretch_array, stress_array, hyper_params, 
                 relax_func, dsdt_func, dt=1e-3, qlv_sign=-1., sign=1.):
    stress_model = qlv(stretch_array, visco_params, hyper_params, relax_func, 
                       dsdt_func, dt=dt, sign=qlv_sign)
    max_index = stress_array.argmax()
    r2 = .5 * (get_r2(stress_array[:max_index], stress_model[:max_index]) + 
               get_r2(stress_array[max_index:], stress_model[max_index:]))
    return sign * r2


def get_r2(exp, model):
    sst = exp.var() * exp.shape[0]
    sse = np.linalg.norm(exp-model)**2
    r2 = 1. - sse / sst
    return r2


if __name__ == '__main__':
    # Load Prony series data
    dt = 1e-3
    prony_params = [.02, .28, .63, .27]
    ogden_params = [1378.61*10.32/2, 10.32]
    stress_array, stretch_array = read_stress_stretch(1)
    stress_model_prony = qlv(stretch_array, prony_params, 
        ogden_params, relax_func_prony, dsdt_func_ogden, sign=-1.)
    time_array = np.arange(stress_array.shape[0]) * dt
    bounds = [(0, 1) for i in range(4)]
    # Fit to stretched exponential function
    x0 = [.1, .5, .2, .6]
    res = minimize(get_r2_visco, x0=x0, args=(stretch_array, 
                   stress_array, ogden_params, relax_func_strexp, 
                   dsdt_func_ogden, dt, -1., -1.), method='L-BFGS-B', 
                   bounds=bounds)
    strexp_params = res.x
    strexp_r2 = -res.fun
    stress_model_strexp = qlv(stretch_array, strexp_params, ogden_params,
                              relax_func_strexp, dsdt_func_ogden, sign=-1.)
    # Plot results
    r2_list = [.9794395, strexp_r2]
    fig, axs = plt.subplots(2, 1, figsize=(3.27, 6))
    axs[0].plot(time_array, stress_array*1e-3, '.', color='.5', 
        label="Experiment")
    axs[0].plot(time_array, stress_model_prony*1e-3, '-k', 
        label="Model")    
    axs[1].plot(time_array, stress_array*1e-3, '.', color='.5', 
        label="Experiment")
    axs[1].plot(time_array, stress_model_strexp*1e-3, '-k', 
        label="Model")    
    for i, label in enumerate(['A', 'B']):
        axes = axs[i]
        axes.set_xlim(-.5, 5.5)
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Stress (kPa)")
        axes.legend()        
        axes.text(-0.12, 1., label, transform=axes.transAxes, fontsize=12, 
            fontweight='bold', va='top')
        annotation = ["Prony series\n", "Stretched exponential\n", 
            "Polynomial\n"][i] + r"$R^2=$%.2f"%r2_list[i]
        axes.text(.35, .5, annotation, transform=axes.transAxes, 
                  fontsize=10, fontweight='normal', ha='left', va='top', 
                  multialignment='center')
    fig.tight_layout()
    fig.savefig('./figs/strexp.png', dpi=300)
    

