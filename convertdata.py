# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 13:26:20 2014

@author: Administrator
"""

import numpy as np, numexpr as ne
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
import re
import matplotlib.pyplot as plt
import os


class Test:

    def __init__(self, test_id, fs=1e3):
        # Defining basic properties
        self.test_id = test_id
        self.fs = fs
        # Load all data
        self.force_trace_list, self.displ_trace_list, contact_displ_skin = \
            self.load_data(1, duration=5.)
        contact_displ_noskin = self.load_data(2, duration=1.)[2]
        self.skin_thickness = contact_displ_noskin - contact_displ_skin
        return

    def load_formula(self, from_file=False):
        if from_file:
            with open(
                './YoshiExperiment/VirginiaIndenterLog_2014-12-22_0%d.txt'\
                % self.test_id, 'r') as f:
                log = f.read()
            force_formula = re.findall('Function:\s+\S+', log)[0][9:].replace(
                ' ', '')
        else:
            force_formula = '7.5079*x-0.389478308184'
        displ_formula = '(x-1.)*10./4.'
        return force_formula, displ_formula

    def load_data(self, block_id, duration):
        force_formula, displ_formula = self.load_formula()
        data = loadmat('./YoshiExperiment/2014-12-22-0%d0%d.mat' %
            (self.test_id, block_id))
        # Check consistency of sampling rate
        assert np.diff(data['samplerate'].ravel()).ravel().nonzero()[0].size\
            == 0, 'Sampling rate not consistent between channels'
        rawfs = data['samplerate'][0][0]
        # Defining start and end of data
        datastart = data['datastart'].astype('int').ravel() - 1
        dataend = data['dataend'].astype('int').ravel()
        # Load voltage arrays
        force_voltage = data['data'][0][datastart[0]:dataend[0]].ravel()[
            ::int(rawfs/self.fs)]
        displ_voltage = data['data'][0][datastart[1]:dataend[1]].ravel()[
            ::int(rawfs/self.fs)]
        # Convert to mN and mm
        def apply_formula(formula, x):
            output = ne.evaluate(formula)
            return output
        force_trace_all = apply_formula(force_formula, force_voltage)
        displ_trace_all = apply_formula(displ_formula, displ_voltage)
        force_trace_all -= force_trace_all[:int(self.fs/100)].mean()
        displ_trace_all -= displ_trace_all[:int(self.fs/100)].mean()
        # Cut into small traces
        cutoff_displ = .1
        mask = displ_trace_all < cutoff_displ
        force_trace_ma = np.ma.array(force_trace_all, mask=mask)
        slices = np.ma.extras.notmasked_contiguous(force_trace_ma)
        force_trace_slice_list = [force_trace_all[slice_] for slice_ in slices
            if slice_.stop - slice_.start > self.fs*duration/2]        
        displ_trace_slice_list = [displ_trace_all[slice_] for slice_ in slices
            if slice_.stop - slice_.start > self.fs*duration/2]        
        # Trim traces
        force_trace_list, displ_trace_list = [], []
        contact_displ_list = []
        contact_threshold = .5
        for i, force_trace_slice in enumerate(force_trace_slice_list):
            displ_trace_slice = displ_trace_slice_list[i]
            if force_trace_slice.max() > contact_threshold:
                contact_index = (force_trace_slice > contact_threshold
                    ).nonzero()[0][0]
                contact_displ_list.append(displ_trace_slice_list[i]
                    [contact_index])
                end_index = contact_index+int(self.fs*duration)
                if not force_trace_slice[end_index] - force_trace_slice[
                    contact_index] < 3:
                    force_trace_list.append(force_trace_slice[
                        contact_index:end_index] - force_trace_slice[
                        contact_index])
                    displ_trace_list.append(displ_trace_slice[
                        contact_index:end_index] - displ_trace_slice[
                        contact_index])
        contact_displ = np.median(contact_displ_list)
        return force_trace_list, displ_trace_list, contact_displ
    



if __name__ == '__main__':
    # Switches
    run_calibration = False
    # Instantiation
    test = Test(test_id=2)
#    use_trace = range(len(test.force_trace_list))
    use_trace = [2, 3, 4]
    force_trace_list, displ_trace_list = [], []
    for i in use_trace:
        force_trace_list.append(test.force_trace_list[i])
        displ_trace_list.append(test.displ_trace_list[i])
    # Set material properties
    thickness = 225.33
    mu, alpha, tau1, tau2, g1, g2 = 6.354e3, 8.787, 0.092, 1.111, 0.482, 0.110
    material = [thickness, mu, alpha, tau1, tau2, g1, g2]
    csv_folder = 'X:/WorkFolder/AbaqusFolder/Viscoelasticity/csvFiles/'
    np.savetxt(csv_folder + 'valid_material.csv', material, delimiter=',')
    if run_calibration:
        # Set up calibration stim
        depth = np.linspace(0.05, 0.12, 5) * 1e-3
        ramp_time = np.ones_like(depth) * 0.1
        output = np.c_[ramp_time, depth]
        np.savetxt(csv_folder + 'valid_stim.csv', output, delimiter=',')
        os.system('call \"C:/SIMULIA/Abaqus/Commands/abaqus.bat\" cae '+\
            'script=X:/WorkFolder/AbaqusFolder/Viscoelasticity/'+\
            'runValidationCalibration.py')
            # Read calibration
        model_force_list, model_displ_list = [], []
        for fname in os.listdir(csv_folder):
            if fname.startswith('validation') and int(fname[10])<depth.size:
                os.rename(csv_folder+fname, csv_folder+fname.replace(
                    'validation', 'calibration'))
    # Read calibration
    model_force_list, model_displ_list = [], []
    for fname in os.listdir(csv_folder):
        if fname.startswith('calibration') and int(fname[11])<len(use_trace):
            _, force, displ = np.loadtxt(csv_folder+fname, delimiter=','
                ).T
            model_force_list.append(force[-1]*1e3)
            model_displ_list.append(displ[-1]*1e3)
    exp_force_list = [force_trace[-1] for force_trace in force_trace_list]
    exp_displ_list = [displ_trace[-1] for displ_trace in displ_trace_list]
    def get_r2(a, abq_force, abq_displ, static_force, exp_displ, sign=1.):
        abq_force = np.array(abq_force)
        abq_displ = np.array(abq_displ)
        static_force = np.array(static_force)
        exp_displ = np.array(exp_displ)
        abq_displ_scaled = a[0] + a[1] * abq_displ
        p = np.polyfit(abq_displ_scaled, abq_force, 3)
        abq_force_interp = np.polyval(p, exp_displ)
        sst = static_force.var(ddof=1) * static_force.shape[0]
        sse = np.linalg.norm(static_force - abq_force_interp) ** 2
        r2 = 1 - sse / sst
        return sign * r2
    res = minimize(get_r2, [0.1, 2.], args=(model_force_list,
                   model_displ_list, exp_force_list, exp_displ_list,
                   -1.), method='L-BFGS-B')
    displ_coeff = res.x
    # Generate parameters for Abaqus runs
    max_idx_list = np.array([force_trace.argmax()
        for force_trace in force_trace_list])
    depth = np.array([displ_trace[-1] for displ_trace in displ_trace_list])
    ramp_time = np.array(max_idx_list) / test.fs
    output = np.c_[ramp_time, (depth-displ_coeff[0])/displ_coeff[1]*1e-3]
    np.savetxt(csv_folder + 'valid_stim.csv', output, delimiter=',')
    # Run abaqus
    os.system('call \"C:/SIMULIA/Abaqus/Commands/abaqus.bat\" cae '+\
        'script=X:/WorkFolder/AbaqusFolder/Viscoelasticity/runValidation.py')
    # Read output data from abaqus
    model_time_list, model_force_list, model_displ_list = [], [], []
    for fname in os.listdir(csv_folder):
        if fname.startswith('validation') and int(fname[10])<len(use_trace):
            time, force, displ = np.loadtxt(csv_folder+fname, delimiter=','
                ).T
            model_time_list.append(time)
            model_force_list.append(force*1e3)
            model_displ_list.append(displ*1e3)
    # Plot result
    fig, axs = plt.subplots()
    for force_trace in force_trace_list:
        plt.plot(np.arange(force_trace.shape[0])/test.fs, force_trace, '-', c='.5')
    for i, model_force in enumerate(model_force_list):
        model_time = model_time_list[i]
        plt.plot(model_time, model_force, '-r')
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Force (mN)')
    fig.tight_layout()
    fig.savefig('1.png')
    # Plot the static force displ curves
    fig, axs = plt.subplots()
    static_displ_list = [displ[-1] for displ in displ_trace_list]
    static_force_list = [force[-1] for force in force_trace_list]
    for i in range(len(static_displ_list)):
        axs.plot(static_displ_list[i], static_force_list[i], 
                 marker=r'$%d$'%i, ms=15)
    axs.set_xlabel('Static displ. (mm)')
    axs.set_ylabel('Static force (mN)')
    # Plot test traces
    fig, axs = plt.subplots(2, 1)
    for i, displ in enumerate(displ_trace_list):
        force = force_trace_list[i]
        axs[0].plot(displ, label='#%d'%use_trace[i])
        axs[1].plot(force, label='#%d'%use_trace[i])
    axs[0].legend(loc=4)
    axs[1].legend(loc=1)
    # %% Try spline fitting
    fig, axs = plt.subplots()
    time = np.arange(displ.shape[0]) / test.fs
    spline = UnivariateSpline(time, displ, s=1e-2)
    new_time = np.logspace(0, np.log10(6), 50) - 1
    new_displ = spline(new_time)
    new_displ[0] = 0
    new_displ[-1] = displ[-100:].mean()
    axs.plot(time, displ, '.k')
    axs.plot(new_time, new_displ, '-r')
    

