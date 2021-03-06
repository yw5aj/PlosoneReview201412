# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 13:26:20 2014

@author: Administrator
"""

import numpy as np, numexpr as ne
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.interpolate import LSQUnivariateSpline
from scipy import signal
import re
import matplotlib.pyplot as plt
import os


class Test:

    def __init__(self, test_id, fs=1e3):
        # Defining basic properties
        self.test_id = test_id
        self.block_id = 1
        self.fs = fs
        # Load all data
        self.force_trace_list, self.displ_trace_list, contact_displ_skin = \
            self.load_data(duration=5.)
        # Hack to get skin_thickness
        self.block_id = 2
        contact_displ_noskin = self.load_data(duration=1.)[2]
        self.block_id = 1
        self.skin_thickness = contact_displ_noskin - contact_displ_skin
        return

    def load_formula(self, force_formula='7.5079*x-0.389478308184'):
        displ_formula = '(x-1.)*10./4.'
        return force_formula, displ_formula

    def get_mat_fname(self):
        fname = './YoshiExperiment/2014-12-22-0%d0%d.mat' %\
            (self.test_id, self.block_id)
        return fname
    
    def load_data(self, duration):
        force_formula, displ_formula = self.load_formula()
        data = loadmat(self.get_mat_fname())
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
        b, a = signal.butter(8, 50/self.fs)
        force_trace_all = signal.filtfilt(b, a, force_trace_all)
        displ_trace_all = signal.filtfilt(b, a, displ_trace_all)
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
        contact_index_list, contact_displ_list = [], []
        contact_threshold = .5
        for i, force_trace_slice in enumerate(force_trace_slice_list):
            displ_trace_slice = displ_trace_slice_list[i]
            if force_trace_slice.max() > contact_threshold:
                contact_index = (force_trace_slice > contact_threshold
                    ).nonzero()[0][0]
                contact_index_list.append(contact_index)
                contact_displ_list.append(displ_trace_slice_list[i]
                    [contact_index])
        contact_index = int(np.median(contact_index_list))
        for i, force_trace_slice in enumerate(force_trace_slice_list):
            displ_trace_slice = displ_trace_slice_list[i]
            if force_trace_slice.max() > contact_threshold:
                end_index = contact_index+int(self.fs*duration)
                if not force_trace_slice[end_index] - force_trace_slice[
                    contact_index] < 1:
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
    run_model = False
    # Instantiation
    test = Test(test_id=2)
#    use_trace = range(len(test.force_trace_list))
    use_trace = [1, 2, 4, 5]
    force_trace_list, displ_trace_list = [], []
    for i in use_trace:
        force_trace_list.append(test.force_trace_list[i])
        displ_trace_list.append(test.displ_trace_list[i])
    time_trace_list = []
    for displ_trace in displ_trace_list:
        time_trace_list.append(np.arange(displ_trace.size)/test.fs)
    # Set material properties
    thickness = 225.33
    mu, alpha, tau1, tau2, g1, g2 = 6.354e3, 8.787, 0.092, 1.111, 0.482, 0.110
    material = [thickness, mu, alpha, tau1, tau2, g1, g2]
    csv_folder = 'X:/WorkFolder/AbaqusFolder/Viscoelasticity/csvFiles/'
    np.savetxt(csv_folder + 'valid_material.csv', material, delimiter=',')
    if run_calibration:
        # Set up calibration stim
        depth = np.linspace(0.08, 0.12, 5) * 1e-3
        ramp_time = np.ones_like(depth) * 0.1
        output = np.c_[ramp_time, depth]
        np.savetxt(csv_folder + 'valid_stim.csv', output, delimiter=',')
        os.system('call \"C:/SIMULIA/Abaqus/Commands/abaqus.bat\" cae '+\
            'script=X:/WorkFolder/AbaqusFolder/Viscoelasticity/'+\
            'runValidationCalibration.py')
    # Read calibration
    model_force_list, model_displ_list = [], []
    for fname in os.listdir(csv_folder):
        if fname.startswith('calibration') and int(fname[11])<5:
            time, force, displ = np.loadtxt(csv_folder+fname, delimiter=','
                ).T
            model_force_list.append(np.interp(2.5, time, force)*1e3)
            model_displ_list.append(np.interp(2.5, time, displ)*1e3)
    exp_force_list = [force_trace[2500] for force_trace in force_trace_list]
    exp_displ_list = [displ_trace[2500] for displ_trace in displ_trace_list]
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
    fig, axs = plt.subplots()
#    for i, displ in enumerate(displ_trace_list):
#        time = time_trace_list[i]
#        force = force_trace_list[i]
#        peaktime = time[force.argmax()]
#        abq_time = np.array([0, peaktime, 5])
#        abq_displ = np.array([0., displ[2500], displ[2500]])
#        axs.plot(abq_time, abq_displ, '-r')
#        axs.plot(time, displ, '.k')
#        output = np.c_[abq_time, (abq_displ-displ_coeff[0])/displ_coeff[1]/1e3]
#        np.savetxt(csv_folder + 'valid_stim%d.csv'%i, output, delimiter=',')
    # Manually write input
    peaktime = np.mean([time_trace_list[i][force_trace_list[i].argmax()]
        for i in (0, 2)])
    meandispl = np.mean([displ_trace_list[i][2500] for i in (0, 2)])
    abq_time = np.array((0., peaktime, 5.))
    abq_displ = np.array((0., meandispl, meandispl))
    output = np.c_[abq_time, (abq_displ-displ_coeff[0])/displ_coeff[1]/1e3]
    np.savetxt(csv_folder + 'valid_stim0.csv', output, delimiter=',')    
    peaktime = np.mean([time_trace_list[i][force_trace_list[i].argmax()]
        for i in (1, 3)])
    meandispl = np.mean([displ_trace_list[i][2500] for i in (1, 3)])
    abq_time = np.array((0., peaktime, 5.))
    abq_displ = np.array((0., meandispl, meandispl))
    output = np.c_[abq_time, (abq_displ-displ_coeff[0])/displ_coeff[1]/1e3]
    np.savetxt(csv_folder + 'valid_stim1.csv', output, delimiter=',')        
    # Run abaqus
    if run_model:
        os.system('call \"C:/SIMULIA/Abaqus/Commands/abaqus.bat\" cae '+\
            'script=X:/WorkFolder/AbaqusFolder/Viscoelasticity/'+\
            'runValidation.py')
    # %% Read output data from abaqus
    model_time_list, model_force_list, model_displ_list = [], [], []
    for fname in os.listdir(csv_folder):
        if fname.startswith('validation') and int(fname[10])<2:
            time, force, displ = np.loadtxt(csv_folder+fname, delimiter=','
                ).T
            model_time_list.append(time)
            model_force_list.append(force*1e3)
            model_displ_list.append(displ*1e3)
    def get_r2(model_time, model_force, exp_time, exp_force):
        exp_time = exp_time[exp_force.argmax():]
        exp_force = exp_force[exp_force.argmax():]        
        fine_model_force = np.interp(exp_time, model_time, model_force)
        sstot = exp_force.var() * exp_force.shape[0]
        ssres = ((exp_force - fine_model_force)**2).sum()
        r2 = 1. - ssres / sstot
        return r2
    r2_list = []
    r2_list.append(
        get_r2(model_time_list[0], model_force_list[0], time_trace_list[0],
        (force_trace_list[0]+force_trace_list[2])/2))
    r2_list.append(
        get_r2(model_time_list[1], model_force_list[1], time_trace_list[1],
        (force_trace_list[1]+force_trace_list[3])/2))
    print(r2_list)            
    # %% Plot result
    fig, axs = plt.subplots(3, 1, figsize=(3.27, 7.5))
    # Get schematic
    im = plt.imread('validation_schematic.png')
    axs[0].imshow(im)
    axs[0].axis('off') 
    # Get model
    im = plt.imread('validation_fem.png')
    axs[1].imshow(im)
    axs[1].axis('off')
    axs[1].axvline(x=5, ls='--', lw=1.5, c='k', dashes=(8, 3, 2, 3))
    axs[1].text(-0.01, 0.5, 'Symmetric axis', va='center', ha='right',
        rotation='vertical', transform=axs[1].transAxes, size=12)
    axs[1].axhline(y=im.shape[0]-5, ls='-', lw=1.5, c='k')
    axs[1].text(0.5, -0.01, 'Compression table', va='top', ha='center',
        rotation='horizontal', transform=axs[1].transAxes, size=12)
    # Plot data trace
    for force_trace in force_trace_list:
        axs[2].plot(np.arange(force_trace.shape[0])/test.fs, force_trace, '-', 
                 c='.7', label='Experiment')
    for i, model_force in enumerate(model_force_list):
        model_time = model_time_list[i]
        axs[2].plot(model_time, model_force, '-k', label='Model')
    handles, labels = axs[2].get_legend_handles_labels()
    axs[2].legend(handles[3:5], labels[3:5], loc=4)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Force (mN)')
    axs[2].set_xlim(0, 5)
    for i in range(2):
        time, force = model_time_list[i], model_force_list[i]
        axs[2].text(2.5, np.interp(2.5, time, force) - (-1)**i * 10, 
             r'${R^2}$ = %.3f' % r2_list[i],
             ha='center', va='center', size=8)
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.125, 1.05, chr(65+axes_id), transform=axes.transAxes,
            fontsize=12, fontweight='bold', va='top')   
    fig.tight_layout()
    fig.savefig('validation.png', dpi=300)
    fig.savefig('validation.tif', dpi=300)
    plt.close(fig)
    # %% Plot the static force displ curves
    fig, axs = plt.subplots()
    static_displ_list = [displ[2500] for displ in displ_trace_list]
    static_force_list = [force[2500] for force in force_trace_list]
    for i in range(len(static_displ_list)):
        axs.plot(static_displ_list[i], static_force_list[i], 
                 marker=r'$%d$'%i, ms=15)
    model_force_list, model_displ_list = [], []
    for fname in os.listdir(csv_folder):
        if fname.startswith('calibration') and int(fname[11])<5:
            time, force, displ = np.loadtxt(csv_folder+fname, delimiter=','
                ).T
            model_force_list.append(np.interp(2.5, time, force)*1e3)
            model_displ_list.append(np.interp(2.5, time, displ)*1e3)
    axs.plot(model_displ_list, model_force_list, '--r')
    axs.plot(displ_coeff[0]+np.array(model_displ_list)*displ_coeff[1], 
             model_force_list, '-r')
    axs.set_xlabel('Static displ. (mm)')
    axs.set_ylabel('Static force (mN)')
    fig.tight_layout()
    fig.savefig('1.png')
    # %% Plot test traces
    fig, axs = plt.subplots(2, 1)
    for i, displ in enumerate(displ_trace_list):
        force = force_trace_list[i]
        axs[0].plot(displ, label='#%d'%use_trace[i])
        axs[1].plot(force, label='#%d'%use_trace[i])
    axs[0].legend(loc=4, ncol=2)
    axs[0].set_xlabel('Time (s)')
    axs[1].set_xlabel('Time (s)')
    axs[0].set_ylabel('Displacement (mm)')
    axs[1].set_ylabel('Force (mN)')
    fig.tight_layout()
    fig.savefig('2.png')
