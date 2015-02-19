import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from strexp import relax_func_prony

DT = 1e-2

class FeRun:
    
    def __init__(self, var_type, var_value, friction):
        base_path = 'X:/WorkFolder/AbaqusFolder/Viscoelasticity/csvFiles/'
        if friction == 'frictionless':
            cof = 0.
        elif friction == 'penalty':
            cof = .3
        elif friction == 'rough':
            cof = np.inf
        if var_type == 'thickness':
            matdata = loadmat('../abqData.mat')['abqTable']
            self.thickness = var_value * 1e-6
            fname = 'ForceDisp%d%s.csv' % (var_value, friction)
            params = matdata[(matdata[:, 0] == var_value) * (
                matdata[:, 1] == cof)]
            assert params.shape == (1, 34), 'More than one match found.'
            self.visco_params = params.ravel()[10:14]
        elif var_type == 'stretch':
            matdata = loadmat('../abqDataSl%s.mat'%friction)['abqSlTable1t']
            self.thickness = 400. * 1e-6
            fname = 'ForceDispsl0_%d%s.csv' % (int(10*var_value), friction)
            params = matdata[np.isclose(matdata[:, -1], var_value)]
            self.visco_params = params.ravel()[:4]
        time, force, displ = np.loadtxt(base_path+fname, delimiter=',').T
        self.time = np.arange(0, time[-1], DT)
        self.force = np.interp(self.time, time, force)
        self.displ = np.interp(self.time, time, displ)
        self.relax_func = relax_func_prony(self.time, *self.visco_params)
        return
    

def hyper_boxplot(fig=None, axs=None):
    # Load all data
    frictionless = loadmat('../abqDataSlFrictionless.mat')
    penalty = loadmat('../abqDataSlPenalty.mat')
    rough = loadmat('../abqDataSlRough.mat')
    if fig is None and axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(3.27, 1.8))
    labels = [r'$\mu_f = 0$', r'$\mu_f = 0.3$', r'$\mu_f = \infty$']
    axs[0].boxplot([
        frictionless['abqSlTable1t_float'][:, -4]/1e3,
        penalty['abqSlTable1t_float'][:, -4]/1e3,
        rough['abqSlTable1t_float'][:, -4]/1e3], labels=labels)
    axs[0].set_ylabel(r'$\mu$ (kPa)')
    axs[1].boxplot([
        frictionless['abqSlTable1t'][:, -3],
        penalty['abqSlTable1t'][:, -3],
        rough['abqSlTable1t'][:, -3]], labels=labels)
    axs[1].set_ylabel(r'$\alpha$')        
    # Adding panel labels
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.3, 1.05, chr(65+axes_id), transform=axes.transAxes,
            fontsize=12, fontweight='bold', va='top')    
    fig.tight_layout()
    fig.savefig('./compare_hyper.png', dpi=300)
    return fig, axs


def extract_visco_stretch(friction_list, stretch_list, fmt_list):
    feRuns = np.empty((len(friction_list), len(stretch_list)), dtype=object)
    fig, axs = plt.subplots(3, 1, figsize=(3.27, 5))
    for i, friction in enumerate(friction_list):
        if friction == 'frictionless':
            label = r'FEA ($\mu_f$ = 0)'
        elif friction == 'penalty':
            label = r'FEA ($\mu_f$ = 0.3)'
        elif friction == 'rough':
            label = r'FEA ($\mu_f$ = $\infty$)'
#        for j, thickness in enumerate(thickness_list):
        for j, stretch in enumerate(stretch_list):
#            feRun = FeRun('thickness', thickness, friction)
            feRun = FeRun('stretch', stretch, friction)
            feRuns[i, j] = feRun
            fmt = fmt_list[i]
            axs[0].plot(feRun.time, -feRun.displ*1e3, fmt, 
                label=label)
            axs[1].plot(feRun.time, -feRun.force, fmt, 
                label=label)
            axs[2].plot(feRun.time, feRun.relax_func, fmt, 
                label=label)
    axs[0].set_ylim(0, .25)
    axs[0].set_xlim(0, 5)
    axs[1].set_xlim(0, 5)
    axs[2].set_xlim(0, 5)
    axs[0].set_xlabel('Time (s)')
    axs[1].set_xlabel('Time (s)')
    axs[2].set_xlabel('Time (s)')
    axs[0].set_ylabel('Displacement (mm)')
    axs[1].set_ylabel('Force (N)')
    axs[2].set_ylabel('Relaxation function G(t)')
    # Add legends
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles[::len(stretch_list)], labels[::len(stretch_list)], 
        loc=4)
    axs[1].legend(handles[::len(stretch_list)], labels[::len(stretch_list)])
    axs[2].legend(handles[::len(stretch_list)], labels[::len(stretch_list)])
    # Adding panel labels and save
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.15, 1.125, chr(65+axes_id), transform=axes.transAxes,
            fontsize=12, fontweight='bold', va='top')    
    fig.tight_layout()
    fig.savefig('./extract_visco_stretch.png', dpi=300)
    fig.savefig('./extract_visco_stretch.tif', dpi=300)
    return fig, axs


def extract_visco_thickness(friction_list, thickness_list, fmt_list):
    feRuns = np.empty((len(friction_list), len(thickness_list)), dtype=object)
    fig, axs = plt.subplots(3, 1, figsize=(3.27, 5))
    for i, friction in enumerate(friction_list):
        if friction == 'frictionless':
            label = r'FEA ($\mu_f$ = 0)'
        elif friction == 'penalty':
            label = r'FEA ($\mu_f$ = 0.3)'
        elif friction == 'rough':
            label = r'FEA ($\mu_f$ = $\infty$)'
        for j, thickness in enumerate(thickness_list):
            feRun = FeRun('thickness', thickness, friction)
            feRuns[i, j] = feRun
            fmt = fmt_list[i]
            axs[0].plot(feRun.time, -feRun.displ*1e3, fmt, 
                label=label)
            axs[1].plot(feRun.time, -feRun.force, fmt, 
                label=label)
            axs[2].plot(feRun.time, feRun.relax_func, fmt, 
                label=label)
    axs[0].set_ylim(0, .25)
    axs[0].set_xlim(0, 5)
    axs[1].set_xlim(0, 5)
    axs[2].set_xlim(0, 5)
    axs[0].set_xlabel('Time (s)')
    axs[1].set_xlabel('Time (s)')
    axs[2].set_xlabel('Time (s)')
    axs[0].set_ylabel('Displacement (mm)')
    axs[1].set_ylabel('Force (N)')
    axs[2].set_ylabel('Relaxation function G(t)')
    # Add legends
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles[::len(thickness_list)], labels[::len(thickness_list)], 
        loc=4)
    axs[1].legend(handles[::len(thickness_list)], labels[::len(thickness_list)])
    axs[2].legend(handles[::len(thickness_list)], labels[::len(thickness_list)])
    # Adding panel labels and save
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.15, 1.125, chr(65+axes_id), transform=axes.transAxes,
            fontsize=12, fontweight='bold', va='top')    
    fig.tight_layout()
    fig.savefig('./extract_visco_thickness.png', dpi=300)
    fig.savefig('./extract_visco_thickness.tif', dpi=300)
    return fig, axs


if __name__ == '__main__':
    thickness_list_full = np.arange(200, 800, 100)
    thickness_list = np.array([400])
    stretch_list_full = np.arange(0.2, 0.9, 0.1)
    stretch_list = np.array([.5, .6, .7])
    fmt_list_marker = ['-ok', ':xk', '--^k']
    fmt_list_full = ['-k', ':k', '--k']
    fmt_list = ['-k', '--k']
    friction_list_full = ['frictionless', 'penalty', 'rough']
    friction_list = ['frictionless', 'rough']
    # Plot stretch
    extract_visco_stretch(friction_list, stretch_list, fmt_list)
    extract_visco_thickness(friction_list, thickness_list, fmt_list)
    
    