# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 14:10:46 2015

@author: Administrator
"""

from convertdata import Test


class OldMouse(Test):

    def __init__(self, fs=1e3):
        self.fs = fs
        self.force_trace_list, self.displ_trace_list, contact_displ_skin = \
            self.load_data(duration=5.)
        contact_displ_noskin = self.load_data(duration=1.)[2]
        self.skin_thickness = contact_displ_noskin - contact_displ_skin        
        pass

    def load_formula(self):
        force_formula = '7.5079*x+-0.365935468548'
        displ_formula = '(x-1.)*10./4.'
        return force_formula, displ_formula

    def get_mat_fname(self):
        fname = './YoshiExperiment/2014-12-23-800-UVA.mat'
        return fname

if __name__ == '__main__':
    # Instantiation
    oldMouse = OldMouse()
    youngMouse = Test(test_id=2, block_id=1)