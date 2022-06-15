#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 20:05:53 2022

@author: new
"""
import sys
sys.path.append('/Users/new/Documents/rakamin/function_error_handling/HW/')
from outlier_detector import detect_outlier
from os.path import join as opj


data_dir= '/Users/new/Documents/new_office_folder/Documents/Thesis/behavioral_exp_data/'
mat_data = opj(data_dir+'sub-004_color_task.mat')
#call the docstring 
print (detect_outlier.__doc__)
#call the function
detect_outlier(mat_data)

