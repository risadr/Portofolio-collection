#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 21:44:14 2022

@author: new
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics 
import sys

#import stdev 
def detect_outlier(mat_data):
    '''
    Objective of detect_outlier function: This function will show the plot visualization  
    of curve distribution  to see the comparison between data with the outlier
    and the data without outlier.
    The data that has been cleaned from the outlier will be preserved 
    and can be collected to be processed to the next stage of analysis
    
    one required parameter is needded (mat_file) contains the data directory
    that has been defined + the file name with .mat extention  --> 
    
    data_dir = /......./
    mat_file= opj(data_dir + xxxxxxxx.mat)
    detect_outlier(mat_file)
    
    The requirement for the data in this analysis :Please to make sure the data
    has been cleaned from missing trials noticed by
    Reaction time == 0
    
    The function run successfully if it shows the output of graph and the data frame
    also notification as below
    
    "Outlier labelling success!
    categorization is done"
    
    ---otherwise---
    
    "Remove missing the  missing trials first
    ERROR! missing trial detected
    categorization is done "
    
    if you see the error massage as above, it means the data contained missing trial,
    please to clean it first and try again!
     
    '''
    
    #data_dir= '/Users/new/Documents/new_office_folder/Documents/Thesis/behavioral_exp_data/'
    #mat_data = data_dir + 'sub-002_color_task.mat'
    mat = scipy.io.loadmat(mat_data)
    tr = mat.get('trial_list')
    #data_trial_list = tr.tolist()
    trial_list = []
    for data_ in tr:
        temp_item=[]
        for dt in data_:
            data_single = dt.tolist()[0][0]
            temp_item.append(data_single)
        trial_list.append(temp_item)
    
    trial_list =pd.DataFrame(trial_list, columns=['conditions', 'stim_col_cond', 'Corr_resp', 'Resp', 'RT'])
            
    #detect outlier in the mix trial
    #look for std
    stdev= statistics.stdev(trial_list['RT'])   
    mean_data= np.mean(trial_list['RT']) 
    low_outlier= mean_data - (2*stdev)
    high_outlier= mean_data + (2*stdev)
    
    trial_list['detect_outlier']= None
    
    try:
        for index in trial_list.index:  
        
            if(trial_list.loc[index,'RT'] > low_outlier) & (trial_list.loc[index,'RT']< high_outlier):
               trial_list.loc[index,'detect_outlier'] = 'preserved'
               # check += 1
            elif trial_list.loc[index,'RT'] == 0:
                raise Exception('ERROR! missing trial detected')
            else:
                trial_list.loc[index,'detect_outlier'] = 'outlier'
                trial_list['detect_outlier'].unique()
        print('Outlier labelling success!')
        data_without_outlier = trial_list[trial_list.detect_outlier != 'outlier']
             
        bins = np.linspace(0.0, 2.5, 100)
        plt.hist(trial_list['RT'],bins,color='orange', alpha=0.5, label='with outlier')
        plt.hist(data_without_outlier['RT'], bins,color='darkblue', alpha=0.5,label='non-outlier')
        plt.legend(loc='upper right')
        plt.show()
    except Exception as e:
        print(e)
        print('solution: Remove the  missing trials first')
    finally: 
        print('categorization is done')
    
    return data_without_outlier
    
            
  
      

