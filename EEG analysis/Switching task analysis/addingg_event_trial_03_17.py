#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 23:54:34 2020

@author: risadwiratnasari
"""
''' 
erasing missing trial 
the plans
-> run the behavioral  file and run the code till conditions categorization
-> compare each condition in eeg data with each condition in behavioral
-> if Rt behavioral 0 then changes trial in eeg data

'''

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import  statistics 
#import stdev 
from function_library_labelling_tr import acc_check,pure_cond,tr_2_tr_cond,detect_outlier,detect_missing_trial
from flg import specifyTrials_eeg,eeg_pure_cond,eeg_tr_2_tr_cond

''''👩‍💻👩‍💻👩‍💻👩‍💻👩‍💻👩‍💻'''
data_dir_behaviour =  '/Users/new/Documents/new_office_folder/Documents/Bids_format_EEG/RAW/data_exp_eeg_bv/'

def analyseBehaviourParticipant(mat_data):    
    #data_dir_mat = '/Users/new/Documents/new_office_folder/download_folder_oldAir/behaviour_data/' 
    mat = scipy.io.loadmat( mat_data)
    tr = mat.get('trial_list')
    data_trial_list = tr.tolist()
    trial_list = []
    for data_ in data_trial_list:
        temp_item=[]
        for dt in data_:
            data_single = dt.tolist()[0][0]
            temp_item.append(data_single)
        trial_list.append(temp_item)
        
    pure_trial = trial_list[:90]
    for index,item in enumerate(trial_list[:90]):
        print("Line {},  Value : {}".format(index, item))
        
    mix_trial = trial_list[90:]
    for index,item in enumerate(trial_list[90:]):
        print("Line {},  Value : {}".format(index, item))
    
    df0= pd.DataFrame(pure_trial, columns=['conditions', 'stim_col_cond', 'Corr_resp', 'Resp', 'RT'])
    df =pd.DataFrame(mix_trial, columns=['conditions', 'stim_col_cond', 'Corr_resp', 'Resp', 'RT'])
      
       #for pure trial
    df0['block_num'] = 0
    df0['trial_num'] = 0
    df0['pure_cond'] = None
    df0['resp_acc'] = None
    
    block_num = 0
    wi_tr_num = 0
    # Loop around to generat the individual analysis matrix
    for trp in df0.iterrows():
        # tiral and evaluation set up
        trial_num = trp[0]
        df0['trial_num'][trial_num] = wi_tr_num
        
        # trial to trial contions assignment
        df0 = pure_cond(df0, trial_num, wi_tr_num)
    
        # Accuracy assignment
        df0 = acc_check(df0, trial_num)
    
        # Trial and block number assignment
        if (trial_num+1)%45 == 0: 
            wi_tr_num = 0
            df0['block_num'][trial_num] = block_num
            block_num += 1
        else:
            wi_tr_num += 1
            df0['block_num'][trial_num] = block_num

    #for mix trial
    df['block_num'] = 0
    df['trial_num'] = 0
    df['tr_2_tr_cond'] = None
    df['resp_acc'] = None
    
    block_num = 0
    wi_tr_num = 0
    # Loop around to generat the individual analysis matrix
    for trm in df.iterrows():
        # tiral and evaluation set up
        trial_num = trm[0]
        df['trial_num'][trial_num] = wi_tr_num
        
        # trial to trial contions assignment
        df = tr_2_tr_cond(df, trial_num, wi_tr_num)
        
        # Accuracy assignment
        df = acc_check(df, trial_num)
        
        # Trial and block number assignment
        if (trial_num+1)%45 == 0: 
            wi_tr_num = 0
            df['block_num'][trial_num] = block_num
            block_num += 1
        else:
            wi_tr_num += 1
            df['block_num'][trial_num] = block_num
    
    #make it same for title label
    df0= df0.rename(columns={'pure_cond':'label'})
    df= df.rename(columns={'tr_2_tr_cond':'label'})
    trial_list = pd.concat([df0,df], ignore_index=True)
    
    #calculate outlier
    Q1 = np.quantile(trial_list['RT'],0.25)
    Q2 = np.quantile(trial_list['RT'],0.50)
    Q3 = np.quantile(trial_list['RT'],0.75)
    IQR = Q3 - Q1
    #look for std
    stdev= statistics.stdev(trial_list['RT'])  
    mean_data= np.mean(trial_list['RT']) 
    low_outlier= mean_data - (2*stdev)
    low_outlier
    high_outlier= mean_data + (2*stdev)
    high_outlier 
    
    #detect outlier
    trial_list['block_num'] = 0
    trial_list['trial_num'] = 0
    trial_list['detect_outlier']= None
    
    block_num = 0
    wi_tr_num = 0
    
    for tr in trial_list.iterrows():
        trial_num = tr[0]
        trial_list['trial_num'][trial_num] = wi_tr_num
        trial_list = detect_outlier(trial_list, trial_num, wi_tr_num)
        
           # Trial and block number assignment
        if (trial_num+1)%45 == 0: 
            wi_tr_num = 0
            df['block_num'][trial_num] = block_num
            block_num += 1
        else:
            wi_tr_num += 1
            df['block_num'][trial_num] = block_num
            
    #label missing trial
    trial_list['block_num'] = 0
    trial_list['trial_num'] = 0
    trial_list['detect_missing_trial']= None
    
    block_num = 0
    wi_tr_num = 0
    for tr in trial_list.iterrows():
        trial_num = tr[0]
        trial_list['trial_num'][trial_num] = wi_tr_num
        trial_list = detect_missing_trial(trial_list, trial_num, wi_tr_num)
         
        # Trial and block number assignment
        if (trial_num+1)%45 == 0: 
            wi_tr_num = 0
            df['block_num'][trial_num] = block_num
            block_num += 1
        else:
            wi_tr_num += 1
            df['block_num'][trial_num] = block_num  
          
    '''
    pure_p   ----> 1
    Pure_s   -----> 2
    stay_pp  ------> 3
    stay_ss  ------> 4
    switch_ps ------> 5
    switch_sp ------> 6
    
    note unusage label number 
    999= ft
    99 = missing trial
    9 = outlier
    '''
    
    #number labelling
    label_number_dict = {"pure_p":1, "pure_s":2, "stay_pp":3, "stay_ss":4, "switch_ps":5, "switch_sp":6, "ft":999, }
    
    label_number_array = []
    
    for index, row in trial_list.iterrows():
        label_number_array.append(label_number_dict[row["label"]])
    
    trial_list['label_number_cond']= label_number_array 
    
    #labelling for missing trial 
    
    for index in trial_list.index:
        if trial_list.loc[index,'RT']== 0:
            trial_list.loc[index,'label_number_cond'] = 99
            
            
    #labelling for outlier        
    for index in trial_list.index:
        if trial_list.loc[index,'detect_outlier']== "outlier":
            trial_list.loc[index,'label_number_cond'] = 9
    
    #change condition label for incorrect trial

    for index in trial_list.index:
        if (trial_list.loc[index,'label_number_cond']==2) & (trial_list.loc[index,'resp_acc']== 0):
            trial_list.loc[index,'label_number_cond'] = 90
        elif(trial_list.loc[index,'label_number_cond']==4) & (trial_list.loc[index,'resp_acc']== 0):
            trial_list.loc[index,'label_number_cond'] = 90       
        elif (trial_list.loc[index,"label_number_cond"] == 5) & (trial_list.loc[index,'resp_acc']== 0):
            trial_list.loc[index,'label_number_cond'] = 90
    
    
    #decrease 1 trial for every bloc so decrease 9 trial all
    
    participant_data = {'trial_list':trial_list}
    
    return participant_data



#first_participant = analyseBehaviourParticipant('Color_task_manual_S01_1.mat.mat')
#trial_list = first_participant['trial_list']
#trial_list01.to_csv("a.csv")


#concatenate_behaviour = np.concatenate((sorted_rt_mix_stay_preference01,sorted_rt_mix_stay_similarity01,sorted_rt_mix_switch_preference_similarity01,sorted_rt_mix_switch_similarity_preference01))
#171 =180-9

'''to do -> now change the eeg data with the new behavioural data formula use flg function to see formula 
 and adjust with pure condition and outlier condition print eeg condition and erase the missing trial in eeg data
 
 PR lagi  how to ambil yang correct trial only , missing trial the key in the label label '''


# %pylab
# %matplotlib inline
# %matplotlib qt5
import os
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs)
from mne.stats import permutation_cluster_test

''''🧠🧠🧠🧠🧠🧠'''
data_dir_eeg = '/Users/new/Documents/new_office_folder/eeg/001/'

def analyseEEGParticipant(fif):
    #Read raw data
    raw = mne.io.read_raw_fif(fif, preload=True)
    # Add triggers to raw data
    # Identify annotated triggers
    events, event_ids = mne.events_from_annotations(raw)
    
    trial_list_eeg = events
       
    pure_trial = trial_list_eeg[:90]
    for index,item in enumerate(trial_list_eeg[:90]):
        print("Line {},  Value : {}".format(index, item))
        
    mix_trial = trial_list_eeg[90:]
    for index,item in enumerate(trial_list_eeg[90:]):
        print("Line {},  Value : {}".format(index, item))
    
    df0= pd.DataFrame(pure_trial, columns=['events', 'zero', 'conditions'])
    df =pd.DataFrame(mix_trial, columns=['events', 'zero', 'conditions'])
    
    trial_list_eeg = pd.concat([df0,df], ignore_index=True)    
    
    #make variable for events
    #make the path to bids format for bids😾
    
    mat_path =  '/Users/new/Documents/new_office_folder/Documents/Bids_format_EEG/switching_task/rawdata/sub-004/behavioral_data/'
    analyze_behaviour = analyseBehaviourParticipant(mat_path+'sub-004_color_task_manual.mat')
    trial_list= analyze_behaviour['trial_list']

    ori_and_modif_conditions = trial_list_eeg.join(trial_list["label_number_cond"])
    modified_conditions =    ori_and_modif_conditions.drop('conditions', 1)
    modified_mix_trial = modified_conditions.iloc[90:].to_numpy()
    
    new_event_idi = dict({'stay_pp':3, 'stay_ss':4, 'switch_ps':5, 'switch_sp':6,'ft':999,'Ict':90})
        
    fig = mne.viz.plot_events(modified_mix_trial,first_samp=raw.first_samp,
                              sfreq=raw.info['sfreq'],event_id=new_event_idi)
    
    #additional
    modified_all_trials = modified_conditions.iloc[:].to_numpy()
    
    new_event_idbi = dict({'pure_pref':1,'pure_sim':2,'stay_pp':3, 'stay_ss':4, 'switch_ps':5, 'switch_sp':6,'ft':999,'Ict':90})   
     
    fig = mne.viz.plot_events(modified_all_trials,first_samp=raw.first_samp,
                              sfreq=raw.info['sfreq'],event_id=new_event_idbi)

    modified_all_trials = {'modified_all_trials':modified_all_trials, 'raw':raw}
    
    return modified_all_trials


                                                    


#eeg_trial_list = analyseEEGParticipant['modified_all_trials']


''' to do 22/12 move this to bids format, how can the bids function can contain this function and everyone file
     key : sub to every file, and apply this function and save plot and array events '''
     

'''🐚🐚🐚🐚🐚🐚🐚🐚🐚💐💐💐💐💐💐💐🦋🦋🦋🦋🦋🦋🦋🦋🍭🍭🍭🍭🍭🍭🍭🍭'''

         



