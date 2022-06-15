#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 15:03:21 2020

@author: risadwiratnasari
"""


"""
pre-processing steps for dw

detrending
re-referencing
ICA removal
change trigger
low pass filtering
epoching
artifact-rejection +/- 80mV
baseline correction
"""

#importing the modules
import os
from os.path import join as opj
import pathlib 
import numpy as np
import scipy
from scipy import signal
import mne 
from os.path import join 
from scipy import io 
# from sys import path
# from pathlib import Path
# from os.path import dirname as dir
# participant = Path()
# print(participant)
from statistics import stdev 
from addingg_event_trial_03_17 import analyseBehaviourParticipant
#from addingg_event_trial import analyseEEGParticipant
from function_library_labelling_tr import (acc_check,pure_cond,tr_2_tr_cond,detect_outlier,detect_missing_trial)
import pandas as pd
from mne.preprocessing import(ICA, create_eog_epochs, corrmap)
%matplotlib qt5
import matplotlib.pyplot as plt
from autoreject import AutoReject, compute_thresholds  # noqa


#create folder for preprocessing result
if not os.path.exists(os.path.join(data_dir,'derivatives')):
    os.mkdir(os.path.join(data_dir,'derivatives'))
if not os.path.exists(os.path.join(data_dir,'derivatives','preprocessing')):
    os.mkdir(os.path.join(data_dir,'derivatives','preprocessing'))
data_dir= '/Volumes/HD710 PRO/work_folder/Bids_format_EEG//switching_task/'
  
df= pd.read_csv(os.path.join('/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/participants.tsv'),\
                                 index_col=0,delimiter='\t')
subjects=df.index.values
n_subs=len(subjects)

for sub in subjects:
    if not os.path.exists(os.path.join(data_dir,'derivatives','preprocessing',sub)):
        os.mkdir(os.path.join(data_dir,'derivatives','preprocessing',sub))
    if not os.path.exists(os.path.join(data_dir,'derivatives','preprocessing',sub,'raw')):
        os.mkdir(os.path.join(data_dir,'derivatives','preprocessing',sub,'raw'))
       
#specify the bad channel 
def raw(sub):
    print('Processing',sub)
    
    #setting the file paths
    data_dir = '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/'
    vhdr_file= os.path.join(data_dir,'rawdata',sub,'eeg',sub+'_task-colourswitching_eeg.vhdr')
    vmrk_file=os.path.join(data_dir,'rawdata',sub,'eeg',sub+'_task-colourswitching_eeg.vmrk')
    eeg_file=os.path.join(data_dir,'rawdata',sub,'eeg',sub+'_task-colourswitching_eeg.eeg')
    #read raw
    raw=mne.io.read_raw_brainvision(vhdr_file, eog=('LEOG', 'REOG', 'EMG1'),preload=True)
    raw.info['misc'] = ['ECG']
    raw_path='/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/raw/'
    raw.save(raw_path + sub +'_color-switching-raw.fif', overwrite=True) 

#run for all subjects
for sub in subjects:
    raw(sub)  

#creating a detrend folder in the preprocessing folder
for sub in subjects:    
    if not os.path.exists(os.path.join(data_dir,'derivatives','preprocessing',sub,'interpolated_channels')):
        os.mkdir(os.path.join(data_dir,'derivatives','preprocessing',sub,'interpolated_channels'))


raw=[]
for sub in subjects:
    print('Processing',sub)
    raw_path ='/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/raw/'
    raw_file = opj(raw_path,sub+'_color-switching-raw.fif')
    raw_data= mne.io.read_raw_fif(raw_file, preload=True)
    raw.append(raw_data)
    
#def interpolate(sub):
raw[0].info['bads'] = ['Oz','FC1','F7','AF7','AF4','Fz','P6','PO3','F5','P4','P8',
                       'FC3','FC1','FC4','F3','F6','CP2','P1','FT7','TP8','TP7']
#II
raw[1].info['bads'] = ['F6','FC3','CP2','F5','Oz','F1','Fz','P8','F7','P4','F3']
#III
raw[2].info['bads'] = ['Fp1','P4','F5','F4','Fp2','AF8','F7','AF7','FC1','TP8','CP6','P2']
#IV
raw[3].info['bads'] = ['Fp2','F7','F8','F5','PO4','Pz','Oz','CP6','AF4','FC4','CP3','F6','FT8','PO7','CPz']
#V
raw[4].info['bads'] = ['Fp2','O1','Pz','CP1','C1','FC4','F5','C2','P2','AF3','CP4','PO4',
                       'F5','F6','C5','AF8','TP7','PO7']
#VI
raw[5].info['bads'] = ['P7','Fz','Cz','Pz','C2','Oz','CP1','FC5','FC6','CP5','CP6','C1','AF4','FC3',
                       'F5','TP7','PO7']
#VII
raw[6].info['bads'] = ['P3','F8','FC2','PO8','CP5','FC3','CP5','CP4','F5'] 
#VIII
raw[7].info['bads'] = ['O1','Oz','FT7','F5','TP7','PO3','C6']    
#IX
raw[8].info['bads'] = ['Oz','PO3','P5','P7']
#X
raw[9].info['bads'] = ['F3','F4','C3','P4','O2','P7','P8','Oz','CP2','CP5','F2',
                       'AF4','P2','C1','FC3','C5','C6','P5','P6','TP8','PO8','CPz']

raw[10].info['bads'] = ['Fz','P2','AF3']
raw[11].info['bads'] = ['Fp1','Fp2','FC2','C4','Fz','FC5','CP5',
                        'AF7','AF8''',,'FC4','C5','FT7','TP7']
raw[12].info['bads'] = ['Fz','CPz','Oz','AF4','FC6']
raw[13].info['bads'] = ['C6','FT8','TP7','Fz','P2']
raw[14].info['bads'] = ['PO4','POz','Fpz','Fz','AF7','FT7','AF4']
raw[15].info['bads'] = ['F3','C3','C4','O2','F2','FC1','FC2','CP1','FC5','FC6',
                        'CP6','C1','AF3','FC3','CP4','PO4','F5','P6','TP8','PO8','FT7']
raw[16].info['bads'] = ['Fp1','F4','P3','O2','F2','AF3','AF4','PO4','P6',
                        'P6','TP7','PO7','Fpz','F2','F5']
raw[17].info['bads'] = ['Fp2','F7','F8','F5','AF7','AF8','TP7']
raw[18].info['bads'] = ['Fp1','Fp2','F3','F4','C3','O2','F8','F2','CP1',
                        'F1','F2','P1','P2','AF3','AF4','FC4','CP4',
                        'PO3','F6','F5','P6','FT8','TP7','PO8','F2','FT8']
raw[19].info['bads'] = ['C4','P4','O2','F7','F8','F2','CP1','CP5','F1','AF3',
                        'AF4','F6','C6','FT8','PO8']


raw_with_bads= [raw[0],raw[1],raw[2],raw[3],raw[4],raw[5],raw[6],raw[7],raw[8],
                raw[9],raw[10],raw[11],raw[12],raw[13],raw[14]]
#,raw[7],raw[8],raw[9],raw[10],raw[11],raw[12],raw[13],raw[14]] 
raw_recheck_bads= raw_with_bads.copy()
#plot channel mapping before interpolate
raw_hide_bads = mne.channels.make_eeg_layout(raw_recheck_bads[1].info)
raw_hide_bads.plot() # bad channels will not be included

interpolated_raw= []
for i in range(len(raw_with_bads)):
    f = raw_with_bads[i].interpolate_bads(reset_bads=True, mode='accurate')
    interpolated_raw.append(f)

#plot channel mapping after interpolate
raw_repaired_bads = mne.channels.make_eeg_layout(interpolated_raw[1].info)
raw_repaired_bads.plot()
#still manual

raw_path='/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/derivatives/preprocessing/sub-003/interpolated_channels/'
raw3.save(raw_path+'sub-003_interpolated_color-switching-raw.fif', overwrite=True)

#save evry elemen inside list
# i=0   
# for sub in subjects:
#     print('Processing',sub)
#     raw_path ='/Users/new/Documents/new_office_folder/Documents/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/interpolated_channels/'
#     for i in iter(raw):
#         raw[i].save(raw_path + sub +'n_interpolated_color-switching-raw.fif') 
#         i += 1

# outF = open("myOutFile.fif", "w") 
# for line in raw:
#   # write line to output file
#   outF.write(line)
#   outF.write("\n")
# outF.close()

'''

def interpolate(sub): 
    for i in range(len(interpolated_raw)):
        raw_path='/Users/new/Documents/new_office_folder/Documents/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/interpolated_channels/'
        interpolated_raw[i+1].save(raw_path+sub +'interpolated_color-switching-raw.fif', overwrite=True)
help!!! how to loop save in different folder from a list, how list can move automaticly inside the list
'''  
        

#creating a detrend folder in the preprocessing folder
for sub in subjects:    
    if not os.path.exists(os.path.join(data_dir,'derivatives','preprocessing',sub,'detrend')):
        os.mkdir(os.path.join(data_dir,'derivatives','preprocessing',sub,'detrend'))

# defining function for detrending
def detrend(sub):
    print('Processing',sub)
    #getting all the files
    
    #setting the file paths
    nobads_raw_path='/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/derivatives/preprocessing/'\
         +sub+'/interpolated_channels//'
    nobads_raw_file=opj(nobads_raw_path,sub+'_interpolated_color-switching-raw.fif')
     
    raw=mne.io.read_raw_fif(nobads_raw_file, preload=True)
    raw_detrend= raw.copy()
    #extracting the EEG data
    x = raw_detrend.get_data()
   #detrending the signal
    mn = np.mean(x, axis=-1, keepdims=True)
    x = signal.detrend(x, axis=- 1, type='linear', bp=0, overwrite_data=False)
    x = x+mn
    #adding data back to raw object
    raw_detrend._data = x
    
    ##plt.plot(x)
    #saving the detrended file 
    
    #detrended=mne.io.RawArray(x,info) 
    detrend_path='/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/detrend/'
    raw_detrend.save( detrend_path + sub +'_color-switching_detrended-raw.fif',overwrite=True)
        
#run for all subjects
for sub in subjects:
    detrend(sub)  
            
#Filter 0.1 - 30.
for sub in subjects:
    if not os.path.exists(os.path.join(data_dir,'derivatives','preprocessing',sub,'Filter')):
        os.mkdir(os.path.join(data_dir,'derivatives','preprocessing',sub,'Filter'))
         
def Filter(sub):
    print('Processing',sub)
     #setting the path for the required(detrended) files
    detrend_path='/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/derivatives/preprocessing/'\
        +sub+'/detrend//'
    detrend_path=opj(detrend_path,sub+'_color-switching_detrended-raw.fif')
    raw_detrend=mne.io.read_raw_fif(detrend_path, preload=True)
    raw_filter= raw_detrend.copy()
    #filter the raw
    raw_filter.filter(0.1, 30., fir_design='firwin')     
    raw_filter.plot_psd(fmax=50)
    fil_path = '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/Filter/'
    raw_filter.save(fil_path + sub + '_color-switching_filtered-raw.fif',overwrite=True)
     
for sub in subjects:
    Filter(sub)

#make directory for remove slow drift 
for sub in subjects:
    if not os.path.exists(os.path.join(data_dir,'derivatives','preprocessing',sub,'remove_slow_drift')):
        os.mkdir(os.path.join(data_dir,'derivatives','preprocessing',sub,'remove_slow_drift'))

#filter slow-drift

def remove_slow_drift(sub):
    print('Processing',sub)
     #setting the path for the required(detrended) files
    filter_path ='/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/Filter/'
    filter_file= opj(filter_path,sub+'_color-switching_filtered-raw.fif')
    #read the raw
    raw_filtered= mne.io.read_raw_fif(filter_file, preload=True)
    #filter the raw
    raw_filtered.filter(l_freq=1., h_freq=None)
    raw_filtered.plot_psd(fmax=50)
    fil_path = '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/remove_slow_drift/'

    raw_filtered.save(fil_path + sub + '_color-switching_slow_drift_removed-raw.fif',overwrite=True)

for sub in subjects:
    remove_slow_drift(sub)



#change the trigger and epoching 
data_dir= '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/'
for sub in subjects:
    if not os.path.exists(os.path.join(data_dir,'derivatives','preprocessing',sub,'change_trigger')):
        os.mkdir(os.path.join(data_dir,'derivatives','preprocessing',sub,'change_trigger'))

#need to set to bid format accordingly
def change_trigger(sub):
        
    print('Processing',sub)
    #setting the path for the required(rereferenced) files
    #mat_path = '/Users/new/Documents/new_office_folder/Documents/Bids_format_EEG/RAW/data_exp_eeg_bv/'
    mat_path= '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/rawdata/'+sub+'/behavioral_data/'
    eeg_path = '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/remove_slow_drift/'
    mat_file= opj(mat_path,sub+'_color_task_manual.mat')
    print(mat_file)
    eeg_file= opj(eeg_path,sub+'_color-switching_slow_drift_removed-raw.fif')
    print(eeg_file)
    #open script adding_event-trial
    analyze_behaviour = analyseBehaviourParticipant(mat_file)   
    #read eeg file
    raw = mne.io.read_raw_fif(eeg_file, preload=True)
    #read events
    events, event_ids = mne.events_from_annotations(raw)
    # making condition of trial type of eeg file
    
    trial_list_eeg = events       
    pure_trial = trial_list_eeg[:90]
    for index,item in enumerate(trial_list_eeg[:90]):
        print("Line {},  Value : {}".format(index, item))
        
    mix_trial = trial_list_eeg[90:]
    for index,item in enumerate(trial_list_eeg[90:]):
        print("Line {},  Value : {}".format(index, item))
    #creating the data frame    
    df0= pd.DataFrame(pure_trial, columns=['events', 'zero', 'conditions'])
    df =pd.DataFrame(mix_trial, columns=['events', 'zero', 'conditions'])
    #combining frame 
    trial_list_eeg = pd.concat([df0,df], ignore_index=True)    
    trial_list = analyze_behaviour['trial_list']     
    # change the trigger
    
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

    #modified_all_trials = {'modified_all_trials':modified_all_trials, 'raw':raw}

    #converting to raw via annotations_from_events
    
    mapping = dict({1: 'pure_pref', 2: 'pure_sim', 3: 'stay_pp',
           4: 'stay_ss', 5: 'switch_ps', 6: 'switch_sp',9:"outlier",99:'missing',999:"ft",90:"Ict"})
    
    annot_from_events = mne.annotations_from_events(modified_all_trials,first_samp=raw.first_samp,
                              sfreq=raw.info['sfreq'],event_desc=mapping)
    #create parameter to change raw    
    modified_all_trials=  modified_all_trials[~np.isnan(modified_all_trials).any(axis=1)]
    new_eventid = np.unique(modified_all_trials[:,2])
    onsets = modified_all_trials[:,0] /raw.info['sfreq']
    durations = np.zeros_like(onsets)  
    descriptions = [mapping[new_eventid] for new_eventid in modified_all_trials[:,2]]
    #applying given parameter
    annot_from_events = mne.Annotations(onset= onsets, duration= durations, description=descriptions, orig_time = raw.info['meas_date']) 
    new_raw= raw.copy()
    #set the change
    new_raw.set_annotations(annot_from_events)
    #checking the changes
    events, event_ids = mne.events_from_annotations(new_raw)
    
    #setting the event_ids via events_from_annotations
    events, _= mne.events_from_annotations(new_raw, event_id={'pure_pref': 1,'pure_sim':2,'stay_pp': 3,
            'stay_ss':4, 'switch_ps':5 ,'switch_sp':6,'ft':999,'Ict':90})
    #event dict epoching
    event_dict = {'pure_pref': 1,'pure_sim':2,'stay_pp': 3,
                'stay_ss':4, 'switch_ps':5 ,'switch_sp':6,"ft":999,"Ict":90}
    # picks_eeg = mne.pick_types(new_raw.info, meg=False, eeg=True, eog=True,
    #                        stim=False, exclude='bads')

    epochs= mne.Epochs(new_raw, events, tmin=-.1, tmax=1.,
            event_id={'pure_pref': 1,'pure_sim':2,'stay_pp': 3,
                      'stay_ss':4, 'switch_ps':5 ,'switch_sp':6} ,
                    preload=True, baseline=(-0.1, 0))
    #epochs.plot()
    
    change_trigger_path= '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/change_trigger/'
    epochs.save(change_trigger_path+sub+'_epochs_task-switching-epo.fif',overwrite=True)

       
#run for all subjects
for sub in subjects:
    change_trigger(sub)  

#re-referencing to mastoids for all subjects

#creating a reference folder in the preprocessing folder
for sub in subjects:
    if not os.path.exists(os.path.join(data_dir,'derivatives','preprocessing',sub,'re-reference')):
        os.mkdir(os.path.join(data_dir,'derivatives','preprocessing',sub,'re-reference'))

#defining a function for re-referencing
def rereference(sub):
     print('Processing',sub)
     #setting the path for the required(detrended) files
     change_trigger_path = '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/change_trigger/'
     change_trigger_file=opj(change_trigger_path,sub+'epochs_task-switching-epo.fif')
     epochs_ref = mne.read_epochs(change_trigger_file, preload=True)
     print(epochs_ref)
     
     epochs_ref= epochs_ref.set_eeg_reference(ref_channels=['A1', 'A2'])
     
     epochs_ref.plot()
     
     ref_path = '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/re-reference/'
     epochs_ref.save(ref_path + sub + '_color-switching_rereference_epo.fif',overwrite=True)
     
     #save_fn = sub+'_task-self_rereference_raw.fif'
     #if os.path.exists(ref_path) != True:
     #    os.mkdir(ref_path)
     # save_path= opj(ref_path, save_fn)
     #ref_raw.save(save_path)
     #return ref_raw

#run for all subjects
for sub in subjects:
    rereference(sub)  
            

#ICA process
# creating a ICA_fit folder in the preprocessing folder
for sub in subjects:
    if not os.path.exists(os.path.join(data_dir,'derivatives','preprocessing',sub,'ICA_analysis')):
        os.mkdir(os.path.join(data_dir,'derivatives','preprocessing',sub,'ICA_analysis'))

 
#defining a function to fit the ICA for all subjects
def fit_ICA(sub):
     print('Processing',sub)
     #setting the path for the required(rereferenced) files
     rereference_path ='/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/change_trigger/'
     rereference_file= opj(rereference_path,sub+'epochs_task-switching-epo.fif')
     epochs = mne.read_epochs(rereference_file, preload=True)
     
     ica=ICA(n_components=25,method="fastica",max_iter=200).fit(epochs)
     a= ica.plot_sources(epochs)
     b = ica.plot_components(inst=epochs)
     ica.exclude = [1,4] 
     epochs_ica = epochs.copy()
     epochs.plot()
     #ica.apply(epochs_ica)
     #before
     ab= epochs['stay_pp'].average()
     bb= epochs['switch_sp'].average()
     #after
     af= epochs_ica['stay_pp'].average()
     bf= epochs_ica['switch_sp'].average()
     
     mne.viz.plot_compare_evokeds(dict(stay_pp= ab , switch_sp= bb), picks='eeg', axes='topo')
     
     mne.viz.plot_compare_evokeds(dict(stay_pp= af , switch_sp= bf), picks='eeg', axes='topo')
     
     
     ICA_path = '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/ICA_analysis/'
     ica.save(ICA_path+sub+'_color-switching_fit-ica.fif')

#run for all subjects
for sub in subjects:
    fit_ICA(sub)   

#---------------------------------------

#removing eyes blinks in another scripts
#for 5,6,7


def re_order(sub):
    print('Processing',sub)
    #setting the path for the required(rereferenced) files
    rereference_path = '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/re-reference/'
    ref_epochs= opj(rereference_path,sub+'_color-switching_rereference_epo.fif')
    epochs_ref = mne.read_epochs(ref_epochs, preload=True)
    
    reorder_epochs= epochs_ref.copy()
    ch_names= ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1',
               'O2','F7','F8','LEOG','REOG','P7','P8','Fz','Cz',
               'Pz','Oz','FC1','FC2','CP1','CP2','FC5','FC6','CP5',
               'CP6','A1','A2','POz','ECG','F1','F2','C1','C2','P1',
               'P2','AF3','AF4','FC3','FC4','CP3','CP4','PO3','PO4',
               'F5','F6','C5','C6','P5','P6','AF7','AF8','FT7','FT8',
               'TP7','TP8','PO7','PO8','Fpz','CPz','EMG1']
    reorder_epochs.reorder_channels(ch_names)
    ref_path = '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/re-reference/'
    reorder_epochs.save(ref_path + sub + '_color-switching_rereference_epo.fif',overwrite=True)

#run for all subjects
subjects= ['sub-005','sub-006','sub-007']
np.array(subjects)
for sub in subjects:
    re_order(sub)     
     
def average_epochs(sub):
    print('Processing',sub)
     #setting the path for the required(detrended) files
    rejected_artifact_path ='/Users/new/Documents/new_office_folder/Documents/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/remove_artifact/'
    rejected_artifact_file= opj(rejected_artifact_path,sub+'Artifact_rejected-epo.fif')
    #read the raw
    epochs= mne.read_epochs(rejected_artifact_file, preload=True)
    #filter the raw
    evoked1=epochs['pure_pref'].average() 
    evoked2=epochs['pure_sim'].average() 
    
    evoked3=epochs['stay_pp'].average() 
    evoked4=epochs['stay_ss'].average() 
    evoked5=epochs['switch_ps'].average() 
    evoked6=epochs['switch_sp'].average() 
    
    evoked7=epochs['stay_pp','switch_sp'].average() 
    evoked8=epochs['stay_ss','switch_ps'].average() 
    all_evoked= [evoked1,evoked2,evoked3,evoked4,evoked5,evoked6,evoked7,evoked8]
    event_id={'pure_pref': 1,'pure_pref': 1,'pure_sim':2,'stay_pp': 3,
                    'stay_ss':4, 'switch_ps':5 ,'switch_sp':6, } 
    all_evokeds = [epochs[cond].average() for cond in sorted(event_id.keys())]

    
    change_trigger_path= '/Users/new/Documents/new_office_folder/Documents/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/evoked/'
    #evoked1.save(change_trigger_path+sub+'evoked1_task-switching-ev.fif')
    #evoked2.save(change_trigger_path+sub+'evoked2_task-switching-ev.fif')
    #evoked3.save(change_trigger_path+sub+'evoked3_task-switching-ev.fif')
    #evoked4.save(change_trigger_path+sub+'evoked4_task-switching-ev.fif')
    #evoked5.save(change_trigger_path+sub+'evoked5_task-switching-ev.fif')
    #evoked6.save(change_trigger_path+sub+'evoked6_task-switching-ev.fif')
    #evoked7.save(change_trigger_path+sub+'evoked7_task-switching-ev.fif')
    #evoked8.save(change_trigger_path+sub+'evoked8_task-switching-ev.fif')
    #all_evokeds.save(change_trigger_path+sub+'all_evoked_task-switching-ev.fif')
    mne.write_evokeds(change_trigger_path+sub+'all_evoked_task-switching_ave-ev.fif', all_evokeds) 

for sub in subjects:
    average_epochs(sub)
'''    
def all_evoked(sub):    
    print('Processing',sub)
    #setting the path for the required(detrended) files
    evoked_path ='/Users/new/Documents/new_office_folder/Documents/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/evoked/'
    evoked1_file = opj(evoked_path,sub+'evoked1_task-switching-ev.fif')
    evoked2_file = opj(evoked_path,sub+'evoked2_task-switching-ev.fif')
    evoked3_file = opj(evoked_path,sub+'evoked3_task-switching-ev.fif')
    evoked4_file = opj(evoked_path,sub+'evoked4_task-switching-ev.fif')
    evoked5_file = opj(evoked_path,sub+'evoked5_task-switching-ev.fif')
    evoked6_file = opj(evoked_path,sub+'evoked6_task-switching-ev.fif')
    evoked7_file = opj(evoked_path,sub+'evoked7_task-switching-ev.fif')
    evoked8_file = opj(evoked_path,sub+'evoked8_task-switching-ev.fif')
    
    #read evoked
    evoked1 = mne.read_evokeds(evoked1_file)
    evoked2 = mne.read_evokeds(evoked2_file)
    evoked3 = mne.read_evokeds(evoked3_file)
    evoked4 = mne.read_evokeds(evoked4_file)
    evoked5 = mne.read_evokeds(evoked5_file)
    evoked6 = mne.read_evokeds(evoked6_file)
    evoked7 = mne.read_evokeds(evoked7_file)
    evoked8 = mne.read_evokeds(evoked8_file)
    
    all_evoked= evoked1,evoked2,evoked3,evoked4,evoked5,evoked6,evoked7,evoked8
    all_evoked.save(change_trigger_path+sub+'_all_evoked_task-switching-ev.fif')

#run for all subjects
for sub in subjects:
    all_evoked(sub) 
'''    
#ERP     


ev=[]
for sub in subjects:
    print('Processing',sub)
    evoked_path ='/Users/new/Documents/new_office_folder/Documents/Bids_format_EEG/switching_task/derivatives/preprocessing/'+sub+'/evoked/'
    evoked_file = opj(evoked_path,sub+'all_evoked_task-switching_ave-ev.fif')
    ev.append(mne.read_evokeds(evoked_file))

ev2= pd.DataFrame.from_records(ev)
    
#cond1 = pd.Series.tolist(ev2[:][0])
#cond2 = pd.Series.tolist(ev2[:][1])
cond3 = pd.Series.tolist(ev2[:][2])
cond4 = pd.Series.tolist(ev2[:][3])
cond5 = pd.Series.tolist(ev2[:][4])
cond6 = pd.Series.tolist(ev2[:][5])
#cond7 = pd.Series.tolist(ev2[:][6])
#cond8 = pd.Series.tolist(ev2[:][7])



#convert to list
#ERP_cond1_ave= mne.grand_average(all_inst=cond1)
#ERP_cond2_ave= mne.grand_average(all_inst=cond2)
ERP_cond3_ave= mne.grand_average(all_inst=cond3).apply_baseline(inteval)
ERP_cond4_ave= mne.grand_average(all_inst=cond4).apply_baseline(inteval)
ERP_cond5_ave= mne.grand_average(all_inst=cond5).apply_baseline(inteval)
ERP_cond6_ave= mne.grand_average(all_inst=cond6).apply_baseline(inteval)

shorter_epochst = ERP_cond3_ave.copy().crop(tmin=0.1, tmax=1, include_tmax=True)
shorter_epochsw = ERP_cond6_ave.copy().crop(tmin=0.1, tmax=1, include_tmax=True)
#get time window
shorter_epochst = shorter_epochst.times
np.where(times==0.25)
np.where(time==0.35)

from mne.stats import permutation_cluster_test


#plot ERP
#ERP_cond1_ave.plot(spatial_colors=True,titles="Pure_preference",picks=['Fp1', 'Fpz','Fp2','F1', 'Fz', 'F2','C1', 'Cz', 'C2','P1', 'Pz', 'P2'])
#ERP_cond2_ave.plot(spatial_colors=True,titles="Pure_similarity",picks=['Fp1', 'Fpz','Fp2','F1', 'Fz', 'F2','C1', 'Cz', 'C2','P1', 'Pz', 'P2'])
ERP_cond3_ave.plot(spatial_colors=True,titles="stay_preference",picks=['Fp1', 'Fpz','Fp2','F1', 'Fz', 'F2','C1', 'Cz', 'C2','P1', 'Pz', 'P2'])
ERP_cond4_ave.plot(spatial_colors=True,titles="stay_similarity",picks=['Fp1', 'Fpz','Fp2','F1', 'Fz', 'F2','C1', 'Cz', 'C2','P1', 'Pz', 'P2'])
ERP_cond5_ave.plot(spatial_colors=True,titles="switch_similarity",picks=['Fp1', 'Fpz','Fp2','F1', 'Fz', 'F2','C1', 'Cz', 'C2','P1', 'Pz', 'P2'])
ERP_cond6_ave.plot(spatial_colors=True,titles="switch_preference",picks=['Fp1', 'Fpz','Fp2','F1', 'Fz', 'F2','C1', 'Cz', 'C2','P1', 'Pz', 'P2'])



#pref
colorsp = dict(stay_pp="steelblue", switch_sp="maroon")

#sim
colorps = dict(stay_ss="sienna", switch_ps="darkgreen")

colorstst = dict(stay_ss="sienna", stay_pp="steelblue")

colorswsw = dict(switch_sp="maroon", switch_ps="darkgreen")


#plot topo
mne.viz.plot_compare_evokeds(dict(stay_pp=ERP_cond3_ave, switch_sp=ERP_cond6_ave),colors=colorsp,ylim = dict(eeg=[-5, 5]), picks='eeg', axes='topo')

mne.viz.plot_compare_evokeds(dict(stay_ss=ERP_cond4_ave, switch_ps=ERP_cond5_ave),colors=colorps,ylim = dict(eeg=[-5, 5]), picks='eeg', axes='topo')

mne.viz.plot_compare_evokeds(dict(stay_ss=ERP_cond3_ave, stay_pp=ERP_cond4_ave),colors=colorss,ylim = dict(eeg=[-5, 5]), picks='eeg', axes='topo')

mne.viz.plot_compare_evokeds(dict(switch_sp=ERP_cond6_ave, switch_ps=ERP_cond5_ave),colors=colorss,ylim = dict(eeg=[-5, 5]), picks='eeg', axes='topo')


#compare condition at region of interest
#pref
colors = dict(stay_pp="crimson", switch_sp="maroon")

mne.viz.plot_compare_evokeds(dict(stay_pp=ERP_cond3_ave, switch_sp=ERP_cond6_ave),vlines='auto',ylim = dict(eeg=[-3, 3]),
                             legend='upper left', show_sensors='upper right',picks=['P1', 'Pz', 'P2'],colors=colors)
#sim   
#warna muda stay #warna tua switch
colors = dict(stay_ss="yellowgreen", switch_ps="darkgreen")

mne.viz.plot_compare_evokeds(dict(stay_ss=ERP_cond4_ave, switch_ps=ERP_cond5_ave),vlines='auto',ylim = dict(eeg=[-3, 3]),
                             legend='upper left', show_sensors='upper right',picks=['C1', 'Cz', 'C2'],colors=colors)

'''
#to do 02/17 -> change removal ica in bids format adjust the exclude channnel


#baseline correction



    
#Artifact Rejection







#evoked 
#- condition --> select event ID

#evoked
#-temporal  --> crop time interest

#ERP
#--- 4 rois  

#F test 

#saving value each subjects to csv to calculate with rumination score. 


#

    




'''
import pickle
import matplotlib
import numpy as np
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt 


def loaddatapickle():
    filehandler=open()  
'''
