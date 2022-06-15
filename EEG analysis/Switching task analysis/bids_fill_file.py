#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:21:52 2020

@author: risadwiratnasari
"""


import os
import mne
import sys
import mne_bids
mne_bids in sys.modules
import mne_bids

from mne import io
import pandas as pd 
from mne_bids import write_raw_bids, BIDSPath, print_dir_tree


'''reminder
always refresh data_dir every change script'''

data_dir = '/Users/new/Documents/new_office_folder/Documents/Bids_format_EEG/'

eog_channels = ['LEOG','REOG','EMG1']

df=pd.read_csv(os.path.join(data_dir+'/switching_task/participants.tsv'),\
                index_col=0, delimiter='\t')
subjects = df.index.values
n_subs = len(subjects)

#converting subject files in mne bids format

#create function filling bids file
def mne_bids(sub, index):
    print('+++++++++++++++++++++++++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++')
    print(sub)

    a = df['orig_id'].loc[sub]
    print(a)
    in_file = os.path.join(data_dir,'RAW','data_exp_eeg',a +'.vhdr')
    print("see what's inside")
    print(in_file)
    in_eeg = io.read_raw_brainvision(in_file, eog = eog_channels)
    in_eeg.info['line_freq'] = 60
    bids_root = '/Users/new/Documents/new_office_folder/Documents/Bids_format_EEG/switching_task/rawdata/'
    bids_path = BIDSPath(f'{index:03}', datatype='eeg', task='colourswitching', root=bids_root)
    write_raw_bids(in_eeg, bids_path,overwrite=True)
    #write_raw_bids(in_eeg, (sub)+'_task_switching',bids_root=outdir)

#run for all subject
for index, sub in enumerate(subjects):
        mne_bids(sub, index+16)

#-------------------------------------------------------------------#          
'''

#fill with fif format 
#subject_id=[13]
#for subject_id in subject_ids:        
in_file = os.path.join(data_dir,'RAW','data_exp_eeg','Sub-013_Raw.fif')
in_eeg =  mne.io.read_raw_fif(in_file)
in_eeg.info['line_freq'] = 60
subject_id=13
bids_root = '/Users/new/Documents/new_office_folder/Documents/Bids_format_EEG/switching_task/rawdata/'
bids_path = BIDSPath(f'{subject_id:03}', datatype='eeg', task='colourswitching', root=bids_root)
write_raw_bids(in_eeg, bids_path, overwrite=True)

        #write_raw_bids(in_eeg, (sub)+'_task_switching',bids_root=outdir)

#run for all subject
for sub in subjects:
    mne_bids(sub)    



#modif
for index, sub in enumerate(subjects):
    modif_index = index+7
    if modif_index != 12:
        mne_bids(sub, modif_index+1)

