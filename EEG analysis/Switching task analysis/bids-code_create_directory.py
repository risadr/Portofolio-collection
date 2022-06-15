#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:37:51 2020

@author: risadwiratnasari
"""


"""
making folders for bids-eeg format
creating a raw data folder containing individual participants folders
creating an eeg folder for each participant folder
"""

import os
import pathlib
import pandas as pd

#Data directory 
data_dir= '/Users/new/Documents/new_office_folder/Documents/Bids_format_EEG/switching_task/'

# Load subject info
#tsv_file = open(data_dir+"participants.tsv")
#read_tsv = csv.reader(tsv_file, delimiter="\t")

#df = pd.read_csv("/Users/risadwiratnasari/Practice_arrange_data/switching_task_three_participants/participants.tsv", sep = "\t",nrows =4, header=0, encoding='utf-8')
df=pd.read_csv(os.path.join(data_dir,'participants.tsv'),index_col=0,\
                 delimiter='\t')
subjects = df.index.values
n_subs = len(subjects)


#creating the rawdata folder

if not os.path.exists(os.path.join(data_dir,'rawdata')):
    os.mkdir(os.path.join(data_dir,'rawdata'))

# Create subject and EEG folders
for sub in subjects: 
    if not os.path.exists(os.path.join(data_dir,'rawdata',sub)):
        os.mkdir(os.path.join(data_dir,'rawdata',sub))
    if not os.path.exists(os.path.join(data_dir,'rawdata',sub,'eeg')):
        os.mkdir(os.path.join(data_dir,'rawdata',sub,'eeg'))
    if not os.path.exists(os.path.join(data_dir,'rawdata',sub,'behavioral_data')):
        os.mkdir(os.path.join(data_dir,'rawdata',sub,'behavioral_data'))    



# Data directory
#data_dir= "/Users/risadwiratnasari/Practice_arrange_data/switching\
 #   _task_three_participants/"


# Load subject info
#df = pd.read_csv((data_dir,'participants.tsv'),index_col=0,delimiter='\t')











