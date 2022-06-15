#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 00:17:08 2020
@author: risadwiratnasari
"""
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import csv
from statistics import stdev 
import pandas as pd

'''------------------------------------------------------------------------''' 
def analyseParticipant(mat_data): 
    data_dir_mat = '/Volumes/DATA/a/'
    #data_dir_mat = '/Volumes/DATA/active_folder/eeg_data/S14/'
    mat = scipy.io.loadmat(data_dir_mat + mat_data)
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
        
    mix_trial = trial_list[90:450]
    for index,item in enumerate(trial_list[90:450]):
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
    stdv= stdev(trial_list['RT'])   
    mean_data= np.mean(trial_list['RT']) 
    low_outlier= mean_data - (2*stdv)
    print(low_outlier)
    high_outlier= mean_data + (2*stdv)
    print(high_outlier)

     
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
      
    
    #remove the outlier
    data_without_outlier = trial_list[trial_list.detect_outlier != 'outlier']
    #remove the missing trial 
    data_clean= data_without_outlier[data_without_outlier.RT != 0]
    data_clean2= trial_list[trial_list.RT != 0]

    # condition selection full set
    pure_p = data_clean.query('label=="pure_p"')
    pure_s = data_clean.query('label=="pure_s"')
    stay_pp= data_clean.query('label=="stay_pp"')
    stay_ss= data_clean.query('label=="stay_ss"')
    switch_ps= data_clean.query('label=="switch_ps"')
    switch_sp= data_clean.query('label=="switch_sp"')
    
    #pick RT all trial 
    RT_pure_p  = pure_p['RT']
    RT_pure_s  = pure_s['RT']
    RT_stay_pp = stay_pp['RT']
    RT_stay_ss = stay_ss['RT']
    RT_switch_ps = switch_ps['RT']
    RT_switch_sp = switch_sp['RT']
    
    #correct trial only full set 
    
    pure_s_c =pure_s.query('[1] in resp_acc') 
    stay_ss_c = stay_ss.query('[1] in resp_acc')
    switch_ps_c = switch_ps.query('[1] in resp_acc')
    
    #RT correct only
    RT_pure_s_c = pure_s_c ['RT']
    RT_stay_ss_c = stay_ss_c['RT']
    RT_switch_ps_c = switch_ps_c['RT']
    #amount correct
    #pure
    pure_accuracy = len(pure_s_c)
    #mix    
    accuracy_stay = len(stay_ss_c)
    accuracy_switch = len(switch_ps_c)
    #sum mix
    accuracy_mix= accuracy_stay+accuracy_switch
    #percentage
    all_pure_similarity = trial_list.query('label=="pure_s"')
    len_pure_similarity = len(all_pure_similarity)
    all_mix_similarity = trial_list.query('label=="stay_ss"')+trial_list.query('label=="switch_ps"')
    len_mix_similarity = len(all_mix_similarity)

    correct_percentage_pure_similarity = pure_accuracy /  len_pure_similarity *  100
    correct_percentage_mix_similarity = accuracy_mix /  len_mix_similarity *  100        
    
    print("this is the amount for pure trial ")
    print (pure_accuracy)
    print("---------") 
    print("percentage")
    print(correct_percentage_pure_similarity)
    print("--------------------######--------------")
    print("--------------------######--------------")

    print("this is the amount for mix trial ")
    print (accuracy_mix)
    print("---------")
    print("percentage")
    print(correct_percentage_mix_similarity)
    print("---------")
    print("--------------------######--------------")


    
    #variable RT all
    
    RT_pure_p
    RT_pure_s
    RT_stay_pp
    RT_stay_ss
    RT_switch_ps
    RT_switch_sp
    
    
    #variable RT correct only 
    RT_pure_s_c
    RT_stay_ss_c
    RT_switch_ps_c
    
    
    #Mean RT all trials 
    print("this is the avg of RT pure preference")
    mean_pure_pref= np.mean(RT_pure_p) 
    print(mean_pure_pref)
    print("---------") 
    
    print("this is the avg of RT pure similarity")
    mean_pure_sim = np.mean(RT_pure_s) 
    print(mean_pure_sim)
    print("---------") 
    
    print("this is the avg of RT mix preference to preference")
    mean_mix_pp= np.mean(RT_stay_pp)
    print(mean_mix_pp)
    print("---------") 
    
    print("this is the avg of RT mix similarity to similarity")
    mean_mix_ss= np.mean(RT_stay_ss)
    print(mean_mix_ss)
    print("---------") 
    
    print("this is the avg of RT mix switch preference to similarity")
    mean_mix_ps= np.mean(RT_switch_ps)
    print(mean_mix_ps)
    print("---------") 
    
    print("this is the avg of RT mix switch similarity to preference")
    mean_mix_sp= np.mean(RT_switch_sp)
    print(mean_mix_sp)
    print("---------")
    
    print("now mean correct only------<>-------<>---------<>------<>---<>----")
    
    #print mean correct only
    
    print("this is the avg of RT pure similarity correct only")
    mean_pure_s= np.mean(RT_pure_s_c)
    print(mean_pure_s)
    print("---------") 
    
    print("this is the avg of RT stay similarity correct only")
    mean_stay_ss= np.mean(RT_stay_ss_c)
    print(mean_stay_ss)
    print("---------") 
    
    print("this is the avg of RT switch similarity")
    mean_switch_s= np.mean(RT_switch_ps_c)
    print(mean_switch_s)
    print("---------")
    
    #RRS value
    
    
    bins = np.linspace(0.0, 2.5, 100)
    plt.hist(data_clean2['RT'],bins,color='orange', alpha=0.5, label='with outlier')
    plt.hist(data_clean['RT'], bins,color='darkblue', alpha=0.5,label='non-outlier')
    plt.legend(loc='upper right')
    plt.show()
    
    
    
    participant_data = {'correct_response_pure':len_pure_similarity,'correct_percentage_pure_similarity':correct_percentage_pure_similarity, 
                            'correct_percentage_mix_similarity':correct_percentage_mix_similarity,'correct_response_mix':len_mix_similarity,
                            'sorted_rt_pure_preference':RT_pure_p, 'sorted_rt_pure_similarity':RT_pure_s,
                            'sorted_rt_mix_stay_preference':RT_stay_pp, 'sorted_rt_mix_stay_similarity':RT_stay_ss,
                            'sorted_rt_mix_switch_preference_similarity':RT_switch_ps, 'sorted_rt_mix_switch_similarity_preference':RT_switch_sp,
                            'mean_pure_pp':mean_pure_pref,'mean_pure_ss':mean_pure_sim, 'mean_stay_pp':mean_mix_pp, 'mean_stay_ss':mean_mix_ss,
                            'mean_switch_ps':mean_mix_ps,'mean_switch_sp':mean_mix_sp,
                            'low_outlier':low_outlier, 'high_outlier':high_outlier,
                            'data_clean':data_clean,'data_clean_non_outlier':data_clean2, 'stdv':stdv, 'IQR':IQR, 
                            'pure_sim_c':RT_pure_s_c, 'stay_sim_c':RT_stay_ss_c, 'switch_sim_c':RT_switch_ps_c }
                
        
    return participant_data 

foureenth_participant = analyseParticipant('Color_task_manual_S21_21.mat')

#method for RRS data

#data_dir_excel= '/Users/risadwiratnasari/Documents/script_data/'
def calling_RRS(df_rrs): 
    data_dir_excel= '/Users/risadwiratnasari/Documents/script_data/'
    df_rrs = pd.read_excel(data_dir_excel + df_rrs)

    sum_participant = df_rrs.sum(axis=1)
    participant = df_rrs.iterrows()
    i=0
    
    array_participant_data = []
    array_participant = []
    for each_participant in df_rrs.index:
        participant_dict = {
            "name":df_rrs['Name'][each_participant],
            "sum_participant":str(sum_participant[i])
            }
        
        array_participant.append(participant_dict)
        
        #print(df_rrs['Name'][each_participant]+" "+str(sum_participant[i]))
        i = i + 1
        
    #print(array_participant)
    #print(array_participant[3]["sum_participant"])
        
    
    Depressive= df_rrs[["Name","Think about how alone you feel", "Think “I won’t be able to do my job if I don’t snap out of this”","Think about your feelings of fatigue and achiness","Think about how hard it is to concentrate"\
                        ,"Think about how passive and unmotivated you feel.","Think about how you don’t seem to feel anything anymore"\
                            ,"Think “Why can’t I get going?”","Think “I won’t be able to concentrate if I keep feeling this way.”","Think about how sad you feel."\
                                , "Think about all your shortcomings, failings, faults, mistakes","Think about how you don’t feel up to doing anything"\
                                    ,"Think about how angry you are with yourself"]]
    sum_participant_Depressive = Depressive.sum(axis=1)
    participant = Depressive.iterrows()
    i=0
    
    for each_participant in Depressive.index:
        participant_dict = {
            "name":Depressive['Name'][each_participant],
            "sum_participant_Depressive":str(sum_participant_Depressive[i])
            }
        array_participant.append(participant_dict)
        #print(Depressive['Name'][each_participant]+" "+str(sum_participant[i]))
        i = i + 1    
    
    
    Brooding = df_rrs[["Name","Think “What am I doing to deserve this?”","Think “Why do I always react this way?”","Think about a recent situation, wishing it had gone better"\
                       , "Think “Why do I have problems other people don’t have?”","Think “Why can’t I handle things better?”",]]
    sum_participant_Brooding = Brooding.sum(axis=1)
    participant = Brooding.iterrows()
    i=0
    
    for each_participant in Brooding.index:
        participant_dict = {
            "name":Brooding['Name'][each_participant],
            "sum_participant_Brooding":str(sum_participant_Brooding[i])
            }
        array_participant.append(participant_dict)
        #print(df_rrs['Name'][each_participant]+" "+str(sum_participant[i]))
        i = i + 1

    
    Reflective = df_rrs[["Name","Analyze recent events to try to understand why you are depressed","Go away by yourself and think about why you feel this way"\
                         ,"Write down what you are thinking about and analyze it","Analyze your personality to try to understand why you are depressed"\
                             ,"Go someplace alone to think about your feelings"]]
    
    sum_participant_Reflective = Reflective.sum(axis=1)
    participant = Reflective.iterrows()
    i=0
    
    for each_participant in Reflective.index:
        participant_dict = {
            "name":Reflective['Name'][each_participant],
            "sum_participant_Reflective":str(sum_participant_Reflective[i])
            }
        array_participant.append(participant_dict)
        #print(df_rrs['Name'][each_participant]+" "+str(sum_participant[i]))
        i = i + 1
    
    
    a = pd.DataFrame(array_participant)
    R= a[['name','sum_participant']].dropna()
    D= a[['sum_participant_Depressive']].dropna().reset_index(drop=True)
    B= a[['sum_participant_Brooding']].dropna().reset_index(drop=True)   
    Re= a[['sum_participant_Reflective']].dropna().reset_index(drop=True) 
    array_participant = pd.concat([R,D,B,Re], ignore_index=False, axis=1)

    return array_participant

    
    '''-------------------------------------------'''







first_participant = analyseParticipant('Color_task_manual_data003_Linlin.mat')



#test to csv

data_clean.to_csv('e.csv')
