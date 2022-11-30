# -*- coding: utf-8 -*-
"""
Script to generate clean datasets from the failure data.
Datasets are writen out as .pickle
Time before and after failure was chosen
"""

###---------------> IMports
import pandas as pd
import datetime as dt
import numpy as np
import pickle
import os


###----------------> Load in data
df = pd.read_csv('data/wind-farm-1-signals-training.csv', sep=';', parse_dates=True)
df['Timestamp'] = pd.to_datetime(df.Timestamp)

df_failures = pd.read_csv('data/wind-farm-1-failures-training.csv', sep=';')
df_failures['Timestamp'] = pd.to_datetime(df_failures.Timestamp)

# multi-index
df_signals = df
df_signals.index = pd.MultiIndex.from_frame(df_signals.iloc[:, 0:2])

###---------------> Choosing the data to be used
relevant_features = ['Gen_RPM_Max', 'Gen_RPM_Min', 'Gen_RPM_Avg', 'Gen_RPM_Std', 
                     'Gen_Bear_Temp_Avg', 
                     'Gen_Phase1_Temp_Avg', 'Gen_Phase2_Temp_Avg', 'Gen_Phase3_Temp_Avg', 
                     'Prod_LatestAvg_ActPwrGen0', 'Prod_LatestAvg_ActPwrGen1', 'Prod_LatestAvg_ActPwrGen2', 
                     'Prod_LatestAvg_TotActPwr',
                     'Prod_LatestAvg_ReactPwrGen0', 'Prod_LatestAvg_ReactPwrGen1', 'Prod_LatestAvg_ReactPwrGen2', 
                     'Prod_LatestAvg_TotReactPwr', 
                     'Gen_SlipRing_Temp_Avg']

#1 Find all turbines
turbines = pd.unique(df_signals['Turbine_ID'])
#2 Set time regions where we want to get rid of data
dt_before = dt.timedelta(weeks = 4*6) # stop taking vals 6 months before failure
dt_after  = dt.timedelta(weeks = 4*1) # waiting 1 month after failure. maybe wait longer?


####----------------> extract usefull timeframes
for turbine in turbines:
    # find all failure timestamps for that turb
    failure_times = df_failures[df_failures.Turbine_ID == turbine]['Timestamp'].to_list()
    df_subset = df_signals[df_signals['Turbine_ID']==turbine].sort_index() # choose only one turbine at the time and sort by time index.
    ##df_subset = df_subset.droplevel(0) # drop the first index, to make indexing easier

    f_mask = [True]*len(df_subset) # use a mask to kick out all failure data for the current wind turbine
    for i in failure_times:
        f_mask = f_mask & ((df_subset['Timestamp'] < (i-dt_before) ) | (df_subset['Timestamp'] > (i+dt_after)))
    data_clean = df_subset.loc[f_mask] # yields all the clean data now. Set a column for which section it belongs to 
    if not os.path.isdir("./data/traindata"): # create directory to store the data 
        os.mkdir("./data/traindata")
    
    # Split the data between the failures and save as pickle
    for i in range(len(failure_times) +1):
        if i ==0: 
            df_save = data_clean.loc[turbine][data_clean.Timestamp.min():failure_times[i]]
        elif i== len(failure_times) : 
            df_save = data_clean.loc[turbine][failure_times[i-1]:data_clean.Timestamp.max()]
        else:
            df_save = data_clean.loc[turbine][failure_times[i-1]:failure_times[i]]
        df_save.to_pickle(f"./data/traindata/{turbine}_set{i}.pickle")
print("Data written to ./data/traindata")
print("Done")
