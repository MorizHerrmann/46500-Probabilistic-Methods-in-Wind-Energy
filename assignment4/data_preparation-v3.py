# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 12:06:21 2022

@author: morit
"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

#%% Load data

# signal
df = pd.read_csv('data/wind-farm-1-signals-training.csv', sep=';', parse_dates=True)
df['Timestamp'] = pd.to_datetime(df.Timestamp)

# failures
df_failures = pd.read_csv('data/wind-farm-1-failures-training.csv', sep=';')
df_failures['Timestamp'] = pd.to_datetime(df_failures.Timestamp)

#%% Separate in turbines
turbines = df['Turbine_ID'].unique()
df_T = [df[df['Turbine_ID'] == ID] for ID in turbines]

#%% Cut out failure times & remove idling
dt_before = dt.timedelta(weeks = 4*2)
dt_after  = dt.timedelta(weeks = 4*1)

df_T_normal = [0]*len(df_T)
for i, turbine in enumerate(turbines):
    
    # find all failure timestamps for that turb
    failure_times = df_failures[df_failures.Turbine_ID == turbine]['Timestamp'].to_list()
    
    # sort by time index
    df_T[i].sort_values(by='Timestamp', inplace=True)

    # use a mask to kick out all failure data for the current wind turbine
    fail_mask = [False]*len(df_T[i]) 
    for j in failure_times:
        fail_j_mask = (j-dt_before < df_T[i].Timestamp) & (df_T[i].Timestamp < j+dt_after)
        fail_mask = fail_mask | fail_j_mask
        # fail_mask = fail_mask & ((df_T[i].Timestamp < (j-dt_before) ) | (df_T[i].Timestamp > (j+dt_after)))
    
    # idling = rotor is not turning
    idle_mask = df_T[i].Rtr_RPM_Avg == 0
    
    # yields all the clean data now
    df_T_normal[i] = df_T[i].loc[~fail_mask & ~idle_mask] 
    
    print()
    print(turbine)
    print(f'Old length:\t{len(df_T[i]):.0f}')
    print(f'Idling: \t{sum(idle_mask)/len(df_T[i])*100:.0f}%')
    print(f'Failure:\t{sum(fail_mask)/len(df_T[i])*100:.0f}%')
    print(f'New length:\t{len(df_T_normal[i])}')
    print(f'Data loss:\t{(len(df_T_normal[i])/len(df_T[i])-1)*100:.0f}%')

#%% Save data

for i, turbine in enumerate(turbines):
    
    df_T[i].to_csv('data/' + turbine + '_full.csv', sep=';')
    df_T_normal[i].to_csv('data/traindata/' + turbine + '_normal.csv', sep=';')
    