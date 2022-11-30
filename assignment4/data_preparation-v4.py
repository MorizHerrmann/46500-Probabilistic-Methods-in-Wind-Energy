# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:44:10 2022

@author: morit
"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

#%% Load data

# signal
df = pd.read_csv('data/wind-farm-1-signals-training.csv', sep=';', parse_dates=True)
df.Timestamp = pd.to_datetime(df.Timestamp)

# failures
df_fail = pd.read_csv('data/wind-farm-1-failures-training.csv', sep=';')
df_fail.Timestamp = pd.to_datetime(df_fail.Timestamp)

print('Data loaded.')

#%% Separate in turbines

turbines = df['Turbine_ID'].unique()
nT = len(turbines)
df_raw = [df[df.Turbine_ID == ID].sort_values('Timestamp') for ID in turbines]

print('Turbines separated.')

#%% Filter idle and stop

df_op = [0] * nT
for i in range(nT):
    # stop = df_raw[i].Rtr_RPM_Min <= 0        # turbine does not always rotate
    # idle = df_raw[i].Grd_Prod_Pwr_Min <= 0   # turbine delivers no power
    
    # df_op[i] = df_raw[i][~stop & ~idle]    # operating turbine
    
    df_op[i] = df_raw[i][(df_raw[i].Gen_RPM_Max >= 1250) & (df_raw[i].Grd_Prod_Pwr_Min >= 1)]    # operating turbine
    
print('Operation found.')

#%% Relevant features

target = ['HVTrafo_Phase1_Temp_Avg', 'HVTrafo_Phase2_Temp_Avg', 'HVTrafo_Phase3_Temp_Avg']
features = ['Gen_Phase1_Temp_Avg', 
            'Gen_Phase1_Temp_Avg',
            'Gen_Phase1_Temp_Avg',
            'Grd_Prod_PsblePwr_Avg', 
            'Prod_LatestAvg_ActPwrGen1']
relevant = ['Timestamp', *target, *features]

df_rel = [df_op[i][relevant] for i in range(nT)]
df_rel = [df_op[i] for i in range(nT)]

print('Relevant features extracted.')

#%% Determine Integral timescale

determine_integral_timescale = False

if determine_integral_timescale:
    target_integral_timescale = np.zeros(nT)
    feature_integral_timescale = np.zeros((nT, len(features)))
    for i in range(nT):
        y = df_rel[i][target]
        
        # target
        lag = 0; autocorr = [1];
        while autocorr[-1] > 0:
            lag += 1
            autocorr.append(y.autocorr(lag))
            
        # plt.figure(); plt.plot(autocorr)
        # plt.grid(); plt.xlabel("Lag [steps]"); plt.ylabel("Auto-correlation of Power")
        target_integral_timescale[i] = np.trapz(autocorr)
        
        # features
        for j, feat in enumerate(features):
            lag = 0; crosscorr = [1];
            while crosscorr[-1] > 0:
                lag += 1
                crosscorr.append(y.corr(df_rel[i][feat].shift(lag)))
            
            # plt.figure(); plt.plot(crosscorr)
            # plt.grid(); plt.xlabel("Lag [steps]"); plt.ylabel("Cross-correlation of power with " + feature)
            feature_integral_timescale[i, j] = np.trapz(crosscorr)
            
    # Better overshoot than underestimate. So the maximum amount of information is retained. 
    plt.figure()
    bins = np.linspace(0, 200, round(200/10+1))
    plt.hist(feature_integral_timescale.flatten(), bins)
    
    plt.hist(target_integral_timescale, bins)
    plt.xticks(bins, rotation=45); plt.xlabel("Integral Timescales"); plt.ylabel("Count")
    plt.legend("Features", "Target")

#%% Low-pass filter

integral_timescale = 10

df_lpf = [0] * nT
for i in range(nT):
    df_lpf[i] = df_rel[i].drop('Timestamp', axis=1).rolling(integral_timescale).mean()
    df_lpf[i]['Timestamp'] = df_rel[i].Timestamp
    df_lpf[i].set_index(pd.Series(range(len(df_lpf[i]))), inplace=True)
    df_lpf[i].drop(range(integral_timescale-1), axis=0, inplace=True)

print('Low-pass filter applied')

#%% Cut out failure times & remove idling

dt_before = dt.timedelta(weeks = 4*6)
dt_after  = dt.timedelta(weeks = 4*1)

df_normal = [0]*len(df_raw)
df_failed = [0]*len(df_raw)
for i in range(nT):
    
    # find all failure timestamps for that turb
    dt_fail = df_fail[df_fail.Turbine_ID == turbines[i]]['Timestamp'].to_list()
    
    # use a mask to kick out all failure data for the current wind turbine
    fail = [False] * len(df_lpf[i]) 
    for j in dt_fail:
        fail = fail | (j-dt_before < df_lpf[i].Timestamp) & (df_lpf[i].Timestamp < j+dt_after)
            
    df_normal[i] = df_lpf[i][~fail]   # yields all the clean data now
    df_failed[i] = df_lpf[i][fail]      # data with failure for testing
    
    print()
    print(turbines[i])
    print(f'Total length:\t\t{len(df_raw[i]):.0f}')
    # print(f'Idle:   \t\t\t{sum(idle)/len(df_raw[i])*100:.0f}%')
    # print(f'Stop:   \t\t\t{sum(stop)/len(df_raw[i])*100:.0f}%')
    print(f'Operating Length:\t{len(df_op[i]):.0f} = {len(df_op[i])/len(df_raw[i])*100:.0f}%')
    print(f'Failure:\t\t\t{sum(fail)/len(df_op[i])*100:.0f}%')

#%% Save data

for i, turbine in enumerate(turbines):
    
    # df_raw[i]
    df_op[i].to_csv('data/' + turbine + '_op.csv', sep=';')
    # df_rel[i].to_csv('data/' + turbine + '.csv', sep=';')
    df_lpf[i].to_csv('data/' + turbine + '_lpf.csv', sep=';')
    df_normal[i].to_csv('data/traindata/' + turbine + '_normal.csv', sep=';')
    df_failed[i].to_csv('data/testdata/' + turbine + '_fail.csv', sep=';')
    pass
    
print('Saved.')
