# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 13:28:35 2022

@author: morit
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%% Load data

turbines = ['T01', 'T06', 'T07', 'T09', 'T11']
df_T = [pd.read_csv('data/traindata/' + turbine + '_normal.csv', sep=';') for turbine in turbines]
for i in range(len(turbines)):
    df_T[i]['Timestamp'] = pd.to_datetime(df_T[i]['Timestamp'])

target = 'Prod_LatestAvg_TotActPwr'
features = ['Grd_Prod_CurPhse1_Avg',
            'Grd_Prod_CurPhse2_Avg',
            'Grd_Prod_CurPhse3_Avg',
            'Amb_WindSpeed_Avg',
            'Gen_Phase1_Temp_Avg',
            'Gen_Phase2_Temp_Avg',
            'Gen_Phase3_Temp_Avg',
            'Amb_Temp_Avg',
            'Gen_RPM_Avg']

#%% Calculate auto-correlation and integral time scale

def autocorrelation(x):
    lag = 0; autocorr = [1];
    while autocorr[-1] > 0:
        lag += 1
        autocorr.append(x.autocorr(lag))
    return autocorr


def crosscorrelation(x, y):
    lag = 0; crosscorr = [1];
    while crosscorr[-1] > 0:
        lag += 1
        crosscorr.append(x.corr(y.shift(lag)))
    return crosscorr


target_integral_timescale = np.zeros(len(turbines))
feature_integral_timescale = np.zeros((len(turbines), len(features)))
for i, turbine in enumerate(turbines):
    y = df_T[i][target]
    X = df_T[i][features]
    
    # target
    autocorr = autocorrelation(y)
        
    # plt.figure(); plt.plot(autocorr)
    # plt.grid(); plt.xlabel("Lag [steps]"); plt.ylabel("Auto-correlation of Power")
    target_integral_timescale[i] = np.trapz(autocorr)
    
    # features
    for j, feature in enumerate(features):
        crosscorr = crosscorrelation(y, X[feature])
        
        # plt.figure(); plt.plot(crosscorr)
        # plt.grid(); plt.xlabel("Lag [steps]"); plt.ylabel("Cross-correlation of power with " + feature)
        feature_integral_timescale[i, j] = np.trapz(crosscorr)
    
#%% Integral timescale

# Better overshoot than underestimate.
# So the maximum amount of information is retained. 

plt.figure()
bins = np.linspace(0, 200, round(200/10+1))
plt.hist(feature_integral_timescale.flatten(), bins)

plt.hist(target_integral_timescale, bins)
plt.xticks(bins, rotation=45); plt.xlabel("Integral Timescales"); plt.ylabel("Count")
plt.legend("Features", "Target")

integral_timescale = 130    # covers everything but outliers

#%% Moving average

relevant_features = features.copy(); 
relevant_features.append(target)

df_T_mvAvg = [0] * len(turbines)
for i, turbine in enumerate(turbines):
    df_T_mvAvg[i] = df_T[i][relevant_features].rolling(integral_timescale).mean()
    df_T_mvAvg[i]['Timestamp'] = df_T[i]['Timestamp']
    df_T_mvAvg[i].to_csv('data/traindata/' + turbine + '_normal_mvAvg.csv', sep=';')

#%% Test: !!! Kills my GPU !!!

import pandas as pd
full = pd.read_csv('data/T01_full.csv', sep=';')
normal = pd.read_csv('data/traindata/T01_normal.csv', sep=';')
mvAvg = pd.read_csv('data/traindata/T01_normal_mvAvg.csv', sep=';')

import matplotlib.pyplot as plt
plt.figure()
plt.plot(full.Timestamp, full.Prod_LatestAvg_TotActPwr, '.')
plt.plot(normal.Timestamp, normal.Prod_LatestAvg_TotActPwr, '.')
plt.plot(mvAvg.Timestamp, mvAvg.Prod_LatestAvg_TotActPwr, '.')
plt.xlabel('Timestamp'); plt.ylabel('Active Power'); plt.grid()
