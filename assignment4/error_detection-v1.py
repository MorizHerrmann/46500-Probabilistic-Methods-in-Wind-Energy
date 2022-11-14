# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 10:37:32 2022

@author: morit
"""

import pandas as pd
import datetime as dt
%matplotlib
import matplotlib.pyplot as plt


#%% Import data

# import failure data
df_failures = pd.read_csv('data/wind-farm-1-failures-training.csv', sep=';')
df_failures['Timestamp'] = pd.to_datetime(df_failures.Timestamp)

# import signals
df_signals = pd.read_csv('data/wind-farm-1-signals-training.csv', sep=';', parse_dates=True)
df_signals['Timestamp'] = pd.to_datetime(df_signals.Timestamp)

# multi-index
df_signals.index = pd.MultiIndex.from_frame(df_signals.iloc[:, 0:2])

#%% Preprocess Data

# select failure types
failed_component =  'GENERATOR'
df_failed_component = df_failures[df_failures['Component'] == failed_component]
failed_turbines = pd.unique(df_failed_component['Turbine_ID'])

# select turbines with failed component
df_signals_failed = df_signals.loc[failed_turbines]

# relevant features (expand or reduce if you like)
relevant_features = ['Gen_RPM_Max', 'Gen_RPM_Min', 'Gen_RPM_Avg', 'Gen_RPM_Std', 
                     'Gen_Bear_Temp_Avg', 
                     'Gen_Phase1_Temp_Avg', 'Gen_Phase2_Temp_Avg', 'Gen_Phase3_Temp_Avg', 
                     'Prod_LatestAvg_ActPwrGen0', 'Prod_LatestAvg_ActPwrGen1', 'Prod_LatestAvg_ActPwrGen2', 
                     'Prod_LatestAvg_TotActPwr',
                     'Prod_LatestAvg_ReactPwrGen0', 'Prod_LatestAvg_ReactPwrGen1', 'Prod_LatestAvg_ReactPwrGen2', 
                     'Prod_LatestAvg_TotReactPwr', 
                     'Gen_SlipRing_Temp_Avg']
df_signals_failed = df_signals_failed[relevant_features]

#%% Look at relevant features "around" failure time step.

failure_id = 6

# for example
turbine = df_failed_component['Turbine_ID'].loc[failure_id]
time = df_failed_component['Timestamp'].loc[failure_id]

df_turbine = df_signals_failed.loc[turbine]
df_values = df_turbine['Gen_SlipRing_Temp_Avg'].values

# Dt = dt.timedelta(days=30)
# mask = (time-Dt < df_turbine.index) & (df_turbine.index < time+Dt)
# df_turbine[['Prod_LatestAvg_TotActPwr']].iloc[mask].plot()

plt.figure()
plt.plot(df_turbine.index, values, '.')
plt.show()

failure_times = df_failed_component['Timestamp'][df_failed_component['Turbine_ID'] == 'T06'].values

plt.vlines(failure_times, min(values), max(values), 'r')
