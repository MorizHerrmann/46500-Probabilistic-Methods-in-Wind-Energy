# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 12:06:21 2022

@author: morit
"""

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt


#%% Load data

# signal
df = pd.read_csv('wind-farm-1-signals-training.csv', sep=';', parse_dates=True)
df['Timestamp'] = pd.to_datetime(df.Timestamp)

# failures
df_failures = pd.read_csv('wind-farm-1-failures-training.csv', sep=';')
df_failures['Timestamp'] = pd.to_datetime(df_failures.Timestamp)


#%% Multi indexing can be favourable when grouping is necessary in later stages
df.index = pd.MultiIndex.from_frame(df.iloc[:,0:2])
df.index.sortlevel()

#Level 0 is the first index in our dataset
df.groupby(level = 0).mean()


#%% Plotting data

# --1. Time series plot
turbine = 'T09'
f, ax = plt.subplots(1,2,figsize=(15,5))
df.xs(turbine, level=0)['Amb_WindSpeed_Avg'].plot(color='b',ax=ax[0])
ax[1].scatter(df.xs(turbine, level=0)['Amb_WindSpeed_Avg'],
                   df.xs(turbine, level=0)['Grd_Prod_Pwr_Avg'],
                   color = 'b')
plt.grid()
ax[0].legend()
f.suptitle('Variables of '+turbine)
ax[1].set_xlabel('WS [m/s]')
ax[1].set_ylabel('Power')

plt.figure()
plt.rcParams.update({'font.size': 10}) #Update fontsize
chan = ['Amb_WindSpeed_Avg','Grd_Prod_Pwr_Avg','Gen_RPM_Avg','Hyd_Oil_Temp_Avg','Grd_Prod_ReactPwr_Avg','Amb_Temp_Avg']
pd.plotting.scatter_matrix(df.xs(turbine,level =0)[chan],
                          figsize = (15,15))
plt.show()


#%% Sort data with time
df.sort_index(axis=0,level = 1,ascending = True, inplace = True)
df.head()

#%% Separate in turbines
turbines = df['Turbine_ID'].unique()
df_T = [df[df['Turbine_ID'] == ID] for ID in turbines]

#%% Cut out failure times & remove idling
# dt_before = dt.timedelta(weeks = 4*2)
# dt_after  = dt.timedelta(weeks = 4*1)

# df_T_normal = [0]*len(df_T)
# df_T_fail = [0]*len(df_T)
# for i, turbine in enumerate(turbines):
    
#     # find all failure timestamps for that turb
#     failure_times = df_failures[df_failures.Turbine_ID == turbine]['Timestamp'].to_list()
    
#     # sort by time index
#     df_T[i].sort_values(by='Timestamp', inplace=True)

#     # use a mask to kick out all failure data for the current wind turbine
#     fail_mask = [False]*len(df_T[i]) 
#     for j in failure_times:
#         fail_j_mask = (j-dt_before < df_T[i].Timestamp) & (df_T[i].Timestamp < j+dt_after)
#         fail_mask = fail_mask | fail_j_mask
#         # fail_mask = fail_mask & ((df_T[i].Timestamp < (j-dt_before) ) | (df_T[i].Timestamp > (j+dt_after)))
    
#     # idling = rotor is not turning
#     idle_mask = df_T[i].Rtr_RPM_Avg == 0
    
#     # yields all the clean data now
#     df_T_normal[i] = df_T[i].loc[~fail_mask & ~idle_mask] 
    
#     # data with failure for testing
#     df_T_fail[i] = df_T[i].loc[fail_mask & ~idle_mask]
    
#     print()
#     print(turbine)
#     print(f'Old length:\t{len(df_T[i]):.0f}')
#     print(f'Idling: \t{sum(idle_mask)/len(df_T[i])*100:.0f}%')
#     print(f'Failure:\t{sum(fail_mask)/len(df_T[i])*100:.0f}%')
#     print(f'New length:\t{len(df_T_normal[i])}')
#     print(f'Data loss:\t{(len(df_T_normal[i])/len(df_T[i])-1)*100:.0f}%')

#%% Cut out failure times & remove idling

min_power_cut = df.Grd_Prod_Pwr_Min <= 1
max_power_cut = df.Grd_Prod_Pwr_Max >= 1

max_RPM_cut = df.Gen_RPM_Max >= 1400

operation = max_RPM_cut & ~min_power_cut

trans = max_power_cut & max_RPM_cut & ~operation

idling = ~operation & ~trans

state = np.zeros((len(operation),1))

for ii in range(len(operation)):
    if( operation.iloc[ii] == True):
        state[ii] = 1
    elif idling.iloc[ii] == True:
        state[ii] = 0
    elif trans.iloc[ii] == True:
        state[ii] = 2
        
df['Status_Flag'] = state  

#%%  Plots

plot = 1
if (plot == 1):
    f, ax = plt.subplots(1,1, figsize=(10,7))
    plt.rcParams.update({'font.size': 12}) 
    plt.plot(df['Amb_WindSpeed_Avg'][operation],
         df['Grd_Prod_Pwr_Avg'][operation],
         '.', color='k', label = '$Operation$')
    plt.plot(df['Amb_WindSpeed_Avg'][trans],
         df['Grd_Prod_Pwr_Avg'][trans],
         '.', color='c', label = '$Transient$')
    plt.plot(df['Amb_WindSpeed_Avg'][idling],
         df['Grd_Prod_Pwr_Avg'][idling],
         '.', color='b', label = '$Idling$')
    plt.xlabel('$WS\;[m/s]$')
    plt.ylabel('$Power\;[kW]$')
    plt.legend(loc = 'upper center',bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncol=3)
    plt.grid()
    plt.tight_layout()
    plt.show()



#%% Save data

for i, turbine in enumerate(turbines):
    
    df_T[i].to_csv('data/' + turbine + '_full.csv', sep=';')
    # df_T_normal[i].to_csv('data/traindata/' + turbine + '_normal.csv', sep=';')
    # df_T_fail[i].to_csv('testdata/' + turbine + '_fail.csv', sep=';')