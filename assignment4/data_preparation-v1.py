# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:00:46 2022

@author: morit
"""

#%% Split into turbines

import pandas as pd

# import signals
df = pd.read_csv('data/wind-farm-1-signals-training.csv', sep=';', parse_dates=True)
df['Timestamp'] = pd.to_datetime(df.Timestamp)

IDs = ['T01', 'T06', 'T07', 'T09', 'T11']
for n, ID in enumerate(IDs):
    df_id = df[df['Turbine_ID']==ID]
    df_id.sort_values('Timestamp', inplace=True)
    df_id.to_csv('data/signal_'+ID+'_10min.csv', sep=';', index=False)
    print(f'{(n+1)/len(IDs)*100:.0f}%')

#%% Daily averages

import pandas as pd

IDs = ['T01', 'T06', 'T07', 'T09', 'T11']

for n, ID in enumerate(IDs):
    df = pd.read_csv('data/signal_' + ID +'_10min.csv', sep=';')
    df.index=pd.to_datetime(df['Timestamp'])
    df_1d = df.resample('D').mean()
    df_1d.to_csv('data/signal_'+ID+'_1d.csv', sep=';')
    print(f'{(n+1)/len(IDs)*100:.0f}%')
