# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 13:00:38 2022

@author: morit
"""

import pandas as pd

#%% Load data

# turbines = ['T01', 'T06', 'T07', 'T09', 'T11']
turbines = ['T07']
df_T = [pd.read_csv('data/' + turbine + '_op.csv', sep=';') for turbine in turbines]

df_T_all = pd.concat(df_T)

#%% Calculate correlation coefficient

correlation = df_T_all.corr()
absolute_correlation = correlation.abs()

#%% Filter most important features

targets = ['HVTrafo_Phase1_Temp_Avg', 'HVTrafo_Phase2_Temp_Avg', 'HVTrafo_Phase3_Temp_Avg']

for target in targets:
    print(' ')
    print(target)
    print('_______________________________________________')
    print(absolute_correlation[target].sort_values(ascending=False)[:15])