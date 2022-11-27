# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 13:00:38 2022

@author: morit
"""

import pandas as pd

#%% Load data

turbines = ['T01', 'T06', 'T07', 'T09', 'T11']
df_T = [pd.read_csv('data/traindata/' + turbine + '_normal.csv', sep=';') for turbine in turbines]

df_T_all = pd.concat(df_T)

#%% Calculate correlation coefficient

correlation = df_T_all.corr()
absolute_correlation = correlation.abs()
