# -*- coding: utf-8 -*-
"""
Script to generate clean datasets from the failure data.
Data with failed state is kicked out and dataset split at the failure.
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

if False:

    df = pd.read_csv('data/wind-farm-1-signals-training.csv', sep=';', parse_dates=True)
    df['Timestamp'] = pd.to_datetime(df.Timestamp)

df_failures = pd.read_csv('../data/wind-farm-1-failures-training.csv', sep=';')
df_failures['Timestamp'] = pd.to_datetime(df_failures.Timestamp)
if True:
    df_signals = pd.read_pickle("./inter_results/inter1.pickle")
if False:
    # multi-index
    df_signals = df
    df_signals.index = pd.MultiIndex.from_frame(df_signals.iloc[:, 0:2])

    ###---------------> Choosing the data to be used

#1 Find all turbines
breakpoint()
turbines = pd.unique(df_signals.index.get_level_values('Turbine_ID'))
breakpoint()
#2 Set time regions where we want to get rid of data
dt_before = dt.timedelta(weeks = 4*6) # stop taking vals 6 months before failure
dt_after  = dt.timedelta(weeks = 4*1) # waiting 1 month after failure. maybe wait longer?


####----------------> extract usefull timeframes
for turbine in turbines:
    # find all failure timestamps for that turb
    failure_times = df_failures[df_failures.Turbine_ID == turbine]['Timestamp'].to_list()
    df_subset = df_signals[df_signals.index.get_level_values('Turbine_ID')==turbine].sort_index() # choose only one turbine at the time and sort by time index.
    ##df_subset = df_subset.droplevel(0) # drop the first index, to make indexing easier

    f_mask = [True]*len(df_subset) # use a mask to kick out all failure data for the current wind turbine
    for i in failure_times:
        f_mask = f_mask & ((df_subset.index.get_level_values('Timestamp') < (i-dt_before) ) | (df_subset.index.get_level_values('Timestamp') > (i+dt_after)))
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
