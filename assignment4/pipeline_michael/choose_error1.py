# -*- coding: utf-8 -*-
"""
Visualize a given failure. Failure and important parameters need to be specified in this code


"""

###--------> Imports
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os
import seaborn as sb
import pickle
#####------------> some parameters

figure_path = "./figures/transformer"
data_path = "./inter_results/inter1"
time_plots  = False 
pairplots   = False
correlation = True
failed_component =  'TRANSFORMER'
turbine_id = 'T07'


### --------------> Import data
#import failure data
df_failures = pd.read_csv('../data/wind-farm-1-failures-training.csv', sep=';')
df_failures['Timestamp'] = pd.to_datetime(df_failures.Timestamp)

###------------>  create and clean dfs

# import signals
df_signals = pd.read_csv('../data/wind-farm-1-signals-training.csv', sep=';', parse_dates=True)
df_signals['Timestamp'] = pd.to_datetime(df_signals.Timestamp)

# multi-index
df_signals.index = pd.MultiIndex.from_frame(df_signals.iloc[:, 0:2])

#%% Preprocess Data
# select failure types

df_failed_component = df_failures[df_failures['Component'] == failed_component]
failed_turbines = pd.unique(df_failed_component['Turbine_ID'])

# select turbines with failed component
df_signals_failed = df_signals.loc[failed_turbines]
#df_signals_failed = df_signals.loc[turbine_id]
# relevant features (expand or reduce if you like)
relevant_features = ['Gen_RPM_Max',
                     'Gen_RPM_Min',
                     'Gen_RPM_Avg',
                     'Gen_RPM_Std', 
                     'Gen_Bear_Temp_Avg', 
                     'Gen_Phase1_Temp_Avg', 'Gen_Phase2_Temp_Avg', 'Gen_Phase3_Temp_Avg', 
                     'Prod_LatestAvg_ActPwrGen0', 'Prod_LatestAvg_ActPwrGen1', 'Prod_LatestAvg_ActPwrGen2', 
                     'Prod_LatestAvg_TotActPwr', 
                     'Prod_LatestAvg_ReactPwrGen0', 'Prod_LatestAvg_ReactPwrGen1', 'Prod_LatestAvg_ReactPwrGen2', 
                     'Prod_LatestAvg_TotReactPwr', 
                     'Gen_SlipRing_Temp_Avg',
                     'Prod_LatestAvg_TotReactPwr',
                     'HVTrafo_Phase1_Temp_Avg',
                     'HVTrafo_Phase2_Temp_Avg',
                     'HVTrafo_Phase3_Temp_Avg',
                     'Grd_InverterPhase1_Temp_Avg',
                     'Grd_Prod_Pwr_Avg',
                     'Grd_Prod_VoltPhse1_Avg', 'Grd_Prod_VoltPhse2_Avg', 'Grd_Prod_VoltPhse3_Avg', 
                     'Grd_Prod_CurPhse1_Avg', 'Grd_Prod_CurPhse2_Avg', 'Grd_Prod_CurPhse3_Avg', 
                     'Grd_Prod_Pwr_Max', 'Grd_Prod_Pwr_Min',
                     'Grd_Busbar_Temp_Avg', 'Amb_WindSpeed_Est_Avg',
                     'Grd_Prod_Pwr_Std', 'Grd_Prod_ReactPwr_Avg', 'Grd_Prod_ReactPwr_Max'
                     ]
target_features = [
                     'HVTrafo_Phase1_Temp_Avg',
                     'HVTrafo_Phase2_Temp_Avg',
                     'HVTrafo_Phase3_Temp_Avg',
                     'Grd_Prod_CurPhse1_Avg',
                     'Grd_Prod_CurPhse2_Avg',
                     'Grd_Prod_CurPhse3_Avg', 
                    ]
df_signals_failed = df_signals_failed[relevant_features]

if not os.path.isdir("./inter_results"): # create directory to store the data 
    os.mkdir("./inter_results")
df_signals_failed.to_pickle("./inter_results/inter1.pickle")
#%% Look at relevant features "around" failure time step.

failure_id = 10

# for example
turbine = df_failed_component['Turbine_ID'].loc[failure_id]
time = df_failed_component['Timestamp'].loc[failure_id]


# loop through the features, 
#create a subplot 
#with 3 subplots for different time ranges 

timedelta1 = dt.timedelta(days=6, hours=1) # different time ranges to plot
timedelta2 = dt.timedelta(days=1,hours=1)
timedelta3 = dt.timedelta(hours=6)

# createing masks to choose the right time data
mask1 = (df_signals_failed.loc[turbine].index > time - timedelta1) & (df_signals_failed.loc[turbine].index < (time + 0.2*timedelta1)) 
mask2 = (df_signals_failed.loc[turbine].index > time - timedelta2) & (df_signals_failed.loc[turbine].index < (time + 0.2*timedelta3)) 
mask3 = (df_signals_failed.loc[turbine].index > time - timedelta3) & (df_signals_failed.loc[turbine].index < (time + 0.2*timedelta3)) 

if not os.path.isdir(figure_path): # create directory to store figs
    os.mkdir(figure_path)

if time_plots:
    print("generating time plots")
    for i,feature1 in enumerate(target_features):
        #breakpoint()
        plotpath = (figure_path + feature1)

        ###-----------> Plot Time series
        fig, axs = plt.subplots(3,1)
        if not os.path.isdir(plotpath):
            os.mkdir( plotpath ) # see if a dir for the feature exists and create it
        
        axs[0].plot(df_signals_failed[feature1].loc[turbine].loc[mask1].sort_index()) #time frame of 6 days
        axs[1].plot(df_signals_failed[feature1].loc[turbine].loc[mask2].sort_index()) #time frame of 1 days
        axs[2].plot(df_signals_failed[feature1].loc[turbine].loc[mask3].sort_index()) # time frame of 6 h

        # show the failure by adding a vertical line in the plots:
        axs[0].axvline(time, color="black")
        axs[1].axvline(time, color="black")
        axs[2].axvline(time, color="black")
        
        # add title and save figure
        fig.suptitle(f"Timeplot of feature: {feature1}")
        time_filename = f"{plotpath}/timeseries.pdf"
        plt.savefig(time_filename, bbox_inches='tight')

        ### ---------------> Do pair plots
        ###!!! ----> we need normalization for this. 
        # Do this in a quick and dirty manner. Using avg over the whole time is probably not a good idea,\
        # as some values stay at a high or low val for a long time

        for j,feature2 in enumerate(relevant_features): # going through every feature again
            fig_c, axs_c = plt.subplots(3,1)
            # plot feature1 
            # normalize time series data by the current max
            f1_1 =df_signals_failed[feature1].loc[turbine].loc[mask1].sort_index() / df_signals_failed[feature1].loc[turbine].loc[mask1].sort_index().max() 
            f1_2 =df_signals_failed[feature1].loc[turbine].loc[mask2].sort_index() / df_signals_failed[feature1].loc[turbine].loc[mask2].sort_index().max() 
            f1_3 =df_signals_failed[feature1].loc[turbine].loc[mask3].sort_index() / df_signals_failed[feature1].loc[turbine].loc[mask3].sort_index().max() 
            f2_1 =df_signals_failed[feature2].loc[turbine].loc[mask1].sort_index() / df_signals_failed[feature2].loc[turbine].loc[mask1].sort_index().max() 
            f2_2 =df_signals_failed[feature2].loc[turbine].loc[mask2].sort_index() / df_signals_failed[feature2].loc[turbine].loc[mask2].sort_index().max() 
            f2_3 =df_signals_failed[feature2].loc[turbine].loc[mask3].sort_index() / df_signals_failed[feature2].loc[turbine].loc[mask3].sort_index().max() 
           
            # plot feature 1
            axs_c[0].plot(f1_1) #time frame of 6 days
            axs_c[1].plot(f1_2) #time frame of 1 days
            axs_c[2].plot(f1_3) #time frame of 6 hours 

            # plot feature 2
            axs_c[0].plot(f2_1) #time frame of 6 days
            axs_c[1].plot(f2_2) #time frame of 1 days
            axs_c[2].plot(f2_3) #time frame of 6 hours
            
            # add a line for the failure:
            axs_c[0].axvline(time, color="black")
            axs_c[1].axvline(time, color="black")
            axs_c[2].axvline(time, color="black")
            
            # title and save
            fig_c.suptitle(f"Pair plot of features: {feature1} and {feature2}")
            pair_filename = f"{plotpath}/{feature1}_to_{feature2}.pdf"
            plt.savefig(pair_filename, bbox_inches='tight')
            plt.close('all')

#### Try out seaborn:
# general aproach:
# create new dir
#1 create time sections 
#2 normalize --> here not necessary I think 
#2 choose observations to plot (not too many at once!)
# pipe frames into seaborn

if pairplots:
    print("generating pair plots")
    # choose a smaller subset of features
    current_features = ['Gen_Bear_Temp_Avg', 
                         'Gen_Phase1_Temp_Avg', 
                         'Gen_Phase2_Temp_Avg',
                         'Gen_Phase3_Temp_Avg']#,
                         #'Prod_LatestAvg_ActPwrGen0',
                         #'Gen_SlipRing_Temp_Avg']

    seaborn_path = figure_path + str("/seaborn") # path to save all the seaborn plots

    if not os.path.isdir(seaborn_path) : # make sure the directory exists
        os.mkdir(seaborn_path)

    # choose a subset of the df. Features can be selected here
    time_section1 = df_signals_failed.loc[turbine].loc[mask1].sort_index()[current_features] 
    time_section2 = df_signals_failed.loc[turbine].loc[mask2].sort_index()[current_features]
    time_section3 = df_signals_failed.loc[turbine].loc[mask3].sort_index()[current_features]
    
    # add time index as column for plotting. maybe there is a more straighforward way to achieve that.
    time_section1["t_index"] = time_section1.index
    time_section2["t_index"] = time_section2.index
    time_section3["t_index"] = time_section3.index
    #breakpoint()
    time_sections = [time_section1, time_section2, time_section3] # list of all time sections to loop over
    print("start creating seaborn plots")
    for i,time in enumerate(["6d","1d","6h"]):
        print(f"start plotting for dt = {time}")
        fig_sb = sb.pairplot(time_sections[i], hue ="t_index",palette="RdBu") # do a pairplot and use timeindex for coloring
        current_figure_path = f"{seaborn_path}/pairplot_t_{time}.pickle" 
        print("done creating figure. \nCreate files")
        pickle.dump(fig_sb, open(current_figure_path, 'wb'))# save the figure as a python pickle
        plt.savefig(f"{seaborn_path}/pairplot_t_{time}.pdf", bbox_inches='tight' ) 
        print("files created")
        plt.close('all')
if correlation:
    cor_mat = df_signals_failed.corr(numeric_only=True)
    corr_plot = sb.heatmap(cor_mat,cmap="vlag")
    #plt.savefig(f"{figure_path}/correlation_plot.pdf", bbox_inches='tight' ) 
    #fig_corr.savefig(f"./correlation_plot.pdf", bbox_inches='tight' ) 

    fig_corr = corr_plot.get_figure()
    fig_corr.savefig("./correlation_plot_trafo.png",bbox_inches='tight') 



