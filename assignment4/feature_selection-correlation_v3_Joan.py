# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:43:12 2022

@author: joanp
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 13:00:38 2022

@author: Joan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection

plot = 1 

#%% Load data

turbines = ['T01', 'T06', 'T07', 'T09', 'T11']
df_T = [pd.read_csv('data/traindata/' + turbine + '_normal.csv', sep=';') for turbine in turbines]

df_T_all = pd.concat(df_T)

#remove indexing
df_T_all.drop(df_T_all.columns[0], axis = 1, inplace = True)

# #Multiindexing
# df_T_all.index = pd.MultiIndex.from_frame(df_T_all.iloc[:,0:2])
# df_T_all.drop(df_T_all.columns[2], axis = 1, inplace = True)
# df_T_all.index.sortlevel()
# df_T_all.head()

#%% Calculate correlation coefficient

correlation = df_T_all.corr()
absolute_correlation = correlation.abs()

#%% Remove all NaN in data

# array_sum = np.sum(df_T_all)
# array_has_nan = np.isnan(df_T_all)

# print(array_has_nan)

variables = df_T_all.columns.values

for ii in range(len(variables)):
    df_T_all.dropna(subset = [variables[ii]],inplace = True)

variables_list = ['Grd_Prod_CurPhse1_Avg']
list_features = pd.DataFrame()

for ch in variables_list:
    name_feature = ch 
    list_features[name_feature] = df_T_all[name_feature]        
        
list_features.head()

#%% check if there are still NaN
# array_sum = np.sum(df_T_all)
# array_has_nan = np.isnan(df_T_all)

# print(array_has_nan)
#%% Corr coeff
Grd_Prod_1 = np.array(df_T_all['Grd_Prod_CurPhse1_Avg'].values,dtype = 'float')

corrcoeff = np.zeros(len(variables))

for ii in range(2,len(variables)-1,1):
    variable_ii = np.array(df_T_all[variables[ii]].values,dtype = 'float')
    if sum(variable_ii)== 0:
        corrcoeff[ii] = 0
       
    else:
        corrcoeff[ii] = np.corrcoef(Grd_Prod_1,variable_ii)[0,1]
        

#%% biggest corr coeff

Corr_coef_variable = dict(zip(variables,corrcoeff))
Corr_coef_variable_sort = {}

sort_coeff = sorted(Corr_coef_variable,key =lambda dict_key1: abs(Corr_coef_variable[dict_key1]),reverse=True)

for w in sort_coeff:
    Corr_coef_variable_sort[w] = Corr_coef_variable[w]



end_plot = 18 - 2;

if plot == 1:
    plt.figure(1)
    plt.title('Grd_Prod_CurPhse1_Avg 1')
    plt.rcParams.update({'font.size': 10}) 
    plt.bar(range(end_plot),list(Corr_coef_variable_sort.values())[0:end_plot],tick_label =list(Corr_coef_variable_sort)[0:end_plot])
    plt.xticks(rotation=90)
    plt.ylabel('$\\rho [-]$')
    plt.tight_layout()
    plt.show()
    
    # plt.figure(2)
    # plt.rcParams.update({'font.size': 14}) #Update fontsize
    # plt.bar(range(endingp_plot),list(Corr_coef2_sort.values())[0:endingp_plot],tick_label =list(Corr_coef2_sort)[0:endingp_plot])
    # plt.xticks(rotation=90)
    # plt.ylabel('$\\rho [-]$')
    # plt.tight_layout()
    # plt.show()
    
    # plt.figure(3)
    # plt.rcParams.update({'font.size': 14}) #Update fontsize
    # plt.bar(range(endingp_plot),list(Corr_coef3_sort.values())[0:endingp_plot],tick_label =list(Corr_coef3_sort)[0:endingp_plot])
    # plt.xticks(rotation=90)
    # plt.ylabel('$\\rho [-]$')
    # plt.tight_layout()
    # plt.show()

    # plt.figure(4)
    # plt.rcParams.update({'font.size': 14}) #Update fontsize
    # plt.bar(range(endingp_plot),list(Corr_coef10_sort.values())[0:endingp_plot],tick_label =list(Corr_coef10_sort)[0:endingp_plot])
    # plt.xticks(rotation=90)
    # plt.ylabel('$\\rho [-]$')
    # plt.tight_layout()
    # plt.show()


#%% create dataframe for training




