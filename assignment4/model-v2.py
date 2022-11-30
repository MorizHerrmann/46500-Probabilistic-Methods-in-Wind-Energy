# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:18:38 2022

@author: morit
"""

import pandas as pd
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.ensemble
import matplotlib.pyplot as plt
import numpy as np

#%% Input

targets = ['HVTrafo_Phase1_Temp_Avg', 'HVTrafo_Phase2_Temp_Avg', 'HVTrafo_Phase3_Temp_Avg']
features = [*targets, 
            'Gen_Phase1_Temp_Avg', 
            'Gen_Phase2_Temp_Avg',
            'Gen_Phase3_Temp_Avg',
            # 'Gen_Bear_Temp_Avg', 
            'Gen_Bear2_Temp_Avg',
            'Gen_SlipRing_Temp_Avg',
            'Grd_Prod_CurPhse1_Avg', 
            'Grd_Prod_CurPhse2_Avg',
            'Grd_Prod_CurPhse3_Avg',
            'Grd_Prod_Pwr_Min', 
            'Grd_Prod_PsblePwr_Avg', 
            'Grd_Prod_Pwr_Avg',
            'Grd_Busbar_Temp_Avg',
            'Grd_RtrInvPhase1_Temp_Avg',
            'Grd_RtrInvPhase2_Temp_Avg',
            'Grd_RtrInvPhase3_Temp_Avg',
            'Grd_InverterPhase1_Temp_Avg',
            'Prod_LatestAvg_ActPwrGen1', 
            'Prod_LatestAvg_TotActPwr', 
            'Cont_VCP_ChokcoilTemp_Avg', 
            'Cont_Top_Temp_Avg']

method = 'NN'

#%% Load data

# IDs = ['T01', 'T06', 'T07', 'T09', 'T11']
IDs = ['T07']

df_T = [pd.read_csv('data/traindata/'+ID+'_normal.csv', sep=';') for ID in IDs]
df_allT = pd.concat(df_T).drop('Unnamed: 0', axis=1)
scaler = sklearn.preprocessing.StandardScaler() 
nT = len(targets)

# Initialize model   
if method == 'NN':     
    settings = {'hidden_layer_sizes': (34, 34), 
                'activation': 'relu', 
                'early_stopping': True, 
                'tol': 1e-4, 
                'max_iter': 1000, 
                'random_state': 0}
    models = [sklearn.neural_network.MLPRegressor(**settings) for target in targets]
elif method == 'RF':
    settings = {}
    models = [sklearn.ensemble.RandomForestRegressor(**settings) for target in targets]

for i, target in enumerate(targets):
    
    X = df_allT[features].drop(target, axis=1).values

    # Removing the mean and scaling to unit variance
    X_scale = scaler.fit_transform(X) 

    y = df_allT[target].values.reshape(-1, 1)
    
    # Train-Test-Split
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_scale, y, test_size = 0.1)
    
    # Training
    models[i].fit(X_train, y_train)

    # Validtion
    y_pred_val = models[i].predict(X_val).reshape(-1, 1)
    res_val = y_pred_val - y_val
    plt.hist(res_val, 50, density=True, alpha=1/nT)
    plt.xlabel('Residual'); plt.ylabel('Frequency (pdf)')
    plt.title(target + ': Validation on normal behavior')
    
    print('Validation of ' + target)
    print(f'Rsq:\t{models[i].score(X_val, y_val)*100:.1f}%')
    print(f'RMSE:\t{(np.sum(res_val**2))**0.5:.0f}')


#%% Validate difference over whole time series

turbine = 'T07'

df_full = pd.read_csv('data/'+turbine+'_op.csv', sep=';')
df_full['Timestamp'] = pd.to_datetime(df_full['Timestamp'])
t = df_full['Timestamp']

diff_true = [(df_full[targets[0]] - df_full[targets[1]])**2, 
             (df_full[targets[1]] - df_full[targets[2]])**2, 
             (df_full[targets[2]] - df_full[targets[0]])**2]

diff_sym_true = ((df_full[targets[0]] - df_full[targets[1]])**2 + (df_full[targets[1]] - df_full[targets[2]])**2 + (df_full[targets[2]] - df_full[targets[0]])**2) ** (1/2)

y_full_pred = [0] * nT
for i in range(nT):
    X = df_full[features].drop(targets[i], axis=1)
    X_scale = scaler.fit_transform(X)
    y_full_pred[i] = models[i].predict(X_scale)

diff_pred = [(y_full_pred[0] - y_full_pred[1])**2, 
             (y_full_pred[1] - y_full_pred[2])**2, 
             (y_full_pred[2] - y_full_pred[0])**2]

diff_sym_pred = ((y_full_pred[0] - y_full_pred[1])**2 + (y_full_pred[1] - y_full_pred[2])**2 + (y_full_pred[2] - y_full_pred[0])**2)**(1/2)

res = [diff_pred[i] - diff_true[i] for i in range(nT)]
res_sym = diff_sym_pred - diff_sym_true

mvAvg = 1000
res_lpf = [res[i].rolling(mvAvg).mean() for i in range(nT)]
res_sym_lpf = res_sym.rolling(mvAvg).mean()

plt.figure()
plt.hist(res_sym, 50, density=True, alpha=0.5)
plt.hist(res_sym_lpf, 50, density=True, alpha=0.5)
plt.xlabel('Residual'); plt.ylabel('Frequency (pdf)')
plt.title(f'Distribution of residual {i} for ' + turbine)

h, hax = plt.subplots(1, nT)
print('Testing:')
for i in range(3):
    print(' ')
    print(f'Permutation {i}')
    print(f'Rsq:\t{sklearn.metrics.r2_score(diff_pred[i], diff_true[i]):.3f}')
    print(f'RMSE:\t{(np.sum(res[i]**2))**0.5:.0f}')
    print(f'RMSE lpf:\t{(np.sum(res_lpf[i]**2))**0.5:.0f}')

    hax[i].hist(res[i], 50, density=True, alpha=1/nT)
    hax[i].hist(res_lpf[i], 50, density=True, alpha=1/nT)
    hax[i].set_xlabel('Residual'); hax[i].set_ylabel('Frequency (pdf)')
    hax[i].set_title(f'Distribution of residual {i} for ' + turbine)

df_fail = pd.read_csv('data/wind-farm-1-failures-training.csv', sep=';')
df_fail.Timestamp = pd.to_datetime(df_fail.Timestamp)
t_fail_trans = df_fail[(df_fail.Turbine_ID == turbine) & (df_fail.Component == 'TRANSFORMER')]['Timestamp']
t_fail_other = df_fail[(df_fail.Turbine_ID == turbine) & (df_fail.Component != 'TRANSFORMER')]['Timestamp']

plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(t, res_lpf[i])
plt.vlines(t_fail_trans, *plt.gca().get_ylim(), alpha=0.5, color='r', linestyles='-')
plt.vlines(t_fail_other, *plt.gca().get_ylim(), alpha=0.5, color='r', linestyles='--')
plt.grid(); plt.xlabel('Time'); plt.ylabel('Residual = Pred. - True'); 
plt.title('Full signal of ' + turbine)

plt.figure(figsize=(10, 6))
plt.plot(t, res_sym_lpf, 'k')
plt.vlines(t_fail_trans, *plt.gca().get_ylim(), alpha=0.5, color='r', linestyles='-')
plt.vlines(t_fail_other, *plt.gca().get_ylim(), alpha=0.5, color='r', linestyles='--')
plt.grid(); plt.xlabel('Time'); plt.ylabel('Residual = Pred. - True'); 
plt.title('Full signal of ' + turbine)
