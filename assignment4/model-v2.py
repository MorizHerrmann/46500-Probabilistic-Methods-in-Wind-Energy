# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:18:38 2022

@author: morit
"""

import pandas as pd
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.ensemble
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

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

train_turbine = 'T07'

if train_turbine == 'all':
    IDs = ['T01', 'T06', 'T07', 'T09', 'T11']
else:
    IDs = [train_turbine]

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

    # # Validtion
    # y_pred_val = models[i].predict(X_val).reshape(-1, 1)
    # res_val = y_pred_val - y_val
    # plt.hist(res_val, 50, density=True, alpha=1/nT)
    # plt.xlabel('Residual'); plt.ylabel('Frequency (pdf)')
    # plt.title(target + ': Validation on normal behavior')
    
    # print('Validation of ' + target)
    # print(f'Rsq:\t{models[i].score(X_val, y_val)*100:.1f}%')
    # print(f'RMSE:\t{(np.sum(res_val**2))**0.5:.0f}')


#%% Validate difference over whole time series

test_turbine = 'T07'

df_full = pd.read_csv('data/'+test_turbine+'_op.csv', sep=';')
df_full['Timestamp'] = pd.to_datetime(df_full['Timestamp'])
t = df_full['Timestamp']

diff_true = [(df_full[targets[0]] - df_full[targets[1]])**2, 
             (df_full[targets[1]] - df_full[targets[2]])**2, 
             (df_full[targets[2]] - df_full[targets[0]])**2]

diff_sym_true = ((df_full[targets[0]] - df_full[targets[1]])**2 + (df_full[targets[1]] - df_full[targets[2]])**2 + (df_full[targets[2]] - df_full[targets[0]])**2) ** (1/2)

y_full_pred = [0] * nT
for i in range(nT):
    X = df_full[features].drop(targets[i], axis=1)
    X_scale = scaler.transform(X)
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
plt.title('Train: '+train_turbine+', Test: '+test_turbine+': Distribution of sym. residual')

df_fail = pd.read_csv('data/wind-farm-1-failures-training.csv', sep=';')
df_fail.Timestamp = pd.to_datetime(df_fail.Timestamp)
t_fail_trans = df_fail[(df_fail.Turbine_ID == test_turbine) & (df_fail.Component == 'TRANSFORMER')]['Timestamp']
t_fail_other = df_fail[(df_fail.Turbine_ID == test_turbine) & (df_fail.Component != 'TRANSFORMER')]['Timestamp']


plt.figure(figsize=(10, 6))
plt.plot(t, res_sym_lpf, 'k', label='sym. residual')
plt.vlines(t_fail_trans, *plt.gca().get_ylim(), alpha=0.5, color='r', linestyles='-', label='Trafo failure')
plt.vlines(t_fail_other, *plt.gca().get_ylim(), alpha=0.5, color='r', linestyles='--', label='Other failure')
plt.grid(); plt.xlabel('Time'); plt.ylabel('Residual = Pred. - True'); 
plt.title('Train: '+train_turbine+', Test: '+test_turbine+': Timeseries of sym. residual')
plt.legend()
plt.savefig('figures/residual.png', dpi=500)

#%% ROC and AUC

# Receiver operating characteristic (ROC)

dt_before = dt.timedelta(weeks = 4*6)
dt_after  = dt.timedelta(weeks = 4*1)

for i in range(nT):
    
    # find all failure timestamps for that turb
    dt_fail = df_fail[df_fail.Turbine_ID == test_turbine]['Timestamp'].to_list()
    
    # use a mask to kick out all failure data for the current wind turbine
    fail = [False] * len(res_sym_lpf) 
    for j in dt_fail:
        fail = fail | (j-dt_before < df_full.Timestamp) & (df_full.Timestamp < j+dt_after)

plt.figure()
plt.plot(df_full.Timestamp[fail], res_sym_lpf[fail], 'r,', label='failure')
plt.plot(df_full.Timestamp[~fail], res_sym_lpf[~fail], 'g,', label='normal')
plt.title('Time series of residual'); plt.xlabel('Time'); plt.ylabel('R'); 
plt.grid(); plt.xticks(rotation=45); plt.legend()
plt.tight_layout()
plt.savefig('figures/timeseres.png', dpi=500)

# +-residual
fpr, tpr, thresholds = sklearn.metrics.roc_curve(fail[mvAvg:], res_sym_lpf[mvAvg:])
AUC = sklearn.metrics.roc_auc_score(fail[mvAvg:], res_sym_lpf[mvAvg:])

# Confusion matrix for "optimum"
opt1 = 2450
opt2 = 9500

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--')
plt.plot(fpr[[opt1, opt2]], tpr[[opt1, opt2]], 'ro')
plt.gca().set_aspect('equal'); plt.grid(); 
plt.title(f'ROC with AUC = {AUC:.2f}'); plt.xlabel('FPR'); plt.ylabel('TPR'); 
plt.tight_layout()
# plt.savefig('figures/ROC.png', dpi=500)

y_true = fail
y_est = res_sym_lpf > thresholds[opt1]
C = np.unique(y_true).shape[0]
cm1 = sklearn.metrics.confusion_matrix(y_true, y_est);
cm1 = cm1/np.sum(cm1)
accuracy = 100*cm1.diagonal().sum()/cm1.sum(); error_rate = 100-accuracy;
fig, ax = plt.subplots(1, 1)
plt.imshow(cm1, cmap='Wistia', interpolation='None', origin='lower');
plt.colorbar(format='%.2f')
plt.xticks(range(C), ['normal', 'failed']); plt.yticks(range(C), ['normal', 'failed']);
plt.xlabel('Predicted class'); plt.ylabel('Actual class');
plt.title(f'Accuracy: {accuracy:.0f}%, Error Rate: {error_rate:.0f}%');

for (j,i),label in np.ndenumerate(cm1):
    ax.text(i,j,round(label, 2),ha='center',va='center')
    ax.text(i,j,round(label, 2),ha='center',va='center')
plt.tight_layout()
# plt.savefig('figures/confusionmatrix1.png', dpi=500)

y_true = fail
y_est = res_sym_lpf > thresholds[opt2]
C = np.unique(y_true).shape[0]
cm2 = sklearn.metrics.confusion_matrix(y_true, y_est);
cm2 = cm2/np.sum(cm2)
accuracy = 100*cm2.diagonal().sum()/cm2.sum(); error_rate = 100-accuracy;
fig, ax = plt.subplots(1, 1)
plt.imshow(cm2, cmap='Wistia', interpolation='None', origin='lower');
plt.colorbar(format='%.2f')
plt.xticks(range(C), ['normal', 'failed']); plt.yticks(range(C), ['normal', 'failed']);
plt.xlabel('Predicted class'); plt.ylabel('Actual class');
plt.title(f'Accuracy: {accuracy:.0f}%, Error Rate: {error_rate:.0f}%');

for (j,i),label in np.ndenumerate(cm2):
    ax.text(i,j,round(label, 2),ha='center',va='center')
    ax.text(i,j,round(label, 2),ha='center',va='center')
plt.tight_layout()
# plt.savefig('figures/confusionmatrix2.png', dpi=500)

# abs residual
fpr_sym, tpr_sym, thresholds_sym = sklearn.metrics.roc_curve(~fail[mvAvg:], res_sym_lpf[mvAvg:].abs())
AUC = sklearn.metrics.roc_auc_score(~fail[mvAvg:], res_sym_lpf[mvAvg:].abs())

opt = 3000

plt.figure()
plt.plot(fpr_sym, tpr_sym)
plt.plot([0, 1], [0, 1], '--')
plt.plot(fpr_sym[opt], tpr_sym[opt], 'ro')
plt.gca().set_aspect('equal'); plt.grid(); 
plt.title(f'ROC with AUC = {AUC:.2f}'); plt.xlabel('FPR'); plt.ylabel('TPR'); 
plt.tight_layout()
# plt.savefig('figures/ROC_sym.pdf')

#%%

t_fail = df_fail[df_fail.Turbine_ID == 'T07']['Timestamp'].to_list()

plt.figure(figsize=(10, 4))
plt.plot(df_full.Timestamp[fail], df_full.HVTrafo_Phase1_Temp_Avg[fail], 'r.', label='error', markersize=1)
plt.plot(df_full.Timestamp[~fail], df_full.HVTrafo_Phase1_Temp_Avg[~fail], 'g.', label='normal', markersize=1)
plt.vlines(t_fail, *plt.gca().get_ylim(), alpha=0.5, color='k', linestyles='--', label='Failure')
plt.title('Time series of residual'); plt.xlabel('Time'); plt.ylabel('HVTrafo_Phase1_Temp_Avg'); 
plt.grid(); plt.xticks(rotation=45); plt.legend()
plt.tight_layout()
plt.savefig('figures/timeseres1.png', dpi=500)
