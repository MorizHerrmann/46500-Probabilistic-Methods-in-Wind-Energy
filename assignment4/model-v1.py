# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:18:38 2022

@author: morit
"""

import pandas as pd
import sklearn.preprocessing
import sklearn.neural_network
import matplotlib.pyplot as plt

#%% Load data

IDs = ['T01', 'T06', 'T07', 'T09', 'T11']
df_T = [pd.read_csv('data/traindata/'+ID+'_normal_mvAvg.csv', sep=';') for ID in IDs]

df_allT = pd.concat(df_T).drop('Unnamed: 0', axis=1)
df_allT.drop(range(130), axis=0, inplace=True)

X = df_allT.drop(['Timestamp', 'Prod_LatestAvg_TotActPwr'], axis=1, inplace=False).values
y = df_allT['Prod_LatestAvg_TotActPwr'].values.reshape(-1, 1)

#%% Scaling

# Removing the mean and scaling to unit variance
scaler = sklearn.preprocessing.StandardScaler() 

# fit the scales and apply the transformation
X_scale = scaler.fit_transform(X) 

#%% Initialize model

# Choose model:
    # RF = Random Forest
    # NN = Neural Network

method = 'NN'

if method == 'NN':
    
    settings = {'hidden_layer_sizes': (10, 10, 10), 
                'activation': 'relu', 
                'early_stopping': True}
    model = sklearn.neural_network.MLPRegressor(**settings)
    
elif method == 'RF':
    print('Random Forest Regression is not yet implemented.')
else:
    print('The chosen method does not exist.')

#%% Train-Test-Split

X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_scale, y, shuffle = True, test_size = 0.1)

#%% Training

model.fit(X_train, y_train)

# validation
print(f'Validation:\t{model.score(X_val, y_val)*100:.1f}%')

#%% Testing on failure set

df_test = pd.read_csv('data/testdata/T01_fail_mvAvg.csv', sep=';')
df_test['Timestamp'] = pd.to_datetime(df_test['Timestamp'])
df_test.drop('Unnamed: 0', axis=1, inplace=True)
df_test.drop(range(130), axis=0, inplace=True)

X = df_test.drop(['Timestamp', 'Prod_LatestAvg_TotActPwr'], axis=1, inplace=False).values
y = df_test['Prod_LatestAvg_TotActPwr'].values.reshape(-1, 1)

X_test = scaler.fit_transform(X)
y_test = y

y_pred = model.predict(X_test)
print(f'Testing:\t{model.score(X_test, y_test)*100:.1f}%')

plt.figure(figsize=(10, 6))
plt.plot(df_test['Timestamp'], df_test['Prod_LatestAvg_TotActPwr'], '.')
plt.plot(df_test['Timestamp'], y_pred, '.')
plt.grid(); plt.xlabel('Time'); plt.ylabel('Active power'); 
plt.legend(['True', 'Pred.'])
