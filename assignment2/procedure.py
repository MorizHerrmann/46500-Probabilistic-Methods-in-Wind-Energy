import numpy as np
from sample import rosenblatt
from loads import MysteriousLoadFunc
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


def model(x, y):

    #%% NORMALIZE DATA
    x_norm = (x - x.mean(axis=0)) / x.std(axis=0)
    y_norm = (y - y.mean()) / y.std()

    #%% NEURAL NETWORK
    # initialize
    param = {'activation': 'relu',
             'learning_rate': 'adaptive',
             'max_iter': 750,
             'random_state': 1,  
             'verbose': False,
             'early_stopping': True,
             'n_iter_no_change': 10}
    reg = MLPRegressor(hidden_layer_sizes=(100,100), **param)

    # split train and test data
    x_train, x_test, y_train, y_test = train_test_split(x_norm, y_norm, test_size=0.2, random_state=1, shuffle=True)

    reg.fit(x_train, y_train)          # train
    y_pred_norm = reg.predict(x_test)  # predict output

    #%% RENORMALIZE DATA
    y_pred = y_pred_norm * y.std() + y.mean()
    y_true = y_test * y.std() + y.mean()

    #%% STATISTICS
    output = {'E': np.mean(y_pred),                                                          # expectation
              'u': np.sqrt( np.sum( np.power(y_pred - y_true, 2) ) / (y_pred.shape[0]-1) ),  # uncertainty
              'Rsq': reg.score(x_test, y_test),                                              # coefficient of determination
              'x': x_test * x.std(axis=0) + x.mean(axis=0), 
              'y_true': y_true, 
              'y_pred': y_pred, 
              'f': reg.predict}

    return output
