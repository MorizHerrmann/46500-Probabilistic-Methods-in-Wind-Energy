# Tool to create slices from a ndarray, corresponding of a length of the slice

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
import keras
import pickle5 as pickle

def create_samples(df, n, in_features, target_feature):
    """
    function to split data into chunks on length n to train a lstm (where one input sample needs to contains a timeseries for each feature for each timestep of the area used
    IN:
        df: input dataframe with several features and the timeseris for each
        n: length of the datachunks
        in_features: features to use as input of the lstm
        target_feature: target feature of the prediction
    Out:
        feature_samples: list of samples for the input
        target_samples: list of samples for the output
        info: a dict containing info about the dataset, as the samples drop column names
    """
    data_len = len(df)
    subset = df[in_features + target_feature].to_numpy() # narrow down the df and turn into np
    info= {} # write usefull info that is dropped in the data
    info["in_features"]  = in_features
    info["target_feature"]  = target_feature
    
    # create samples
     

   


    feature_samples = [k]
    return feature_samples, target_samples, info

def create_dataset(dataset, prev_time=20): # take data in the 1h range, MAke target last ! 
    dataX, dataY = [], []
    for i in range(len(dataset)-prev_time-1):
        a = dataset[i:(i+prev_time), 0:-1] # takes only the last one as output
        b =dataset[i + prev_time, -1] 
       
        # append samples
        dataX.append(a) # so a is a sample
        dataY.append(b)
    return np.array(dataX), np.array(dataY)

if __name__== "__main__": 
    # Testing: Get a dataset, and split it into coherent sections using the function
    #1 get data
    #df = pd.read_pickle("../inter_results/traindata/T07_set3.pickle")
    paths = ["./inter_results/traindata/T01_set1.csv",
             "./inter_results/traindata/T06_set3.csv",
             "./inter_results/traindata/T06_set7.csv",
             "./inter_results/traindata/T07_set3.csv",
             "./inter_results/traindata/T09_set5.csv",
             "./inter_results/traindata/T11_set3.csv"
            ]
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
                     'Prod_LatestAvg_TotReactPwr']

    in_features = [
            'Grd_Prod_CurPhse1_Avg','Grd_Prod_CurPhse2_Avg', 'Grd_Prod_CurPhse3_Avg',
            'Gen_Phase1_Temp_Avg', 'Gen_Phase2_Temp_Avg', 'Gen_Phase3_Temp_Avg', 
            'Prod_LatestAvg_ActPwrGen1', 'Prod_LatestAvg_TotActPwr', 
            'Grd_Prod_PsblePwr_Avg']
    target_feature =['HVTrafo_Phase1_Temp_Avg']  
    # init some containers
    #x_train = np.empty((0,6,9))
    #y_train = np.empty((0,6,1))
    #x_test = np.empty((0,6,9))
    #y_test  = np.empty((0,6,1))
    
    for path in paths:
        breakpoint()
        df = pd.read_csv(path)
        n = 5


        df_in = df[in_features + target_feature] # combine!
        #df_target = df[target_feature]
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)) # scale data between 0 and 1. maybe use diffrent scaling ? 
        std_scaler = preprocessing.StandardScaler() # mean and std
        #dataset = std_scaler.fit_transform(df_in) # should be okay
        dataset = min_max_scaler.fit_transform(df_in) # should be okay
        #target = std_scaler.fit_transform(df_target) # now numpy array

        train_size = int(len(dataset) * 0.8) # train on 80 %
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

        print(len(train), len(test))

        x_train_n, y_train_n = create_dataset(train, prev_time=6) 
        x_test_n, y_test_n = create_dataset(test, prev_time=6) 
        breakpoint()
        #x_train = np.concatenate((x_train,x_train_n),axis=0)
        #y_train = np.concatenate((y_train,y_train_n),axis=0)
        #x_test = np.concatenate((x_test,x_test),axis=0)
        #y_test = np.concatenate((y_test,y_test),axis=0)
        if not 'x_train' in locals():
            x_train = x_train_n
            y_train = y_train_n
            x_test = x_test_n
            y_test = y_test_n
        else:
            x_train = np.concatenate((x_train,x_train_n),axis=0)
            y_train = np.concatenate((y_train,y_train_n),axis=0)
            x_test = np.concatenate((x_test,x_test),axis=0)
            y_test = np.concatenate((y_test,y_test),axis=0)

    breakpoint()
    print(f"Len traindata: {train.shape}")
    print(f"Shape X train: {x_train.shape}")
    print(f"Shape  Y train: {y_train.shape}")
    print(f"Shape X test: {x_test.shape}")
    print(f"Shape  Y test: {y_test.shape}")
    ### --------------> train a model! 

    model = Sequential()
    model.add(LSTM(32,activation='relu',return_sequences = True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(32,activation='relu'))
    #model.add(Dense(100))
    #model.add(Dense(100))
    #model.add(Dense(100))
    #model.add(Dense(100))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    model.fit(x_train, y_train, epochs=20)
    #model.summary()

    y_train_modeled = model.predict(x_train)
    y_test_modeled = model.predict(x_test)
    print(f"Shape of y_train_mod:{y_train_modeled.shape}") 
    print(f"Shape of y_test_mod:{y_test_modeled.shape}") 

    plt.figure(1) 
    plt.plot(y_train,label="real")
    plt.plot(y_train_modeled,label="model")
    plt.legend()

    print(y_train_modeled.shape)
    plt.figure(2)


    plt.plot(y_test,label="real")
    plt.plot(y_test_modeled,label="model")
    plt.legend()
    plt.show()
    #print(y_diff.shape)

    plt.figure(3) 
    plt.plot(y_test_modeled - y_test.reshape(-1,1))
    plt.show()

    results = model.evaluate(x_test, y_test)
    print(results)

