# -*- coding: utf-8 -*-
"""
DTU 46500 : Assignment 1
Fitting extreme distributions, uncertainty in fitting (including sample-length)

Created on Tue Sep 27 10:03:39 2022

@author: Moritz s213890
"""

# In[]: IMPORTS ###############################################################

import numpy as np
import datetime
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt

# In[]: LOAD DATA #############################################################

Data = np.genfromtxt('Hovsore2004to2020_U+dir.csv',delimiter=',')
Dnames = ['Timestamp','Wsp','MeanDir']

# create a datetime array so we can use ∆T for removing bad data
T = np.empty(Data[:,0].shape, dtype = datetime.datetime) # Empty numpy array of type datetime
for i in range(len(Data[:,0])):
    T[i] = datetime.datetime.strptime(str(Data[i,0].astype('int64')),'%Y%m%d%H%M')
DeltaT = np.diff(T)

# In[]: FILTER DATA ###########################################################

# limits
LowestWindSpeed = 0.
CrazyWind = 75.
maxDeltaU10min = 10.

WspOK = ((Data[:,1] > LowestWindSpeed)&(Data[:,1] < CrazyWind))

DeltaU = np.diff(Data[:,1])
dU10min = np.empty(DeltaU.shape,dtype = 'float')
for i in range(len(DeltaU)):
    dU10min[i] = DeltaU[i]/(DeltaT[i].seconds/600) 
DeltaUOK = (DeltaU > -maxDeltaU10min) & (DeltaU < maxDeltaU10min) 

OK1 = (WspOK[0:-1]==True) & (DeltaUOK == True)
OK2 = (WspOK[1:]==True) & (DeltaUOK == True)
OK1 = np.append(OK1,True)
OK2 = np.insert(OK2,0,True)

DataOK = OK1 & OK2

T = T[DataOK]
Timestamp = Data[DataOK,0]
Wsp = Data[DataOK,1]
wdir = Data[DataOK,2]

# Compute year and month
Year = np.floor(Timestamp/1e8).astype('int64')
Month = np.floor(np.mod(Timestamp,1e8)/1e6).astype('int64')
#Day = np.floor(np.mod(Timestamp,1e6)/1e4).astype('int64')
#Hour = np.floor(np.mod(Timestamp,1e4)/1e2).astype('int64')
#Minute = np.floor(np.mod(Timestamp,1e2)/1).astype('int64')

# plot the filtered data to check it:
plt.plot(T,Wsp)
plt.show()

del Data
del Dnames
del DeltaT
del CrazyWind
del LowestWindSpeed
del maxDeltaU10min
del WspOK
del DeltaU
del dU10min
del DeltaUOK
del DataOK
del T
del Timestamp

# In[]: COMPUTE ANNUAL MAXIMA #################################################
Years = np.unique(Year)
nyears = np.max(Years) - np.min(Years)

Umax = np.empty(nyears,dtype = 'float')
# since data starts in October, make each "year" start then
for i in range(nyears):
    CurrentPeriod = ((Year==Years[i]) & (Month >= 10)) | ((Year == (Years[i]+1)) & (Month < 10))
    Umax[i] = max(Wsp[CurrentPeriod]) 

# In[]: 1a) GUMBER FIT - METHOD OF PROBABILITY-WEIGHTET MOMENTS (PWM) #########

def Vgumbel(alpha, beta, T, T0=1):
    return beta - alpha * np.log(np.log(1 / (1- T0/T)))

def PWM(Umax, T0=1, T50=50, chunksize=0):
    """
    Calculate the extreme wind according to a gumbel distribution which is fit 
    to the data with the probability weighted moments method.

    Parameters
    ----------
    Umax : maxima for each base period (default yearly).
    T0 : base period. The default is 1.
    T50 : return period. The default is 50.

    Returns
    -------
    alpha : gumbel parameter.
    beta : gumbel parameter.
    Vxtr : extreme wind.

    """
    
    if chunksize:
        Umax = np.reshape(Umax, [-1, chunksize])
        
    n = Umax.shape[-1]
    
    # rank the yearly maxima (along last axis)
    UmaxSorted = np.sort(Umax)
    
    # compute moments
    b0 = np.sum(Umax, -1)/n
    b1 = np.sum(np.multiply(np.arange(0, n), UmaxSorted), -1) / n / (n-1)
    
    # compute gumbel parameters
    gamma = 0.577
    alpha = (2*b1 - b0) / np.log(2)
    beta = b0 - gamma * alpha
    
    Vxtr = Vgumbel(alpha, beta, T50, T0=T0)
    
    return alpha, beta, Vxtr, b0, b1
    
# In[]: PWM using the whole dataset

alpha, beta, V50, b0, b1 = PWM(Umax)

print('PWM: V50= '+str(V50.round(1))+' m/s; alpha='+str(alpha.round(2))+', beta='+str(beta.round(1))+' m/s')

T = np.arange(start=1, stop=50)
Vxtr = Vgumbel(alpha, beta, T)

# In[]: PWM using 8 different 2-year chunks

alpha2y, beta2y, V502y, b02y, b12y = PWM(Umax, chunksize=2)
Vxtr2y = np.empty((len(T), len(alpha2y)))
for i in range(len(alpha2y)):
    Vxtr2y[:, i] = Vgumbel(alpha2y[i], beta2y[i], T)


# In[]: Comparison

plt.figure()
plt.title('Comparison of data with model')
plt.semilogx(np.sort(Umax), 'o', label='ranked yearly maxima')
plt.semilogx(Vxtr, label='Gumbel-PWM of whole data')
plt.semilogx(Vxtr2y, 'k', alpha=0.3, label='Gumbel-PWM of 2y data')

plt.grid()
plt.xlabel('return period [years]')
plt.ylabel('Extreme wind [m/s]')
plt.show()
