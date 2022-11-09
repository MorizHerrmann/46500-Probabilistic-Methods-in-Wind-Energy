# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 08:55:39 2022

@author: morit
"""

def MysteriousLoadFunc(X):
    import numpy as np
    Alsq = np.array([-8.49821625e+03,  3.21111941e+03, -3.44642937e+03, -8.33496535e+02,
       -2.28585769e+02,  7.27203251e+03,  9.25677112e+02,  1.16755478e+02,
        1.18918605e+03,  5.07954840e+00, -3.79944018e+03, -3.99715920e+01,
        7.18288327e+01, -1.82188802e+02, -2.61134584e+00, -2.53566886e+02,
        1.11035167e+03,  8.03450614e+00, -1.49376926e+02,  6.50965086e+00])

    X1 = X[:,0]
    X2 = X[:,1]
    X3 = X[:,2]
    X0 = np.ones(X1.shape)
    Xmatrix = np.array((X0,X1,X2,X3,X1**2,X2**2,X1*X2,X1*X3,X2*X3,X1**3,X2**3,(X1**2)*X2, (X1**2)*X3, \
                       (X2**2)*X1,(X2**2)*X3,X1*X2*X3, X2**4, (X1**2)*(X2**2), X2**5, X2**6)).T

    Y = np.dot(Xmatrix,Alsq) + 50e3 + np.random.randn(X0.shape[0])*4.0e3
    return Y

# Normal distribution
def NormalDist(task,x,mu=0,sigma=1):
    import numpy as np
    if task == 0: # PDF
        y = (1.0/(sigma*np.sqrt(2.0*np.pi)))*np.exp(-((x - mu)**2)/(2.0*(sigma**2)))
    elif task == 1: # Cumulative
        from scipy.special import erf
        y = 0.5*(1.0 + erf((x - mu)/(sigma*np.sqrt(2))))
    elif task == 2: # Inverse
        from scipy.special import erfinv
        y = mu + sigma*np.sqrt(2)*erfinv(2*x - 1)        
    return y


# Lognormal distribution
def LogNormDist(task,x,mu,sigma):
    import numpy as np
    Eps   = np.sqrt(np.log( 1.0+(sigma/mu)**2 ) )
    Ksi   = np.log(mu)-0.5*Eps**2
    if task == 0: # PDF
        x[x<=0] = 1e-8
        u =(np.log(x)-Ksi)/Eps
        y = np.exp(-u*u/2.0)/(Eps*x*np.sqrt(2.0*np.pi))
    elif task == 1: # Cummulative
        x[x<=0] = 1e-8
        u =(np.log(x)-Ksi)/Eps
        y= NormalDist(1, u)
    elif task == 2: # Inverse
        y= np.exp(Ksi+Eps*NormalDist(2, x))
    
    return y

# sample wind data =[mu, sigma, alpha] as joint distribution with rosenblatt transformation
def rosenblatt(N, Aweib=11.28, Kweib=2, Iref=0.14, c_alpha=0.088):
    import numpy as np
    import scipy.stats
    #--> for shear alpha
    MuAlphaFunc = lambda u: c_alpha*(np.log(u) - 1) #takes u, yields mu_alpha
    SigmaAlphaFunc = lambda u: 1/u

    #--> for turbulence sigma
    MuSigmaFunc = lambda u: Iref*(0.75*u + 3.8)
    SigmaSigmaU = 2.8*Iref

    #-->params for Weibull distribution of U
    Fu = np.random.rand(N) 
    u = scipy.stats.weibull_min.ppf(Fu,c = Kweib,scale = Aweib) # uses the weibull function to yield a large number of data

    # calculate params for other 2 distributions, mu and sigma each
    muSigma = MuSigmaFunc(u)
    sigmaSigma = SigmaSigmaU*np.ones(muSigma.shape)

    Fsigma = np.random.rand(N) # create new random sample
    sigmaU = LogNormDist(2,Fsigma,muSigma,sigmaSigma) #draw monte carlo samples from distibution

    muAlpha = MuAlphaFunc(u)
    sigmaAlpha = SigmaAlphaFunc(u)
    Falpha = np.random.rand(N)
    alpha = NormalDist(2,Falpha,muAlpha,sigmaAlpha)
    
    winddata = np.transpose(np.array([u,sigmaU,alpha]))
    
    return winddata

def model(x, y):
    import numpy as np
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split

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
              'f': lambda x: reg.predict((x - x.mean(axis=0)) / x.std(axis=0)) * y.std() + y.mean()}

    return output
