# Rosenblatt transformation

# In this exercise, we will get introduced to sampling from correlated (dependent) variables using the Rosenblatt transformation. 
# We consider 3 variables: wind speed $u$, turbulence $\sigma_u$, and wind shear exponent $\alpha$. The turbulence is considered conditionally dependent on the wind speed with distribution parameters: 

# $\mu(\sigma_u) (u)=I_{ref} (0.75u+3.8); \sigma_{\sigma_u} = 2.8 I_{ref} $ ; (Lognormal)
# where $I_{ref} = 0.14$ is a constant;

# and the wind shear explonent is also considered conditionally dependent on the wind speed:
# $\mu_\alpha (u)=0.088(log⁡(u)−1); \sigma_{\alpha} (u)=1/u $; (Normal)

import numpy as np
import scipy.stats as stats

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


# Helper functions - lognormal distribution
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
    
    #--> for shear alpha
    MuAlphaFunc = lambda u: c_alpha*(np.log(u) - 1) #takes u, yields mu_alpha
    SigmaAlphaFunc = lambda u: 1/u

    #--> for turbulence sigma
    MuSigmaFunc = lambda u: Iref*(0.75*u + 3.8)
    SigmaSigmaU = 2.8*Iref

    #-->params for Weibull distribution of U
    Fu = np.random.rand(N) 
    u = stats.weibull_min.ppf(Fu,c = Kweib,scale = Aweib) # uses the weibull function to yield a large number of data

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