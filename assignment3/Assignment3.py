# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 08:55:16 2022

@author: morit
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn
import sklearn.neural_network
import scipy.optimize
import scipy.stats
from toolbox import MysteriousLoadFunc, rosenblatt, LogNormDist, NormalDist, model

#%% Preparation
# Number of samples
N = 10000

# sample wind data
wind = rosenblatt(N)

# Material strength X_R
mu_XR = 90e3 
s_XR = 4.5e3
F_XR = np.random.rand(N)
X_XR = LogNormDist(2, F_XR, mu_XR, s_XR)

X_R_0 = stats.norm.ppf(F_XR, loc = 0, scale = 1);

# Stress calculation method uncertainty X_STR
mu_STR = 1 
s_STR = 0.03
F_STR = np.random.rand(N)
X_STR = LogNormDist(2, F_STR, mu_STR, s_STR)

X_STR_0 = stats.norm.ppf(F_STR, loc = 0, scale = 1);

# Blade geometry uncertainty X_GEOM
mu_GEOM = 1 
s_GEOM = 0.03
F_GEOM = np.random.rand(N)
X_GEOM = LogNormDist(2, F_GEOM, mu_GEOM, s_GEOM)

X_GEOM_0 = stats.norm.ppf(F_GEOM, loc = 0, scale = 1);

# Data
X = np.vstack((wind.T,X_XR,X_STR,X_GEOM))

alpha = 0.5


#%% Calculate load with MysteriousLoadFunc
S_mlf = MysteriousLoadFunc(wind)
R = X_XR*X_GEOM*X_STR;
G_mlf = R - S_mlf
    
fail_mlf = G_mlf < 0

# plot limit state function
fig, axs = plt.subplots(1, 3, figsize=(12, 5))
axs[0].scatter(wind[:, 0], wind[:, 1], alpha=alpha, c=G_mlf)
axs[0].scatter(wind[fail_mlf, 0], wind[fail_mlf, 1], alpha=alpha, edgecolor='r', linewidth=1)
axs[0].set_xlabel('u in m/s')
axs[0].set_ylabel(r'$\sigma$ in m/s')

axs[1].scatter(wind[:, 1], wind[:, 2], alpha=alpha, c=G_mlf)
axs[1].scatter(wind[fail_mlf, 1], wind[fail_mlf, 2], alpha=alpha, edgecolor='r', linewidth=1)
axs[1].set_xlabel(r'$\sigma$ in m/s')
axs[1].set_ylabel(r'$\alpha$')

axs[2].scatter(wind[:, 2], wind[:, 0], alpha=alpha, c=G_mlf)
s=axs[2].scatter(wind[fail_mlf, 2], wind[fail_mlf, 0], alpha=alpha, edgecolor='r', linewidth=1)
axs[2].set_xlabel(r'$\alpha$')
axs[2].set_ylabel('u in m/s')

s.set_clim(G_mlf.min(), G_mlf.max())
fig.colorbar(s)
plt.suptitle("Limit state function g(X) [kNm]")
plt.tight_layout()
plt.savefig('g.pdf')
plt.savefig('g.png')


# plot failure surface
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(wind[~fail_mlf, 0], wind[~fail_mlf, 1], 'g.', alpha=alpha)
axs[0].plot(wind[fail_mlf, 0], wind[fail_mlf, 1], 'r.', alpha=alpha)
axs[0].set_xlabel('u in m/s')
axs[0].set_ylabel(r'$\sigma$ in m/s')

axs[1].plot(wind[~fail_mlf, 1], wind[~fail_mlf, 2], 'g.', alpha=alpha)
axs[1].plot(wind[fail_mlf, 1], wind[fail_mlf, 2], 'r.', alpha=alpha)
axs[1].set_xlabel(r'$\sigma$ in m/s')
axs[1].set_ylabel(r'$\alpha$')

axs[2].plot(wind[~fail_mlf, 2], wind[~fail_mlf, 0], 'g.', alpha=alpha)
axs[2].plot(wind[fail_mlf, 2], wind[fail_mlf, 0], 'r.', alpha=alpha)
axs[2].set_xlabel(r'$\alpha$')
axs[2].set_ylabel('u in m/s')

plt.suptitle("Failure Surface with MysteriousLoadFunc")
plt.tight_layout()

p_mlf = fail_mlf.mean()
beta_mlf = scipy.stats.norm.ppf(1 - p_mlf)


#%% Calculate load with surrogate model
surrogate = model(wind, S_mlf)
S_sur = surrogate["f"](wind)
G_sur = R - S_sur 
fail_sur = G_sur < 0

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(wind[fail_sur, 0], wind[fail_sur, 1], 'r.', alpha=alpha)
axs[0].plot(wind[~fail_sur, 0], wind[~fail_sur, 1], 'g.', alpha=alpha)
axs[0].set_xlabel('u in m/s')
axs[0].set_ylabel(r'$\sigma$ in m/s')

axs[1].plot(wind[fail_sur, 1], wind[fail_sur, 2], 'r.', alpha=alpha)
axs[1].plot(wind[~fail_sur, 1], wind[~fail_sur, 2], 'g.', alpha=alpha)
axs[1].set_xlabel(r'$\sigma$ in m/s')
axs[1].set_ylabel(r'$\alpha$')

axs[2].plot(wind[fail_sur, 2], wind[fail_sur, 0], 'r.', alpha=alpha)
axs[2].plot(wind[~fail_sur, 2], wind[~fail_sur, 0], 'g.', alpha=alpha)
axs[2].set_xlabel(r'$\alpha$')
axs[2].set_ylabel('u in m/s')

plt.suptitle("Failure Surface with surrogate model")
plt.tight_layout()

p_sur = fail_sur.mean()
beta_sur = scipy.stats.norm.ppf(1 - p_sur)

#%% Calculate surrogate model for g
g_model = model(wind, G_mlf)
G_grg = g_model["f"](wind)
fail_grg = G_grg < 0

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(wind[fail_grg, 0], wind[fail_grg, 1], 'r.', alpha=alpha)
axs[0].plot(wind[~fail_grg, 0], wind[~fail_grg, 1], 'g.', alpha=alpha)
axs[0].set_xlabel('u in m/s')
axs[0].set_ylabel(r'$\sigma$ in m/s')

axs[1].plot(wind[fail_grg, 1], wind[fail_grg, 2], 'r.', alpha=alpha)
axs[1].plot(wind[~fail_grg, 1], wind[~fail_grg, 2], 'g.', alpha=alpha)
axs[1].set_xlabel(r'$\sigma$ in m/s')
axs[1].set_ylabel(r'$\alpha$')

axs[2].plot(wind[fail_grg, 2], wind[fail_grg, 0], 'r.', alpha=alpha)
axs[2].plot(wind[~fail_grg, 2], wind[~fail_grg, 0], 'g.', alpha=alpha)
axs[2].set_xlabel(r'$\alpha$')
axs[2].set_ylabel('u in m/s')

plt.suptitle("Failure Surface with surrogate model for g")
plt.tight_layout()

p_grg = fail_grg.mean()
beta_grg = scipy.stats.norm.ppf(1 - p_grg)
