# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:13:35 2018

@author: WiGram
"""

import likelihoodModule as llm
import plotsModule as pltm
import numpy as np
import pandas as pd
import scipy.optimize as opt
np.set_printoptions(suppress = True)   # Disable scientific notation

# 0. Load S&P 500 data

sp500   = pd.DataFrame(pd.read_excel('C:/Users/wigr11ab/Dropbox/KU/K3/FE/Exercises/SP500.xlsx'))
returns = np.array(sp500['log-ret_x100'][15096:])

# 1. Set initial parameters
y        = returns
mat      = len(y)
states   = 2
s1, s2   = 1.0, 0.5

f1    = 1 / np.sqrt(2 * np.pi * s1) * np.exp(- 0.5 * y ** 2 / s1)
f2    = 1 / np.sqrt(2 * np.pi * s2) * np.exp(- 0.5 * y ** 2 / s2)

p11, p22 = 0.5, 0.5
p12, p21 = 1.0 - p11, 1.0 - p22

# 2. Forward - Backward algorithm
# A. Forward algorithm
a   = (f1[0] / mat, f2[0] / mat)
a   = np.repeat(a, mat).reshape(states, mat)
a_s = np.ones(mat)                               # a_scale
a_r = np.ones(states * mat).reshape(states, mat) # a_rescale

# t = 0
a_s[0]    = sum(a)
a_r[:, 0] = a / a_s[0]

# t in [1, T]
for t in range(1, mat):
    a[0, t]   = sum(f1[t] * [p11, p21] * a_r[:, t-1])
    a[1, t]   = sum(f2[t] * [p12, p22] * a_r[:, t-1])
    a_s[t]    = sum(a[:, t])
    a_r[:, t] = a[:,t] / a_s[t]

# B. Backward algorithm
b   = np.ones(states * mat).reshape(states, mat)
b_s = np.ones(mat)                               # b_scale
b_r = np.ones(states * mat).reshape(states, mat) # b_rescale

# t = T (= mat - 1)
b_s[mat-1]      = sum(b[:, mat - 1])
b_r[:, mat - 1] = b[:, mat - 1] / b_s[mat - 1]

# t in [0, T - 1] (= mat - 2, stops at previous index, i.e. 0)
for t in range(mat - 2, -1, -1):
    b[0, t]   = sum(b_r[0, t+1] * f1[t+1] * p11, b_r[1, t+1] * f2[t+1] * p12)
    b[1, t]   = sum(b_r[0, t+1] * f1[t+1] * p21, b_r[1, t+1] * f2[t+1] * p22)
    b_s[t]    = sum(b[:,t])
    b_r[:, t] = b[:, t] / b_s[t]

# C. Probabilities
# C.1 Smoothed probabilities
pStar  = np.ones(states * mat).reshape(states, mat)
denom = b_r[0,:] * a_r[0,:] + b_r[1,:] * a_r[1,:]
for s in range(states):
    pStar[s, :] = (b_r[s, :] * a_r[s, :]) / denom

# C.2 Smoothed TRANSITION probabilities (these are joint probabilities, not conditional)
pStarT = np.ones(states * states *mat).reshape(states * states, mat)

denom = denom * a_s
pStarT[:, 0] = np.array([p11, p12, p21, p22]) / 2 # Assume the first period is given
for t in range(1, mat):
    pStarT[0, t] = b_r[0, t] * f1[t] * p11 * a_r[0, t-1] / denom[t]
    pStarT[1, t] = b_r[1, t] * f2[t] * p12 * a_r[0, t-1] / denom[t]
    pStarT[2, t] = b_r[0, t] * f1[t] * p21 * a_r[1, t-1] / denom[t]
    pStarT[3, t] = b_r[1, t] * f2[t] * p22 * a_r[1, t-1] / denom[t]

# Initial checks that all is OK
'OK: Minimum is weakly larger than 0' if min([min(pStar[i,:]) >= 0.0 for i in range(states)])                     else 'Error: Minimum is less than zero'
'OK: Maximum is weakly less than 1'   if min([max(pStar[i,:]) <= 1.0 for i in range(states)])                     else 'Error: Maximum is larger than 1'
'OK: Probabilities sum to 1'          if min(np.round(pStar[0,:] + pStar[1,:], 10) == 1.0)                        else "Error: Probabilities don't sum to 1"
'OK: Minimum is weakly larger than 0' if min([min(pStarT[i,:]) >= 0.0 for i in range(states ** 2)])               else 'Error: Minimum is less than zero'
'OK: Maximum is weakly less than 1'   if min([max(pStarT[i,:]) <= 1.0 for i in range(states ** 2)])               else 'Error: Maximum is larger than 1'
'OK: Probabilities sum to 1'          if min(np.round(sum([pStarT[i,:] for i in range(states ** 2)]), 10) == 1.0) else "Error: Probabilities don't sum to 1"


# 3. EM-loop until convergence (we attempt at rep repeats)
for m in range(rep):
    # Reevaluate parameters given pStar
    s1  = sum(pStar[0, :] * y ** 2) / sum(pStar[0,:])
    s2  = sum(pStar[1, :] * y ** 2) / sum(pStar[1,:])
    
    # New densities
    f1     = 1 / np.sqrt(2 * np.pi * s1) * np.exp(- 0.5 * y ** 2 / s1)
    f2     = 1 / np.sqrt(2 * np.pi * s2) * np.exp(- 0.5 * y ** 2 / s2)
    
    # New Steady state probabilities
    p11 = sum(pStarT[0,:]) / sum(pStarT[0,:] + pStarT[1,:])
    p22 = sum(pStarT[3,:]) / sum(pStarT[2,:] + pStarT[3,:])
    p12, p21 = 1 - p11, 1 - p22
    
    # New smoothed probabilities
    pStar  = p * f1 / (p * f1 + (1-p) * f2) # Compute new pStar

    # Compute the log-likelihood to maximise
    logLik = pStar * np.log(f1 * p) + (1 - pStar) * np.log(f2 * (1 - p))
    sVol   = pStar * s1 + (1 - pStar) * s2

    # Save parameters for later plotting (redundant wrt optimisation)
    par[0,m], par[1,m], par[2,m] = s1, s2, p
    llh[m] = sum(logLik)

pltm.plotDuo(range(rep), par[0,:], par[1,:], 'Sigma_h', 'Sigma_l', 'Time', 'Volatility')
pltm.plotUno(range(rep), par[2,:])
pltm.plotUno(range(rep), llh)

