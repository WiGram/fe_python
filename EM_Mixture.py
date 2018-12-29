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

# ============================================= #
# ===== Load S&P 500 data ===================== #
# ============================================= #

sp500   = pd.DataFrame(pd.read_excel('C:/Users/wigr11ab/Dropbox/KU/K3/FE/Exercises/SP500.xlsx'))
date    = np.array(sp500[['Date']][15096:])
returns = np.array(sp500[['log-ret_x100']][15096:])

# ============================================= #
# ===== Parameterisation ====================== #
# ============================================= #

mat   = len(returns)
pStar = np.zeros(mat)
rep   = 500
par   = np.zeros(3 * rep).reshape(3, rep)
llh   = np.zeros(rep)

# ============================================= #
# ===== EM Algorithm, iid states ============== #
# ============================================= #

# Further parameterisation
y = returns
s1, s2, p = 1.0, 0.5, 0.3

# Compute initial pStar given initial parameters
f1    = 1 / np.sqrt(2 * np.pi * s1) * np.exp(- 0.5 * y ** 2 / s1)
f2    = 1 / np.sqrt(2 * np.pi * s2) * np.exp(- 0.5 * y ** 2 / s2)
pStar = p * f1 / (p * f1 + (1-p) * f2)

for m in range(rep):
    # Reevaluate parameters given pStar
    s1 = sum(pStar * y ** 2) / sum(pStar)
    s2 = sum((1 - pStar) * y ** 2) / sum(1 - pStar)
    p  = sum(pStar) / mat

    # Update pStar given new parameters
    f1     = 1 / np.sqrt(2 * np.pi * s1) * np.exp(- 0.5 * y ** 2 / s1)
    f2     = 1 / np.sqrt(2 * np.pi * s2) * np.exp(- 0.5 * y ** 2 / s2)
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

