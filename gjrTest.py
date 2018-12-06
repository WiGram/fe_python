# -*- coding: utf-8 -*-
"""
Created on Mon Dec 03 11:05:02 2018

@author: wigr11ab
"""
import pandas as pd                    # Reads data
import numpy as np                     # Efficient programming
import numdifftools as nd              # Finding derivatives
import scipy.optimize as opt           # Minimisation for MLE
import statsmodels.api as sm           # OLS estimation
import sys                             # Appending library of cuntions
sys.path.append("C:/Users/wigr11ab/Dropbox/KU/K3/FE/Python/")
import timeSeriesModule as tsm         # Import ARCH simulation
import score                           # Score module
import scipy.stats as ss               # Distribution functions
import plotsModule as pltm             # Custom plotting
import likelihoodModule as llm         # Likelihood functions
np.set_printoptions(suppress = True)   # Disable scientific notation

# Next exercise 1.7
sp500   = pd.DataFrame(pd.read_excel('C:/Users/wigr11ab/Dropbox/KU/K3/FE/Exercises/SP500.xlsx'))
date    = np.array(sp500[['Date']][15096:])
returns = np.array(sp500[['log-ret_x100']][15096:])

def llfGjrArch(theta, y):
    if len(theta) != 3:
        return 'Parameter must have dimension 3.'
    end = len(y)
    lSigma2, lAlpha, lGamma = theta
    
    idx = (y < 0)
    s2     = np.exp(lSigma2) + np.exp(lAlpha) * y[:end - 1] ** 2 + np.exp(lGamma) * idx[:end - 1] * y[:end - 1] ** 2
    log_s2 = np.log(s2)

    return -0.5 * (np.log(2 * np.pi) + log_s2 + y[1:] ** 2 / s2)

def llfGjrArchSum(theta, y):
    return -sum(llfGjrArch(theta, y))

initPar = np.array([1.0, 2.0, 2.0])  # sigma^2, alpha, gamma
resGjr  = opt.minimize(llfGjrArchSum, initPar, args = returns, method = 'L-BFGS-B')
gjrPar  = np.exp(resGjr.x)
mlVal   = llfGjrArchSum(gjrPar, returns)

# Standard error calculation
hFct = nd.Hessian(llfGjrArchSum)
hess = np.linalg.inv(hFct(gjrPar, returns))
se   = np.sqrt(np.diag(hess))
tVal = gjrPar / se

jFct  = nd.Jacobian(llfGjrArch)
jac   = jFct(gjrPar, returns)
jac   = np.transpose(np.squeeze(jac, axis=0)) # Squeez removes a redundant dimension.
score = np.inner(jac, jac)

seRobust   = np.sqrt(np.diag(hess.dot(score).dot(hess)))
tValRobust = gjrPar / seRobust

mlResults = pd.DataFrame([gjrPar, se, seRobust, tVal, tValRobust, mlVal], \
                         columns=['sigma2', 'alpha', 'gamma'], \
                         index=['estimate', 'se', 'robust se', 't-val', 'robust t-val', 'ml val'])
mlResults