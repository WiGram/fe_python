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
import scoreModule as score            # Score module
import scipy.stats as ss               # Distribution functions
import plotsModule as pltm             # Custom plotting
import likelihoodModule as llm         # Likelihood functions
np.set_printoptions(suppress = True)   # Disable scientific notation

# Next exercise 1.7
sp500   = pd.DataFrame(pd.read_excel('C:/Users/wigr11ab/Dropbox/KU/K3/FE/Exercises/SP500.xlsx'))
date    = np.array(sp500[['Date']][15096:])
returns = np.array(sp500[['log-ret_x100']][15096:])

def llfGjrArch(theta, y):
    lSigma2, lAlpha, lGamma = theta
    end = len(y)
    
    s2     = np.exp(lSigma2) + np.exp(lAlpha) * y[:end - 1] ** 2 + np.exp(lGamma) * (y < 0)[:end - 1] * y[:end - 1] ** 2
    log_s2 = np.log(s2)

    return -0.5 * (np.log(2 * np.pi) + log_s2 + y[1:] ** 2 / s2)

def llfGjrArchSum(theta, y):
    return -sum(llfGjrArch(theta, y))

def seGjrArch(theta, y):
    sig2, alpha, gamma = theta
    end = len(y)

    s2     = sig2 + alpha * y[:end - 1] ** 2 + gamma * (y < 0)[:end - 1] * y[:end - 1] ** 2
    log_s2 = np.log(s2)

    return -0.5 * (np.log(2 * np.pi) + log_s2 + y[1:] ** 2 / s2)

def seGjrArchSum(theta, y):
    return - sum(seGjrArch(theta, y))

initPar = np.array([2.0, 2.0, 2.0])  # sigma^2, alpha, gamma
resGjr  = opt.minimize(llfGjrArchSum, initPar, args = returns, method = 'L-BFGS-B')
gjrPar  = resGjr.x
mlVal   = llfGjrArchSum(gjrPar, returns)

# Standard errors by Delta-method

# Calculate the information (2nd deriv)
hFct = nd.Hessian(llfGjrArchSum)
hess = np.linalg.inv(hFct(gjrPar, returns))

# Calculate the score (1st derivative)
jFct  = nd.Jacobian(llfGjrArch)
jac   = jFct(gjrPar, returns)
jac   = np.transpose(np.squeeze(jac, axis = 0))
score = np.inner(jac, jac)

# Calculate the Sandwich se
Sandwich = hess.dot(score).dot(hess)

# Calculate derivatives matrix
jacA = nd.Jacobian(np.exp)
A    = jacA(gjrPar)

# Calculate the SE belonging to non-transformed data
se   = np.sqrt(np.diag(A.dot(Sandwich).dot(np.transpose(A))))
tVal = np.exp(gjrPar) / se

mlResults = pd.DataFrame([np.exp(gjrPar), se, tVal, mlVal], \
                         columns=['sigma2', 'alpha', 'gamma'], \
                         index=['estimate', 'se', 't-val', 'ml val'])
mlResults

# Calculate A analytically or numerically
A    = np.array([[np.exp(gjrPar[0]), 0, 0],
                 [0, np.exp(gjrPar[1]), 0],
                 [0, 0, np.exp(gjrPar[2])]])

# Crude method
# Standard error calculation
hFct = nd.Hessian(seGjrArchSum)
hess = np.linalg.inv(hFct(gjrPar, returns))
se   = np.sqrt(np.diag(hess))
tVal = gjrPar / se

jFct  = nd.Jacobian(seGjrArch)
jac   = jFct(gjrPar, returns)
jac   = np.transpose(np.squeeze(jac, axis=0)) # Squeez removes a redundant dimension.
score = np.inner(jac, jac)

seRobust   = np.sqrt(np.diag(hess.dot(score).dot(hess)))
tValRobust = gjrPar / seRobust

mlcResults = pd.DataFrame([gjrPar, se, seRobust, tVal, tValRobust, mlVal], \
                         columns=['sigma2', 'alpha', 'gamma'], \
                         index=['estimate', 'se', 'robust se', 't-val', 'robust t-val', 'ml val'])
mlcResults