# -*- coding: utf-8 -*-
"""
Created on Sun Dec 09 15:08:51 2018

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
import scoreModule as scm              # Score module
import scipy.stats as ss               # Distribution functions
import plotsModule as pltm             # Custom plotting
import likelihoodModule as llm         # Likelihood functions
np.set_printoptions(suppress = True)   # Disable scientific notation

sp500 = pd.DataFrame(pd.read_excel('C:/Users/wigr11ab/Dropbox/KU/K3/FE/Exercises/SP500.xlsx'))
date  = np.array(sp500[['Date']][15096:])
y     = np.array(sp500[['log-ret_x100']][15096:]) # Returns

# Exercise 6.1.5
initPar = np.array([1.0, 1.0, 1.0]) # sig2, alphaP, alphaN
resA    = opt.minimize(llm.llfAArchSum, initPar, args = y, method = 'L-BFGS-B')

# Parameters
aPar    = resA.x

# ML Value
mlVal    = llm.llfAArchSum(aPar, y)

# Information matrix
hFct = nd.Hessian(llm.llfAArchSum)
hess = np.linalg.inv(hFct(aPar, y))

# Score
jFct  = nd.Jacobian(llm.llfAArch)
jac   = jFct(aPar, y)
jac   = np.transpose(np.squeeze(jac, axis=0)) # Squeez removes a redundant dimension.
score = np.inner(jac, jac)

# Sandwich standard errors
sandwich = hess.dot(score).dot(hess)

# Back transform standard errors
jacA = nd.Jacobian(np.exp)
A    = jacA(aPar)

# Applying the delta method to transformed parameters to
# find se for non-transformed parameters
se = np.sqrt(np.diag(A.dot(sandwich).dot(np.transpose(A))))
tVal = np.exp(aPar) / se

mlResults = pd.DataFrame([np.exp(aPar), se, tVal, mlVal], \
                         columns=['sigma2', 'alphaP', 'alphaN'], \
                         index=['estimate', 'se', 't-val', 'ml val'])
mlResults

# Analysing the residuals z = x / sigma
theta = np.exp(aPar)
n  = len(y)

x  = np.squeeze(y)
xLag2 = x[:n - 1] ** 2

idxP = (x > 0)[:n - 1]
idxN = (x < 0)[:n - 1]

s2 = theta[0] + theta[1] * idxP * xLag2 + theta[2] * idxN * xLag2
z  = x[1:] / s2

pltm.qqPlot(z)
pltm.hist(z)
pltm.plotUno(np.arange(n-1), z, yLab='Residuals')

# Use the delta method to find se for GJR through AArch parameters (done manually)
a    = np.array([np.exp(aPar[0]), np.exp(aPar[1]), np.exp(aPar[2]) - np.exp(aPar[1])])
A    = np.array([[a[0], 0, 0], \
                 [0, a[1], 0], \
                 [0, -a[1], a[2] + a[1]]])
se = np.sqrt(np.diag(A.dot(sandwich).dot(np.transpose(A))))
tVal = a / se

gjrResults = pd.DataFrame([a, se, tVal, mlVal], \
                           columns=['sigma2', 'alpha', 'gamma'], \
                           index=['estimate', 'se', 't-val', 'ml val'])
gjrResults
mlResults