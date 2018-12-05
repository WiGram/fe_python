# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 19:54:41 2018

@author: wigr11ab
"""
import pandas as pd                    # Reads data
import numpy as np                     # Efficient programming
from matplotlib import pyplot as plt   # Plotting
import numdifftools as nd              # Finding derivatives
import scipy.optimize as opt           # Minimisation for MLE
import statsmodels.api as sm
import sys                             # Appending library of cuntions
sys.path.append("C:/Users/wigr11ab/Dropbox/KU/K3/FE/Python/")
import timeSeriesModule as tsm         # Import time series simulation
from likelihoodModule import llfAr, llfArSum     # Import llh contributions and fct.
np.set_printoptions(suppress = True)   #disable scientific notation

# ============================================= #
# ===== Load data and visualise time series === #
# ============================================= #
spread = pd.read_excel('C:/Users/wigr11ab/Dropbox/KU/K3/FE/Exercises/USspreadraw.xls')
spread = np.array(spread)
plt.plot(spread)
plt.show()

# ============================================= #
# ===== Simulate AR time series to test MLE === #
# ============================================= #

periods = len(spread)
theta = [1.0, 0.5, 2.0]

ar = tsm.arFct(*theta, periods)
plt.plot(ar[0])
plt.show()

# Finding parameters
par = [2., 0.05, 1.]
ar_opt = opt.minimize(llfArSum, par, args = ar[0], method = 'L-BFGS-B')

# ============================================= #
# ===== Applying llh function to actual data == #
# ============================================= #

# Finding parameters
spread_opt = opt.minimize(llfArSum, par, args = spread, method = 'L-BFGS-B')
estPar = spread_opt.x

# Calculate standard errors
hFct = nd.Hessian(llfArSum)
hess  = np.linalg.inv(hFct(estPar, spread))
se    = np.sqrt(np.diag(hess))

jac_fct = nd.Jacobian(llfAr)
jac     = jac_fct(estPar, spread)
jac     = np.transpose(np.squeeze(jac, axis=0)) # Squeez removes a redundant dimension.
score   = np.inner(jac, jac)

robustSe = np.sqrt(np.diag(hess.dot(score).dot(hess)))
robustSe

# Simulate new series
arNew = tsm.arFct(*estPar, periods*4) # Increased length to identify stationarity
plt.plot(arNew)
plt.show()

# Consider OLS regression
model = sm.OLS(spread[1:], sm.add_constant(spread[0:periods - 1]))
results = model.fit()
mu, rho = results.params

residuals = spread[1:] - mu - rho * spread[0:periods- 1]
sig = np.std(residuals)
olsPar = np.array([mu, rho, sig])

# Compare parameters from MLE with OLS
olsPar
estPar
