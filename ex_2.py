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
from ar_function import ar_fct         # Import time series simulation
from llf import llfAr, llfArSum     # Import llh contributions and fct.
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

ar = ar_fct(*theta, periods)
plt.plot(ar)
plt.show()

# Finding parameters
par = [5., 0.05, 1.]
ar_opt = opt.minimize(llfArSum, par, args = ar, method = 'L-BFGS-B')

# ============================================= #
# ===== Applying llh function to actual data == #
# ============================================= #

# Finding parameters
spread_opt = opt.minimize(llfArSum, par, args = spread, method = 'L-BFGS-B')
est_par = spread_opt.x

# Calculate standard errors
h_fct = nd.Hessian(llfArSum)
hess  = np.linalg.inv(h_fct(est_par, spread))
se    = np.sqrt(np.diag(hess))

jac_fct = nd.Jacobian(llfAr)
jac     = jac_fct(est_par, spread)
jac     = np.transpose(jac.reshape(366,3))
score   = np.inner(jac, jac)

robust_se = np.sqrt(np.diag(hess.dot(score).dot(hess)))
robust_se

# Simulate new series
ar_new = ar_fct(*est_par, periods*4) # Increased length to identify stationarity
plt.plot(ar_new)
plt.show()

# Consider OLS regression
model = sm.OLS(spread[1:], sm.add_constant(spread[0:periods - 1]))
results = model.fit()
mu, rho = results.params

residuals = spread[1:] - mu - rho * spread[0:periods- 1]
sig = np.std(residuals)
ols_par = np.array([mu, rho, sig])

# Compare parameters from MLE with OLS
ols_par
est_par
