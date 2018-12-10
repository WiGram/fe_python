# -*- coding: utf-8 -*-
"""
Created on Sun Mon 10 12:47:47 2018

@author: wigr11ab
"""

# import pandas as pd                    # Reads data
import numpy as np                     # Efficient programming
# import numdifftools as nd              # Finding derivatives
# import scipy.optimize as opt           # Minimisation for MLE
# import statsmodels.api as sm           # OLS estimation
import sys                             # Appending library of cuntions
sys.path.append("C:/Users/wigr11ab/Dropbox/KU/K3/FE/Python/")
import timeSeriesModule as tsm         # Import ARCH simulation
# import scoreModule as scm              # Score module
import scipy.stats as ss               # Distribution functions
import plotsModule as pltm             # Custom plotting
import likelihoodModule as llm         # Likelihood functions
from matplotlib import pyplot as plt   # Bespoke plotting
np.set_printoptions(suppress = True)   # Disable scientific notation

# ============================================= #
# ===== Exercise 1 ============================ #
# ============================================= #
" 1.3 Mixture model "
mat = 10 ** 3
s_h = 4.0 
s_l = 0.1
p   = 0.5

z = np.random.normal(0,1, size = mat)
u = np.random.uniform(0, 1, size = mat)

state   = (u > p) * 1 + (u < p) * 0
returns = ((u > p) * s_h + (u < p) * s_l) * z

x = np.arange(0,mat)
pltm.plotDuo(x = x, y1 = returns, y2 = state, \
            yLab1 = 'Returns', yLab2 = 'State', \
            yLab = 'Returns', xLab = 'Time')

" 1.4 Comparison to theoretical distribution"
def normPdf(z, mu, sig2):
    return 1 / np.sqrt(2 * np.pi * sig2) * np.exp( - 0.5 * (z - mu) ** 2 / sig2)

def mixturePdf(z, mu, sig12, sig22):
    return 0.5 * (normPdf(z, mu, sig12) + normPdf(z, mu, sig22))

x = np.arange(min(returns), max(returns), 1 / mat)
y = mixturePdf(x, 0, s_h ** 2, s_l ** 2)

# Plotting the returns density against the theoretical density
fig, ax = plt.subplots(figsize = (8,5))
ax.plot(x, y, label = 'Theoretical density')
ax.hist(returns, density = True, bins = 500, label = 'Returns density')
ax.legend(loc = 'upper right', shadow = False)
plt.show()

# ============================================= #
# ===== Exercise 2 ============================ #
# ============================================= #
" Two-state hidden Markov volatility model "

p11 = 0.95
p22 = 0.90

state_ms = np.repeat(0, mat)
for t in range(1,mat):
    if state_ms[t-1] == 0:
        state_ms[t] = (u[t] < p11) * 0 + (u[t] > p11) * 1
    else:
        state_ms[t] = (u[t] < p22) * 1 + (u[t] > p22) * 0

x = np.arange(0,mat)
returns_ms = (state_ms * s_h + (1 - state_ms) * s_l) * z
pltm.plotDuo(x = x, y1 = returns_ms, y2 = state_ms, \
            yLab1 = 'Returns', yLab2 = 'State', \
            yLab = 'Returns', xLab = 'Time')