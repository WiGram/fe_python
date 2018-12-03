# -*- coding: utf-8 -*-
"""
Created on Mon Dec 03 11:05:02 2018

@author: wigr11ab
"""
import pandas as pd                    # Reads data
import numpy as np                     # Efficient programming
import scipy.optimize as opt           # Minimisation for MLE
import statsmodels.api as sm           # OLS estimation
import sys                             # Appending library of cuntions
sys.path.append("C:/Users/wigr11ab/Dropbox/KU/K3/FE/Python/")
import timeSeriesModule as tsm         # Import ARCH simulation
import score                           # Score module
import scipy.stats as ss               # Distribution functions
import plotFct as pltf                 # Custom plotting
import llf
np.set_printoptions(suppress = True)   # Disable scientific notation

# First exercise 1.5
x = np.arange(-5,5,0.01)
y = ss.norm.pdf(x)
z = ss.t.pdf(x, 3)

pltf.plotDuo(x, y, z, 'Standard Normal PDF', "Student's t (v=3)", '', 'Density', "Comparison between PDF's", loc = 'upper right')

tArch = tsm.tArchFct(1., 0.8, 3, 10000)

sig2    = 1.0
alpha   = 0.8
periods = 100

n = 10000
scores = np.zeros(n)
for i in np.arange(n):
    tArch = tsm.tArchFct(sig2, alpha, 3, periods)
    s     = score.tArch3(tArch)
    scores[i] = s

np.mean(scores)
np.var(scores)

x = np.arange(n)
pltf.plotUno(x, scores, 'Score value', 'Simulation trial', 'Distribution of scores', 'upper right')

x = np.sort(np.random.normal(size = n))
# sm.qqplot(scores, )
pltf.scatterUno(x, np.sort(scores), yLab = 'Emprical quantiles: Score', xLab='Theoretical quantiles', title='Normal QQ-Plot')

pltf.hist(scores, title = 'Histogram of scores')


# Next exercise 1.7
sp500 = pd.DataFrame(pd.read_excel('C:/Users/wigr11ab/Dropbox/KU/K3/FE/Exercises/SP500.xlsx'))
date    = np.array(sp500[['Date']][15096:])
returns = np.array(sp500[['log-ret_x100']][15096:])
pltf.plotUno(date, returns)

initPar = np.array([1.2, 0.2])
res = opt.minimize(llf.llfTArchSum, initPar, args = returns, method = 'L-BFGS-B')

# Next exercise 2.5
gjrArch = tsm.gjrArchFct(1., 0.5, 0.9, 100)
x = np.arange(len(gjrArch[0]))
pltf.plotUno(x, gjrArch[0])

gjrPar = np.array([0.8, 0.7, 0.9])
resGjr = opt.minimize(llf.llfGjrArchSum, gjrPar, args = gjrArch[0], method = 'L-BFGS-B')

# Conclusion: This solver is very unstable and not very efficient.

# Apply to SP500 data.
initPar = np.array([0.05, 0.5, 0.3])
resSp = opt.minimize(llf.llfGjrArchSum, initPar, args = returns, method = 'L-BFGS-B')