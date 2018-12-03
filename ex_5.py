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
import scipy.stats as ss               # Distribution functions
import plotFct as pltf                 # Custom plotting
np.set_printoptions(suppress = True)   # Disable scientific notation

# First exercise 1.5
x = np.arange(-5,5,0.01)
y = ss.norm.pdf(x)
z = ss.t.pdf(x, 3)

pltf.plotDuo(x, y, z, 'Standard Normal PDF', "Student's t (v=3)", '', 'Density', "Comparison between PDF's", loc = 'upper right')

def tArchFct(sig2, alpha, periods):
    z  = np.sqrt(1 / 3) * np.random.standard_t(3, periods)
    x  = np.zeros(periods)
    s2 = np.ones(periods)

    for t in np.arange(1, periods):
        s2[t] = sig2 + alpha * x[t-1] ** 2
        x[t]  = np.sqrt(s2[t]) * z[t]
    
    return np.array([z, x, s2])

tArch = tArchFct(1., 0.8, 100)

def score(ts):
    periods = len(ts[0])
    y = (3. * ts[0][1:] ** 2 - 1.) / (1. + ts[0][1:] ** 2)
    u = ts[1][:periods - 1] ** 2 / ts[2][1:]

    return 1 / np.sqrt(periods - 1) * sum(y * u)

def score_ols(ts):
    periods = len(ts[0])
    y = (3. * ts[0][1:] ** 2 - 1.) / (1. + ts[0][1:] ** 2)
    u = ts[1][:periods - 1] ** 2 / ts[2][1:]

    Y = (ts[1] ** 2) - np.ones(periods)
    X = ts[1][:periods - 1]

sig2    = 1.0
alpha   = 0.8
periods = 100

n = 10000
scores = np.zeros(n)
for i in np.arange(n):
    tArch = tArchFct(sig2, alpha, periods)
    s     = score(tArch)
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
date    = sp500[['Date']][15096:]
returns = sp500[['log-ret_x100']][15096:]
pltf.plotUno(date, returns)

initPar = np.array([1.0, 0.8])
llfTArchSum(initPar, returns)

res = opt.minimize(llfTArchSum, initPar, args = returns)
