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

# First exercise 1.5
x = np.arange(-5,5,0.01)
y = ss.norm.pdf(x)
z = ss.t.pdf(x, 3)

pltm.plotDuo(x, y, z, 'Standard Normal PDF', "Student's t (v=3)", '', 'Density', "Comparison between PDF's", loc = 'upper right')

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
pltm.plotUno(x, scores, 'Score value', 'Simulation trial', 'Distribution of scores', 'upper right')

x = np.sort(np.random.normal(size = n))
# sm.qqplot(scores, )
pltm.scatterUno(x, np.sort(scores), yLab = 'Emprical quantiles: Score', xLab='Theoretical quantiles', title='Normal QQ-Plot')

pltm.hist(scores, title = 'Histogram of scores')


# Next exercise 1.7
sp500   = pd.DataFrame(pd.read_excel('C:/Users/wigr11ab/Dropbox/KU/K3/FE/Exercises/SP500.xlsx'))
date    = np.array(sp500[['Date']][15096:])
returns = np.array(sp500[['log-ret_x100']][15096:])
pltm.plotUno(date, returns)

initPar = np.array([0.05, 0.5])  # sigma^2; alpha
res = opt.minimize(llm.llfArchSum, initPar, args = returns, method = 'L-BFGS-B')
estPar = res.x
mlVal  = llm.llfArchSum(estPar, returns)

# Standard error calculation
hFct = nd.Hessian(llm.llfArchSum)
hess = np.linalg.inv(hFct(estPar, returns))
se   = np.sqrt(np.diag(hess))
tVal = estPar / se

jFct  = nd.Jacobian(llm.llfArch)
jac   = jFct(estPar, returns)
jac   = np.transpose(np.squeeze(jac, axis=0)) # Squeez removes a redundant dimension.
score = np.inner(jac, jac)

seRobust   = np.sqrt(np.diag(hess.dot(score).dot(hess)))
tValRobust = estPar / seRobust

mlResults = pd.DataFrame([estPar, se, seRobust, tVal, tValRobust, mlVal], \
                         columns=['sigma2', 'alpha', 'gamma'], \
                         index=['estimate', 'se', 'robust se', 't-val', 'robust t-val', 'ml val'])
mlResults

# Next exercise 2.5
gjrArch = tsm.gjrArchFct(1., 0.5, 0.9, 100)
x = np.arange(len(gjrArch[0]))
pltm.plotUno(x, gjrArch[0], title = 'GJR-ARCH(1) Return process')

# Apply to SP500 data (we transform the parameters - will need the delta method)
initPar = np.array([2.0, 2.0, 2.0])  # log(sigma^2, alpha, gamma)
resGjr  = opt.minimize(llm.deltaGjrArchSum, initPar, args = returns, method = 'L-BFGS-B')
gjrPar  = resGjr.x
mlVal   = llm.deltaGjrArchSum(gjrPar, returns)

# Standard error calculation
hFct = nd.Hessian(llm.deltaGjrArchSum)
hess = np.linalg.inv(hFct(gjrPar, returns))

jFct  = nd.Jacobian(llm.deltaGjrArch)
jac   = jFct(gjrPar, returns)
jac   = np.transpose(np.squeeze(jac, axis=0)) # Squeez removes a redundant dimension.
score = np.inner(jac, jac)

sandwich = hess.dot(score).dot(hess)

jacA = nd.Jacobian(np.exp)
A    = jacA(gjrPar)

# Applying the delta method to transformed parameters to
# find se for non-transformed parameters
se = np.sqrt(np.diag(A.dot(sandwich).dot(np.transpose(A))))
tVal = np.exp(gjrPar) / se

mlResults = pd.DataFrame([np.exp(gjrPar), se, tVal, mlVal], \
                         columns=['sigma2', 'alpha', 'gamma'], \
                         index=['estimate', 'se', 't-val', 'ml val'])
mlResults

# Analysing the residuals z = x / sigma
theta = np.exp(gjrPar)
n  = len(returns)
x  = np.squeeze(returns)
s2 = theta[0] + theta[1] * x[:n - 1]**2 + theta[2] * (x < 0)[:n - 1] * x[:n - 1]**2
z  = x[1:] / s2

pltm.qqPlot(z)
pltm.hist(z)
pltm.plotUno(np.arange(n-1), z, yLab='Residuals')