# -*- coding: utf-8 -*-
"""
Created on Wed 12 11:51:47 2018

@author: wigr11ab
"""

import numpy as np                     # Efficient programming
import sys                             # Appending library of cuntions
sys.path.append("C:/Users/wigr11ab/Dropbox/KU/K3/FE/Python/")
import plotsModule as pltm             # Custom plotting
from matplotlib import pyplot as plt   # Bespoke plotting
np.set_printoptions(suppress = True)   # Disable scientific notation

# ============================================= #
# ===== Exercise 3 ============================ #
# ============================================= #
" 3.1 Mixture ARCH model "
mat = 10 ** 3
s_h = 2.0
a_h = 0.5
a_l = 0.9
s_l = 0.5
p   = 0.5

# s ~ iid
np.random.seed(12345)
z = np.random.normal(0,1, size = mat)
u = np.random.uniform(0, 1, size = mat)

state   = (u > p) * 1 + (u < p) * 0
vol = [1] * mat
ret = [0] * mat
for t in range(1, mat):
    vol[t] = (s_h ** 2 + a_h * ret[t-1] ** 2) if state[t] else (s_l ** 2 + a_l * ret[t-1] ** 2)
    ret[t] = vol[t] ** .5 * z[t]

x = np.arange(0,mat)
pltm.plotDuo(x = x, y1 = ret, y2 = state, \
            yLab1 = 'Returns', yLab2 = 'State', \
            yLab = 'Returns', xLab = 'Time')

# Compare with old model
returns = ((u > p) * s_h + (u < p) * s_l) * z
pltm.plotDuo(x = x, y1 = returns, y2 = state, \
            yLab1 = 'Returns', yLab2 = 'State', \
            yLab = 'Returns', xLab = 'Time')

# s ~ Markov chain
" Two-state hidden Markov volatility model "

p11 = 0.95
p22 = 0.90

state_ms = np.repeat(0, mat)
for t in range(1,mat):
    state_ms[t] = (u[t] < p11) * 0 + (u[t] > p11) * 1 if state_ms[t-1] == 0 else (u[t] < p22) * 1 + (u[t] > p22) * 0

vol_ms = [1] * mat
ret_ms = [0] * mat
for t in range(1, mat):
    vol_ms[t] = (s_h ** 2 + a_h * ret_ms[t-1] ** 2) if state_ms[t] else (s_l ** 2 + a_l * ret_ms[t-1] ** 2)
    ret_ms[t] = vol_ms[t] ** .5 * z[t]

pltm.plotDuo(x = x, y1 = ret_ms, y2 = state_ms, \
            yLab1 = 'Returns', yLab2 = 'State', \
            yLab = 'Returns', xLab = 'Time')

# Compare with old model
returns_ms = (state_ms * s_h + (1 - state_ms) * s_l) * z
pltm.plotDuo(x = x, y1 = returns_ms, y2 = state_ms, \
            yLab1 = 'Returns', yLab2 = 'State', \
            yLab = 'Returns', xLab = 'Time')


# ============================================= #
# ===== Exercise 4.2 ========================== #
# ============================================= #

a, b = 0.5, 4.0
y = [0] * mat
s = [0] * mat
p = [0.5] * mat

for t in range(1,mat):
    p[t] = np.exp(a * (y[t-1] - b)) / (1 + np.exp(a * (y[t-1] - b)))
    s[t] = int(u[t] > p[t])
    y[t] = s_h * z[t] if s[t] else s_l * z[t]

pltm.plotDuo(x = x, y1 = y, y2 = s, \
            yLab1 = 'Returns', yLab2 = 'State', \
            yLab = 'Returns', xLab = 'Time')