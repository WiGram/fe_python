# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:13:35 2018

@author: WiGram
"""

import likelihoodModule as llm
import plotsModule as pltm
import numpy as np
import pandas as pd
import scipy.optimize as opt
np.set_printoptions(suppress = True)   # Disable scientific notation

# ============================================= #
# ===== Load S&P 500 data ===================== #
# ============================================= #

sp500   = pd.DataFrame(pd.read_excel('C:/Users/wigr11ab/Dropbox/KU/K3/FE/Exercises/SP500.xlsx'))
date    = np.array(sp500[['Date']][15096:])
returns = np.array(sp500[['log-ret_x100']][15096:])

# ============================================= #
# ===== Parameterisation ====================== #
# ============================================= #

# Start two-dimensionally
y        = returns
mat      = len(y)
states   = 2
s1, s2   = 1.0, 0.5
p11, p22 = 0.5, 0.5
p12, p21 = 1.0 - p11, 1.0 - p22

f1    = 1 / np.sqrt(2 * np.pi * s1) * np.exp(- 0.5 * y ** 2 / s1)
f2    = 1 / np.sqrt(2 * np.pi * s2) * np.exp(- 0.5 * y ** 2 / s2)

a = np.ones(states * mat).reshape(states, mat)
b = np.ones(states * mat).reshape(states, mat)

a[0,0], a[1,0] = f1[0] / mat, f2[0] / mat

for t in range(1, mat):
    a[0,t] = f1[t] * (p11 * a[0, t-1] + p21 * a[1, t-1]) # Has to end in state 1
    a[1,t] = f2[t] * (p12 * a[0, t-1] + p22 * a[1, t-1]) # Has to end in state 2
    # a[0, t] = f1[t] * sum([p11, p21] * a[:, t-1]) <-- Slightly more dynamic

for t in range(mat-2, -1, -1):
    b[0, t] = f1[t+1] * b[0, t+1] * p11 + f2[t+1] * b[1, t+1] * p12
    b[1, t] = f1[t+1] * b[0, t+1] * p21 + f2[t+1] * b[1, t+1] * p22

# We note, that there is a problem because of extremely small values of a, b.
pltm.plotDuo(range(mat), a[0,:], a[1,:], 'State 1', 'State 2', 'Time', 'Forward')
pltm.plotDuo(range(mat), b[0,:], b[1,:], 'State 1', 'State 2', 'Time', 'Backward')

# Because of these extremely small values, stable solutions are not found, and
# for this reason we skip the computation of probabilities.

# To fix the problem of scale, we rescale a, b to find stable solutions.

# ============================================= #
# ===== Rescaled EM algorithm ================= #
# ============================================= #

# A. Forward algorithm
a_s = np.ones(mat)                               # a_scale
a_r = np.ones(states * mat).reshape(states, mat) # a_rescale

# t = 0
a_s[0]    = sum(a[:, 0])
a_r[:, 0] = a[:, 0] / a_s[0]

# t in [1, T]
for t in range(1, mat):
    a[0, t]   = sum(f1[t] * [p11, p21] * a_r[:, t-1])
    a[1, t]   = sum(f2[t] * [p12, p22] * a_r[:, t-1])
    a_s[t]    = sum(a[:, t])
    a_r[:, t] = a[:,t] / a_s[t]

pltm.plotDuo(range(mat), a_r[0,:], a_r[1,:], \
             'State 1', 'State 2', 'Time', 'Forward')

# B. Backward algorithm
b_s = np.ones(mat)                               # b_scale
b_r = np.ones(states * mat).reshape(states, mat) # b_rescale

# t = T (= mat - 1)
b_s[mat-1]      = sum(b[:, mat - 1])
b_r[:, mat - 1] = b[:, mat - 1] / b_s[mat - 1]

# t in [0, T - 1] (= mat - 2, stops at previous index, i.e. 0)
for t in range(mat - 2, -1, -1):
    b[0, t]   = sum(b_r[0, t+1] * f1[t+1] * p11, b_r[1, t+1] * f2[t+1] * p12)
    b[1, t]   = sum(b_r[0, t+1] * f1[t+1] * p21, b_r[1, t+1] * f2[t+1] * p22)
    b_s[t]    = sum(b[:,t])
    b_r[:, t] = b[:, t] / b_s[t]

# if p11 = p22 = 0.5 then b_r = 0.5 for all t by construction
pltm.plotDuo(range(mat), b_r[0,:], b_r[1,:], \
             'State 1', 'State 2', 'Time', 'Forward')

# --------------------------------------------- #
# ----- Transition probabilities -------------- #
# --------------------------------------------- #

# --------------------------------------------- #
# 1. Smoothed probabilities in Markov state switching (FB: Forward-Backward)
pStarFB  = np.ones(states * mat).reshape(states, mat)
denom = b_r[0,:] * a_r[0,:] + b_r[1,:] * a_r[1,:]
for s in range(states):
    pStarFB[s, :] = (b_r[s, :] * a_r[s, :]) / denom

# Check that we computed actual probabilities
print('OK: Minimum is weakly larger than 0') if min([min(pStarFB[i,:]) >= 0.0 for i in range(states)]) else 'Problem: Minimum is less than zero'
print('OK: Maximum is weakly less than 1') if min([max(pStarFB[i,:]) <= 1.0 for i in range(states)]) else 'Problem: Maximum is larger than 1'
print('OK: Probabilities sum to 1') if min(np.round(pStarFB[0,:] + pStarFB[1,:], 10) == 1.0) else "Problem: Probabilities don't sum to 1"

# Check that probabilities are dynamic
pltm.plotDuo(range(mat), pStarFB[0,:], pStarFB[1,:], \
             'State 1', 'State 2', 'Time', 'Probability', title = 'Smoothed probabilities')

# --------------------------------------------- #
# 2. Smoothed TRANSITION probabilities in Markov state switching (these are joint probabilities, not conditional)
pStarFBT = np.ones(states * states *mat).reshape(states * states, mat)

denom = denom * a_s
pStarFBT[:, 0] = np.array([p11, p12, p21, p22]) / 2 # Assume the first period is given
for t in range(1, mat):
    pStarFBT[0, t] = b_r[0, t] * f1[t] * p11 * a_r[0, t-1] / denom[t]
    pStarFBT[1, t] = b_r[1, t] * f2[t] * p12 * a_r[0, t-1] / denom[t]
    pStarFBT[2, t] = b_r[0, t] * f1[t] * p21 * a_r[1, t-1] / denom[t]
    pStarFBT[3, t] = b_r[1, t] * f2[t] * p22 * a_r[1, t-1] / denom[t]

# Check that we computed actual probabilities
print('OK: Minimum is weakly larger than 0') if min([min(pStarFBT[i,:]) >= 0.0 for i in range(states ** 2)]) else 'Problem: Minimum is less than zero'
print('OK: Maximum is weakly less than 1') if min([max(pStarFBT[i,:]) <= 1.0 for i in range(states ** 2)]) else 'Problem: Maximum is larger than 1'
print('OK: Probabilities sum to 1') if min(np.round(sum([pStarFBT[k,:] for k in range(states ** 2)]), 10) == 1.0) else "Problem: Probabilities don't sum to 1"

# Check that probabilities are dynamic
pltm.plotDuo(range(mat), pStarFBT[0,:], pStarFBT[1,:], \
             'State 1', 'State 2', 'Time', 'Probability', title = 'Smoothed transition probabilities (joint), with s(t-1) = 1')

pltm.plotDuo(range(mat), pStarFBT[2,:], pStarFBT[3,:], \
             'State 1', 'State 2', 'Time', 'Probability', title = 'Smoothed transition probabilities (joint), with s(t-1) = 2')

# Do realise, that we have computed one guess only,
# That is, we have not completed the maximisation step.