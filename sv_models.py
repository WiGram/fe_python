# -*- coding: utf-8 -*-
"""
Spyder Editor

Created on Sun Nov  4 08:57:28 2018

@author: William Gram
"""
# %reset -f
# %clear

import numpy as np
import matplotlib.pyplot as plt

# ============================================= #
# ===== Functions ============================= #
# ============================================= #

def plotting(x, y1, y2, y1_lab, y2_lab, x_lab, y_lab, title = ""):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(x, y1, label = y1_lab)
    ax.plot(x, y2, label = y2_lab, marker = 'o', alpha = 0.5, markerfacecolor="None")
    ax.set_title(title)
    ax.legend(loc = 'lower right', shadow = False)
    ax.set_ylabel(y_lab)
    ax.set_xlabel(x_lab)
    fig.tight_layout()
    return plt.show()

# ============================================= #
# ===== Models ================================ #
# ============================================= #

# --------------------------------------------- #
" Mixture model "
mat = 250
s_h = 4.0 
s_l = 0.1
p   = 0.5

z = np.random.normal(0,1, size = mat)
u = np.random.uniform(0, 1, size = mat)

state   = (u > p) * 1 + (u < p) * 0
returns = (u > p) * s_h * z + (u < p) * s_l * z

plot_time = np.arange(0,mat)
plotting(plot_time, returns, state, 'Returns', 'State',
         'Time', 'Returns', 'Mixture Model')


# --------------------------------------------- #
" Markov switching SV model "

p11 = 0.95
p22 = 0.90

state_ms = np.repeat(0, mat)
for t in range(1,mat):
    if state_ms[t-1] == 0:
        state_ms[t] = (u[t] < p11) * 0 + (u[t] > p11) * 1
    else:
        state_ms[t] = (u[t] < p22) * 1 + (u[t] > p22) * 0

returns_ms = (state_ms * s_h + (1 - state_ms) * s_l) * z
plotting(plot_time, returns_ms, state_ms, 'Returns', 'State',
         'Time', 'Returns', 'Markov Switching SV Model')

# --------------------------------------------- #
" Continuous SV model "
mat   =  1000
gamma = -0.50
sigma =  0.31
phi   =  0.95
eta   = np.random.normal(0, sigma, size = mat)
z     = np.random.normal(0, 1, size = mat)
x     = np.arange(0, mat)

log_s2 = np.repeat(-10.0, mat)

for t in range(1, mat):
    log_s2[t] = gamma + phi * log_s2[t-1] + eta[t]

s_cm = np.sqrt(np.exp(log_s2))
returns_cm = s_cm * z

plot_time = np.arange(mat)
plotting(plot_time, returns_cm, s_cm, 'Returns', 'Volatility',
         'Time', 'Returns', 'Continuous SV Model')
# ============================================= #
# ============================================= #