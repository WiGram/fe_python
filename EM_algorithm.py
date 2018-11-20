# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 21:58:35 2018

@author: WiGram
"""

"""
Here I am building the ME algorithm to estimate
parameters in stochastic volatility models.

I am preparing a guide in JupyterLab that roughly
introduces the theory.
"""
# %reset -f
# %clear

import numpy as np
import matplotlib.pyplot as plt

" Plotting function "
def plotting(x, y1, y2, y1_lab, y2_lab, x_lab, y_lab, title = ""):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(x, y1, label = y1_lab)
    ax.plot(x, y2, label = y2_lab, marker = 'o', 
            alpha = 0.5, markerfacecolor="None")
    ax.set_title(title)
    ax.legend(loc = 'lower right', shadow = False)
    ax.set_ylabel(y_lab)
    ax.set_xlabel(x_lab)
    fig.tight_layout()
    return plt.show()

" Initial parameters "
mat = 250  # no. of periods
s_t = 2    # no. of states
s_h = 4.0  # high vol state
s_l = 0.1  # low vol state
p   = 0.5  # transition probability

" Random numbers - z: noise, u: state probability "
z = np.random.normal(0,1, size = mat)
u = np.random.uniform(0, 1, size = mat)

state   = (u > p) * 1 + (u < p) * 0
returns = (u > p) * s_h * z + (u < p) * s_l * z

plot_time = np.arange(0,mat) # x-axis
plotting(plot_time, returns, state, 'Returns', 'State',
         'Time', 'Returns', 'Mixture Model')

# ============================================= #
# ===== Building the EM algorithm ============= #
# ============================================= #

"""
A succesful implementation of the EM algorithm
should allow me to retrieve the model parameters,
even with wrongful initial guesses.

Much easier to understand with the JupyterLab
guide.
"""

# Initial two-state parameter guesses; must be wrong.
theta = np.array([0.25, 0.25, 2, 2])

# Conditional probability given state (hence vol)
def cond_dens(returns, vol):
    return 1 / np.sqrt(2 * np.pi * vol) * np.exp( - 0.5 * (returns / vol) ** 2)

# Probability transition matrix
# Assigns probabilities, 1 for each of s_t states
p_matrix = np.zeros(s_t * s_t).reshape(s_t, s_t)
p_matrix[:,0] = theta[:s_t]
    
for p in np.arange(1,s_t):
    if p < s_t-1: 
        p_matrix[:,p] = theta[p*(s_t + 1) : (p + 1) * (s_t + 1)]
    else:
        p_matrix[:,p] = 1 - p_matrix.sum(axis = 1) 

# --------------------------------------------- #
# The forward Algorithm
forward_a = np.zeros(s_t * mat).reshape(s_t, mat)

# Takes all rows (:) and produces for all vol's
forward_a[:, 0] = cond_dens(returns[0], theta[s_t:]) / s_t

a_scale = np.repeat(forward_a.sum(axis = 0)[0], mat)
a_rs    = np.zeros(s_t * mat).reshape(s_t, mat)
a_rs[:, 0] = forward_a[:, 0] / a_scale[0]

# This part still needs testing
for t in np.arange(1, mat):
    lag_a           = p_matrix * a_rs[:,t-1]
    forward_a[:, t] = cond_dens(returns[t], theta[s_t:]) * lag_a.sum(axis = 1)
    a_scale[t]      = sum(forward_a[:, t])
    a_rs[:, t]      = forward_a[:, t] / a_scale[t]
    
# --------------------------------------------- #
# The backward algorithm
backward_b = np.repeat(1., s_t * mat).reshape(s_t, mat)

b_scale = np.repeat(backward_b.sum(axis = 0)[mat-1], mat)
b_rs    = np.zeros(s_t * mat).reshape(s_t, mat)
b_rs[:, mat-1] = backward_b[:, mat - 1] / b_scale[mat - 1]

for t in np.arange(mat - 1)[::-1]:
    lead_b           = p_matrix * b_rs[:, t+1]
    backward_b[:, t] = tuple((cond_dens(returns[t+1], theta[s_t:]) * lead_b).sum(axis = 1))
    b_scale[t]       = sum(backward_b[:, t])
    b_rs[:, t]       = backward_b[:, t] / b_scale[t]
    
# --------------------------------------------- #
# Smoothed probabilities, present and conditional: WIP.
smooth_p  = np.zeros(s_t * mat).reshape(s_t, mat)
denom     = (b_rs * a_rs).sum(axis = 0)
for t in np.arange(mat):
    smooth_p[:, t] = b_rs[:, t] * a_rs[:, t] / denom[t]

smooth_cp = np.zeros(s_t * s_t * (mat - 1) ).reshape((mat - 1), s_t, s_t)
denom_cp  = denom * a_scale
for t in np.arange(1,mat):
    smooth_cp[t-1,:,:] = b_rs[:,t] * cond_dens(returns[t], theta[len(theta) - s_t:]) *\
    p_matrix * a_rs[:, t-1] / denom_cp[t]

# --------------------------------------------- #
# Writing the expectation of the log-likelihood
constant = 1
first_sum = sum(sum(np.log(p_matrix) * smooth_cp.sum(axis = 0)))
second_sum = sum(np.array([cond_dens(returns, v) for v in theta[len(theta) - s_t:]]).sum(axis = 1))
log_lik = constant + first_sum + second_sum
log_lik
