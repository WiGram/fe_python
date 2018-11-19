# -*- coding: utf-8 -*-
"""
Spyder Editor

Created on Sun Nov  4 08:57:28 2018

@author: William Gram
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================= #
# ===== Functions ============================= #
# ============================================= #

" Plotting function"
def plotting(x, y1, y2, y1_lab, y2_lab, x_lab, y_lab, title = ""):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(x, y1, label = y1_lab)
    ax.plot(x, y2, label = y2_lab, marker = 'o', alpha = 0.5, markerfacecolor="None")
    ax.set_title(title)
    ax.legend(loc = 'lower right', shadow = False)
    ax.set_ylabel(y_lab)
    ax.set_xlabel(x_lab)
    fig.tight_layout()

# ============================================= #
# ===== Models ================================ #
# ============================================= #

# --------------------------------------------- #
" Likelihood estimation "
def llh(x, sig2, mu):
    1 / np.sqrt(2 * np.pi * sig2) * np.exp(- 0.5 * (x - mu) ** 2 / sig2)

def log_step(x, s, sig2_0, sig2_1, p):
    (s == 0) * (np.log(llh(x, sig2_0, 0) + np.log(p))) \
    + (s == 1) * (np.log(llh(x, sig2_1, 0) + np.log(1 - p)))

def smoothed_prob(x, sig2_0, sig2_1, p_tilde, s):
    if s == 0:
        sig = sig2_0
    else:
        sig = sig2_1
    llh(x, sig, 0) * p_tilde \
    / (llh(x, sig2_0, 0) * p_tilde + llh(x, sig2_1, 0) * (1 - p_tilde))

def exp_step(x, s, sig2_0, sig2_1, p):
    smooth_p = smoothed_prob(x, sig2_0, sig2_1, p_tilde, 0)
    log_lik_0 = np.log(llh(x, sig2_0, 0))
    log_lik_1 = np.log(llh(x, sig2_1, 0))
    smooth_p * (log_lik_0 + np.log(p)) \
    + (1 - smooth_p) * (log_lik_1 + np.log(1 - p))

# --------------------------------------------- #