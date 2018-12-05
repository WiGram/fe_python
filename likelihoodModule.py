# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 06:59:27 2018

@author: wigr11ab
"""

import numpy as np

def llfAr(theta, y):
    end = len(y)
    mu, rho, sd = theta
    
    mu_cond = mu + rho * y[:end-1]
    log_sd2 = np.log(2 * np.pi * sd ** 2)
    
    return 0.5 * (log_sd2 + ((y[1:] - mu_cond) / sd) ** 2)

def llfArSum(theta, y):
    return sum(llfAr(theta, y))

def llfTArch(theta, y):
    end = len(y)
    sig2, alpha = theta

    s2 = np.array(sig2 + alpha * y[:end - 1] ** 2)

    return -(-np.log(s2) - 4 * np.log(1 + y[1:] ** 2 / s2))

def llfTArchSum(theta, y):
    return sum(llfTArch(theta, y))

def llfGjrArch(theta, y):
    if len(theta) != 3:
        return 'Parameter must have dimension 3.'
    end = len(y)
    sig2, alpha, gamma = theta
    
    idx = (y < 0)
    s2     = sig2 + alpha * y[:end - 1] ** 2 + gamma * idx[:end - 1] * y[:end - 1] ** 2
    log_s2 = np.log(s2)

    return -0.5 * (np.log(2 * np.pi) + log_s2 + y[1:] ** 2 / s2)

def llfGjrArchSum(theta, y):
    return -sum(llfGjrArch(theta, y))


