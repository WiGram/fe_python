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

    s2      = np.array(sig2 + alpha * y[:end - 1] ** 2)
    x       = np.array(y[1:] ** 2)
    z       = x / s2
    log_sd2 = np.log(s2)

    return -(-log_sd2 - 4 * np.log(1 + z))

def llfTArchSum(theta, y):
    return sum(llfTArch(theta, y))

