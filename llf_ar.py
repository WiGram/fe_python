# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 06:59:27 2018

@author: wigr11ab
"""

import numpy as np

def llf_ar(theta, y):
    end = len(y)
    mu, rho, sd = theta
    
    mu_cond = mu + rho * y[:end-1]
    log_sd2 = np.log(2 * np.pi * sd ** 2)
    
    return 0.5 * (log_sd2 + ((y[1:] - mu_cond) / sd) ** 2)

def llf_ar_sum(theta, y):
    return sum(llf_ar(theta, y))