# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 18:53:57 2018

@author: wigr11ab
"""
import numpy as np

def ar_fct(mu, rho, sig, periods):
    y = np.zeros(periods)
    
    eps = np.random.normal(0,sig,periods)
    
    for t in np.arange(1,periods):
        y[t] = mu + rho * y[t-1] + eps[t]
        
    return y
