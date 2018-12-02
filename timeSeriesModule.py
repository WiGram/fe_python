# -*- coding: utf-8 -*-

import numpy as np

def archFct(sig2, alpha, periods):
    x  = np.zeros(periods)
    z  = np.random.normal(0, 1, periods)
    v2 = np.ones(periods)

    for t in np.arange(1, periods):
        v2[t] = sig2 + alpha * x[t-1] ** 2
        x[t]  = np.sqrt(v2[t]) * z[t]
    
    return np.array([x, v2])

def aArchFct(sig2, alpha_n, alpha_p, periods):
    x  = np.zeros(periods)
    z  = np.random.normal(0, 1, periods)
    v2 = np.ones(periods)

    for t in np.arange(1, periods):
        v2[t] = sig2 + (x[t-1] < 0) * alpha_n * x[t-1] ** 2 + (x[t-1] > 0) * alpha_p * x[t-1] ** 2
        x[t]  = np.sqrt(v2[t]) * z[t]
    
    return np.array([x, v2])

def arFct(mu, rho, sig, periods):
    x = np.zeros(periods)
    
    eps = np.random.normal(0,sig,periods)
    
    for t in np.arange(1,periods):
        x[t] = mu + rho * x[t-1] + eps[t]
        
    return np.array(x)

def tarFct(mu1, mu2, rho1, rho2, sig, lmbd, periods):
    x = np.zeros(periods)
    z = np.random.normal(0, sig, periods)

    for t in np.arange(1, periods):
        x[t] = (np.abs(x[t-1]) <= lmbd ) * (mu1 + rho1 * x[t-1]) + (np.abs(x[t-1]) > lmbd) * (mu2 + rho2 * x[t-1]) + z[t]

    return np.array(x)

