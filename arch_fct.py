# -*- coding: utf-8 -*-

import numpy as np

def arch_fct(sig2, alpha, periods):
    x  = np.zeros(periods)
    z  = np.random.normal(0, 1, periods)
    v2 = np.ones(periods)


    for t in np.arange(1, periods):
        v2[t] = sig2 + alpha * x[t-1] ** 2
        x[t]  = np.sqrt(v2[t]) * z[t]
    
    return x, v2