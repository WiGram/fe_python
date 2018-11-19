# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 18:00:30 2018

@author: wigr11ab
"""

import numpy as np
import matplotlib.pyplot as plt

def kalman_filter(A, omega_w, Phi, Omega_v, 
                  X_filt_lag, Omega_filt_lag, Zt):
    
    I = np.identity(len(X_filt_lag))
    
    X_pred     = Phi * X_filt_lag
    Omega_pred = Phi * Omega_filt_lag * Phi.transpose() #only works if np.array()
    
    K = Omega_pred * A.transpose() * 