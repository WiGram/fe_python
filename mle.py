# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:18:21 2018

@author: wigr11ab
"""
!cls
import sys
sys.path.append("C:/Users/wigr11ab/Dropbox/KU/K3/FE/Python/")
from ar_function import ar_fct
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt

data   = pd.read_excel("C:/Users/wigr11ab/Dropbox/KU/K3/FE/Exercises/USspreadraw.xls")
spread = data.Y_uncor
end    = len(spread)

theta = np.array([1., 0.5, 2.]) #mu, rho, sigma

test = ar_fct(*theta, end) # * unpacks theta's three args

def llf_ar(theta, y):
    end = len(y)
    mu, rho, sd = theta
    
    mu_cond = mu + rho * y[:end-1]
    log_sd2 = np.log(2 * np.pi * sd ** 2)
    
    return 0.5 * sum(log_sd2 + ((y[1:] - mu_cond) / sd) ** 2)

theta = [1.0, 0.5, 2.0]

llf_ar(theta, test)

minimize(llf_ar, theta, args = test)
