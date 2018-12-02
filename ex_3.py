# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 19:54:41 2018

@author: wigr11ab
"""
import pandas as pd                    # Reads data
import numpy as np                     # Efficient programming
from matplotlib import pyplot as plt   # Plotting
import numdifftools as nd              # Finding derivatives
import scipy.optimize as opt           # Minimisation for MLE
import statsmodels.api as sm
import sys                             # Appending library of cuntions
sys.path.append("C:/Users/wigr11ab/Dropbox/KU/K3/FE/Python/")
import time_series_module as tsm       # Import ARCH simulation
from llf_ar import llf_ar, llf_ar_sum  # Import llh contributions and fct.
np.set_printoptions(suppress = True)   # Disable scientific notation

# ============================================= #
# ===== Functions ============================= #
# ============================================= #

def plotting(x, y1, y2, y1_lab, y2_lab, x_lab, y_lab, title = ""):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(x, y1, label = y1_lab)
    ax.plot(x, y2, label = y2_lab, marker = 'o', alpha = 0.5, markerfacecolor="None")
    ax.set_title(title)
    ax.legend(loc = 'lower right', shadow = False)
    ax.set_ylabel(y_lab)
    ax.set_xlabel(x_lab)
    fig.tight_layout()
    return plt.show()

# ============================================= #

arch  = tsm.archFct(1, 0.5, 100)
aarch = tsm.aArchFct(1, 0.6, 0.4, 100)

plt.plot(arch[0])
plt.plot(aarch[0])
plt.show()