# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 19:54:41 2018

@author: wigr11ab
"""
import pandas as pd                    # Reads data
import numpy as np                     # Efficient programming
import scipy.optimize as opt           # Minimisation for MLE
import statsmodels.api as sm           # OLS estimation
import sys                             # Appending library of cuntions
sys.path.append("C:/Users/wigr11ab/Dropbox/KU/K3/FE/Python/")
import timeSeriesModule as tsm         # Import ARCH simulation
import plotFct as pltf                 # Custom plotting
np.set_printoptions(suppress = True)   # Disable scientific notation

periods = 10000
x     = np.arange(0,periods)

arch  = tsm.archFct(1., 0.5, periods)
aarch = tsm.aArchFct(1, 0.6, 0.4, periods)

x1 = arch[0][:periods - 1]
y1 = arch[1][1:]

x2 = aarch[0][:periods - 1]
y2 = aarch[1][1:]

pltf.scatterDuo(x1, x2, y1, y2, 'ARCH', 'A-ARCH', title = 'News Curve')


# TAR model

tar = tsm.tarFct(0., 0., 0.2, 0.8, 1., 2.5, periods)

xTar = tar[:periods - 1]
yTar = tar[1:]

pltf.scatterUno(xTar, yTar, xLab = 'Lagged returns', title = 'TAR model')
pltf.plotUno(x, tar, title = 'TAR Process')
