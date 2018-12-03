# -*- coding: utf-8 -*-
"""
Created on Mon Dec 03 21:02:39 2018

@author: wigr11ab
"""
import numpy as np

def tArch3(ts, df = 3):
    periods = len(ts[0])
    y = (3. * ts[2][1:] ** 2 - 1.) / (1. + ts[2][1:] ** 2)
    u = ts[0][:periods - 1] ** 2 / ts[1][1:]

    return 1 / np.sqrt(periods - 1.) * sum(y * u)
