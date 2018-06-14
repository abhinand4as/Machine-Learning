# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 10:30:08 2018

@author: Abhinand A S
"""

scores = [3.0, 5.0, 0.2]
import numpy as np
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis = 0)
print(softmax(scores))

#plot softmax curves

import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2*np.ones_like(x)]) 
plt.plot(x, softmax(scores).T, linewidth = 2)
plt.show()
   
