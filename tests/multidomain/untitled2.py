# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:00:46 2024

@author: Nathan
"""

import numpy as np

x = np.arange(0,10000)

ftn = np.zeros((4,10000))
for j in range(4):
    ftn[j,:] = x**2
print(ftn)
y = np.array([0,2,5,8])

df_dy, df_dx = np.gradient(ftn, y, x)
print(f'df_dy {df_dy}')
print(f'df_dx {df_dx}')
print(2*x)
