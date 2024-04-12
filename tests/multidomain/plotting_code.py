# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:08:21 2024

@author: Nathan
"""

import numpy as np
import matplotlib.pyplot as plt

plt.figure()
plt.plot(52-final_res_15[:,0], final_res_15[:,1], label='15 terms - [200,128]')
plt.plot(52-final_res_20[:,0], final_res_20[:,1], label='20 terms - [200,128]')
plt.plot(52-final_res_25[:,0], final_res_25[:,1], label='25 terms - [200,128]')
plt.plot(52-final_res_28[:,0], final_res_28[:,1], label='28 terms - [200,128]')
plt.plot(52-final_res_30[:,0], final_res_30[:,1], label='30 terms - [200,128]')
plt.plot(52-final_res_30_1000[:,0], final_res_30_1000[:,1], label='30 terms - [1000,128]')

plt.plot(np.array([np.min(52-final_res_30[:,0]), np.max(52-final_res_30[:,0])]), np.array([27.73181005, 27.73181005]), label='LEFM')
plt.plot(np.array([np.min(52-final_res_30[:,0]), np.max(52-final_res_30[:,0])]), np.array([26.96092978, 26.96092978]), label='FEM')

plt.ylim(26,35)
plt.xlabel('Length of top3 subpanel [mm] \n min=0 max=52')
plt.ylabel('Force [N] - Line integral of Qx')
plt.title('Effect of order of shape functions \n [grid_x,y_gauss]')

plt.grid()
plt.legend()
plt.show()

# %% Convergence of no_y_gauss

plt.figure()
plt.plot(np.arange(1,8,1), final_res_30_ref[:,1], label='a2=20, gridx=1000')
plt.plot(np.arange(1,8,1), final_res_30_ref_1[:,1], label='a2=20, gridx=5000')
plt.plot(np.arange(1,8,1), final_res_30_ref_2[:,1], label='a2=20, gridx=10000')
plt.xticks(np.arange(1,8,1), labels=iter_index)
plt.ylabel('Force [N] - Line integral of Qx')
plt.xlabel('Gauss integration points along y')
plt.title('Convergence of Gauss integration points along y')

plt.grid()
plt.legend()
plt.show()

# %% Convergence of grid_x

plt.figure(figsize=(10,5))
plt.plot(np.arange(1,16,1), final_res_30_ref_2[:,1], label='a2=20')
plt.plot(np.arange(1,16,1), final_res_30[:,1], label='a2=30')

plt.plot(np.array([1,15]), np.array([27.73181005, 27.73181005]), label='LEFM')
plt.plot(np.array([1,15]), np.array([26.96092978, 26.96092978]), label='FEM')

plt.xticks(np.arange(1,16,1), labels=iter_index)
plt.ylabel('Force [N] - Line integral of Qx')
plt.xlabel('Points along x')
plt.title('Convergence of (differentiation) points along x')

plt.grid()
plt.legend()
plt.show()

# %% Plot w at tip

plt.figure()
if False:
    plt.plot(final_res_15['y'][2][:,-1], final_res_15['w'][2][:,-1], label = '15 terms')
    plt.plot(final_res_20['y'][2][:,-1], final_res_20['w'][2][:,-1], label = '20 terms')
    plt.plot(final_res_25['y'][2][:,-1], final_res_25['w'][2][:,-1], label = '25 terms')
    plt.plot(final_res_30['y'][2][:,-1], final_res_30['w'][2][:,-1], label = '30 terms')
if False:
    plt.plot(final_res_15['y'][2][:,-1], final_res_15['Mxx'][2][:,-1], label = '15 terms')
    plt.plot(final_res_20['y'][2][:,-1], final_res_20['Mxx'][2][:,-1], label = '20 terms')
    plt.plot(final_res_25['y'][2][:,-1], final_res_25['Mxx'][2][:,-1], label = '25 terms')
    plt.plot(final_res_30['y'][2][:,-1], final_res_30['Mxx'][2][:,-1], label = '30 terms')
    plt.ylabel('Mxx')
if True:
    plt.plot(final_res, final_res_15, label = '15 terms')
    plt.plot(final_res, final_res_20, label = '20 terms')
    plt.plot(final_res, final_res_25, label = '25 terms')
    plt.plot(final_res, final_res_30, label = '30 terms')
    plt.ylabel('Qx')

plt.xlabel('Width (along y) [mm]')
plt.title('Variation for a2=30mm @ top3.a')
# plt.ylim(np.min(final_res_15['w'][2][:,-1]), np.max(final_res_15['w'][2][:,-1]))

plt.grid()
plt.legend()
plt.show()

# %% 

plt.figure()
plt.plot(100-temp_res[:,0], temp_res[:,1], label='30 terms')
plt.plot(100-temp_res_15[:,0], temp_res_15[:,1], label='15 terms')
plt.plot(np.array([np.min(temp_res[:,0]), np.max(temp_res[:,0])]), np.array([118.827, 118.827]), label='FEM')

plt.xlabel('Dimensions of tip panel [mm] \n min=0 max=100')
plt.ylabel('Force [N] - Line integral of Qx')
plt.title('Single plate with 2 panels (no DCB)')
# plt.ylim(np.min(final_res_15['w'][2][:,-1]), np.max(final_res_15['w'][2][:,-1]))

plt.grid()
plt.legend()
plt.show()

# %%  2 panels no DCB variation along the edge

plt.figure(figsize=(10,5))
dy = dy_Mxx.copy()
plt.plot(dy, temp_res_15, label='15 terms; a=95 mm')
# plt.plot(dy, temp_res_20, label='20 terms; a=95 mm')
# plt.plot(dy, temp_res_25, label='25 terms; a=95 mm')
plt.plot(dy, temp_res_30, label='30 terms; a=95 mm')

plt.plot(dy, temp_res_50_15, label='15 terms; a=50 mm')
# plt.plot(dy, temp_res_50_20, label='20 terms; a=50 mm')
# plt.plot(dy, temp_res_50_25, label='25 terms; a=50 mm')
plt.plot(dy, temp_res_50_30, label='30 terms; a=50 mm')
# plt.plot(np.array([np.min(dy), np.max(dy)]), np.array([118.827, 118.827]), label='FEM')

plt.xlabel('Width (along y) [mm]')
plt.ylabel('Mxx \n (limits clipped)')
plt.title('Single plate with 2 panels (no DCB) @ 15 mm tip displ')
plt.ylim(-1,1)

plt.grid()
plt.legend()
plt.show()