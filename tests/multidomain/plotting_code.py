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

plt.figure(figsize=(8,5))

plt.plot(dy_Qxx_15_30, Qxx_end_15_10, label='15 terms; a2=10 mm')
plt.plot(dy_Qxx_15_30, Qxx_end_30_10, label='30 terms; a2=10 mm')

plt.plot(dy_Qxx_15_30, Qxx_end_15_30, label='15 terms; a2=30 mm')
plt.plot(dy_Qxx_15_30, Qxx_end_30_30, label='30 terms; a2=30 mm')
# plt.plot(np.array([np.min(dy), np.max(dy)]), np.array([118.827, 118.827]), label='FEM')

plt.xlabel('Width (along y) [mm]')
plt.ylabel('Qxx')
plt.title('Applied Load - DCB')
# plt.ylim(-0.25,0.2)

plt.grid()
plt.legend()
plt.show()

# %% Effect of kw (displacement penalty stiffness)

plt.figure(figsize=(8,7))

plt.plot(dy_Qxx_15_8, Qxx_end_15_5, label='15 terms - kw_disp=1e5')
plt.plot(dy_Qxx_15_8, Qxx_end_15_6, label='15 terms - kw_disp=1e6')
plt.plot(dy_Qxx_15_8, Qxx_end_15_7, label='15 terms - kw_disp=1e7')
plt.plot(dy_Qxx_15_8, Qxx_end_15_8, label='15 terms - kw_disp=1e8')
plt.plot(dy_Qxx_15_8, Qxx_end_15_9, label='15 terms - kw_disp=1e9')

plt.plot(dy_Qxx_30_8, Qxx_end_30_5, label='30 terms - kw_disp=1e5')
plt.plot(dy_Qxx_30_8, Qxx_end_30_6, label='30 terms - kw_disp=1e6')
plt.plot(dy_Qxx_30_8, Qxx_end_30_7, label='30 terms - kw_disp=1e7')
plt.plot(dy_Qxx_30_8, Qxx_end_30_8, label='30 terms - kw_disp=1e8')
plt.plot(dy_Qxx_30_8, Qxx_end_30_9, label='30 terms - kw_disp=1e9')

plt.xlabel('Width (along y) [mm]')
plt.ylabel('Qxx')
plt.title('Variation of kw_disp - Load DCB - a2=30mm')

plt.grid()
plt.legend()
plt.show()

# %% - Single panel 

plt.figure(figsize=(7,5))

a1_plot = sp_kr_2_5_1[:, 0]

# 1st no = kr factor; 2nd = kth thousand of gridx; 3rd = kth hundred of gridy
plt.plot(a1_plot, sp_kr_2_5_05[:,1], label='gridx=500; gridy=100')
plt.plot(a1_plot, sp_kr_2_5_1[:,1], label='gridx=1000; gridy=100')
plt.plot(a1_plot, sp_kr_2_5_5[:,1], label='gridx=5000; gridy=100')

plt.plot(a1_plot, sp_kr_2_5_05_2[:,1], label='gridx=500; gridy=200')
plt.plot(a1_plot, sp_kr_2_5_1_2[:,1], label='gridx=1000; gridy=200')
plt.plot(a1_plot, sp_kr_2_5_5_2[:,1], label='gridx=5000; gridy=200')

plt.plot(np.array([np.min(a1_plot), np.max(a1_plot)]), np.array([118.827, 118.827]), label='FEM')


plt.xlabel('Second panel dimension (a2) [mm]')
plt.ylabel('Force (line integral) [N]')
plt.title('SP - kr=2.5*k_calc=1.2E5 - 15 terms - @top3.a @15mm')
plt.xlim(35,65)
plt.ylim(117.25,118.9)

plt.grid()
plt.legend(ncol=2)
plt.show()

# %%
plt.figure(figsize=(7,5))

a1_plot = sp_kr2_5[:, 0]

# 1st no = kr factor; 2nd = kth thousand of gridx; 3rd = kth hundred of gridy

plt.plot(a1_plot, sp_kr6_t8[:,1], label='terms=8 kr=1E6')
plt.plot(a1_plot, sp_kr7_t8[:,1], label='terms=8 kr=1E6')

plt.plot(a1_plot, sp_kr2_5[:,1], label='terms=15 kr=1.2E5')
plt.plot(a1_plot, sp_kr6[:,1], label='terms=15 kr=1E6')
plt.plot(a1_plot, sp_kr7[:,1], label='terms=15 kr=1E7')
plt.plot(a1_plot, sp_kr8[:,1], label='terms=15 kr=1E8')
# plt.plot(a1_plot, sp_kr9[:,1], label='terms=15 kr=1E9')

# plt.plot(a1_plot, sp_kr12_5_t30[:,1], label='terms=30 kr=1.2E5')
plt.plot(a1_plot, sp_kr6_t30[:,1], label='terms=30 kr=1E6')
plt.plot(a1_plot, sp_kr7_t30[:,1], label='terms=30 kr=1E7')
plt.plot(a1_plot, sp_kr8_t30[:,1], label='terms=30 kr=1E8')
# plt.plot(a1_plot, sp_kr9_t30[:,1], label='terms=30 kr=1E9')


plt.plot(np.array([np.min(a1_plot), np.max(a1_plot)]), np.array([118.827, 118.827]), label='FEM')


plt.xlabel('Second panel dimension (a2) [mm]')
plt.ylabel('Force (line integral) [N]')
plt.title('SP - Vartn of kr - @top3.a @15mm - gridx=500, gridy=100')

plt.grid()
plt.legend(ncol=2)
plt.show()

# %% 

plt.figure(figsize=(7,5))

a1_plot = [0.5,1,5,10,20,30,40,45]

# 1st no = kr factor; 2nd = kth thousand of gridx; 3rd = kth hundred of gridy

# plt.plot(a1_plot, hp_kr6_ksb6_t8[:,1], label='terms=8 kr=1E6 ktSB=1e6')

plt.plot(a1_plot, hp_kr6_ksb6_t15[:,1], label='terms=15 kr=1E6 ktSB=1e6')
plt.plot(a1_plot, hp_kr6_ksb7_t15[:,1], label='terms=15 kr=1E6 ktSB=1e7')
plt.plot(a1_plot, hp_kr6_ksb8_t15[:,1], label='terms=15 kr=1E6 ktSB=1e8')
plt.plot(a1_plot, hp_kr6_ksb9_t15[:,1], label='terms=15 kr=1E6 ktSB=1e9')

# plt.plot(a1_plot, hp_kr6_ksb7_t8[:,1], label='terms=8 kr=1E6 ktSB=1e7')


# plt.plot(a1_plot, hp_kr6_ksb6_t30[:,1], label='terms=30 kr=1E6 ktSB=1e6')
# plt.plot(a1_plot, hp_kr6_ksb7_t30[:,1], label='terms=30 kr=1E6 ktSB=1e7')
# plt.plot(a1_plot, hp_kr6_ksb8_t30[:,1], label='terms=30 kr=1E6 ktSB=1e8')
# plt.plot(a1_plot, hp_kr6_ksb9_t30[:,1], label='terms=30 kr=1E6 ktSB=1e9')




plt.plot(np.array([np.min(a1_plot), np.max(a1_plot)]), np.array([1075.01, 1075.01]), label='FEM')


plt.xlabel('Second panel dimension (a2) [mm]')
plt.ylabel('Force (line integral) [N]')
plt.title('HP - @top3.a @15mm - gridx=1000, gridy=300')
plt.ylim(800,1200)

plt.grid()
plt.legend(ncol=2)
plt.show()