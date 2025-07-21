# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 19:32:10 2024

@author: Nathan
"""

import numpy as np
import matplotlib.pyplot as plt

# Equations used from DOI: 10.1016/J.COMPOSITESA.2022.107101. as it was easier but theyre
# the same as Hashemi 1990 10.1098/RSPA.1990.0007

# Define variables
G_Ic = 1.12  # Fracture toughness
b = 25    # Width
h = 2.1     # thickness per DCB arm
E11 = (138300. + 128000.)/2. 
E22 = (10400. + 11500.)/2.
G13 = 5190 # Modulus of elasticity
a = 48    # Crack length
n_step = 200 # no step LEFM
n_step_2 = 200 # no step propagation

Gamma = 1.18 * np.sqrt(E11 * E22) / G13
chi =  np.sqrt((E11 / (11 * G13)) * (3 - 2 * (Gamma / (1 + Gamma))**2))

P = np.sqrt((G_Ic * b**2 * h**3 * E11) / (12 * (a + chi * h)**2))
disp = (8 * P * (a + chi * h)**3) / (b * h**3 * E11)
print(f'Load: {P:.3f} N -- Disp: {disp:.3f} [mm]')


load_disp_1 = np.zeros((n_step,2))
count = 0
for P in np.linspace(0,200,n_step): # CHANGE THIS TO INCREASE RANGE OF LINEAR PART
    load_disp_1[count,1] = P
    load_disp_1[count,0] = (8 * P * (a + chi * h)**3) / (b * h**3 * E11)
    count += 1

n_step_2 = 200
load_disp_2 = np.zeros((n_step_2,2))
count = 0
for a in np.linspace(30,80,n_step_2): # CHANGE THIS TO INCREASE RANGE OF PROPAGATION PART
# low precrack changes start point of the curve, high precrack changes end
    P = np.sqrt((G_Ic * b**2 * h**3 * E11) / (12 * (a + chi * h)**2))
    load_disp_2[count,1] = P
    load_disp_2[count,0] = (8 * P * (a + chi * h)**3) / (b * h**3 * E11)
    count += 1

# points of load 1 at disp corresponding to disp 2
load1_disp2 = np.interp(load_disp_2[:,0], load_disp_1[:,0], load_disp_1[:,1])

diff_curves = load1_disp2 - load_disp_2[:,1]
pos_intst = np.argmin(np.abs(diff_curves))
disp_intst = load_disp_2[pos_intst,0] # displ at intersection
load_disp = np.vstack((load_disp_1[:np.max(np.where(load_disp_1[:,0] <= disp_intst)),:],
     load_disp_2[np.max(np.where(load_disp_2[:,0] <= disp_intst)):,:]))

plt.figure()
plt.plot(load_disp[:,0], load_disp[:,1], linewidth=3)
# plt.plot(load_disp_1[:,0], load_disp_1[:,1]) # individual curve Linear
# plt.plot(load_disp_2[:,0], load_disp_2[:,1]) # individual curve Propagation
plt.grid()
plt.xlabel('Displacement [mm]', fontsize=14)
plt.ylabel('Force [N]', fontsize=14)
plt.title('Hasehemi LEFM - 10.1098/RSPA.1990.0007')
plt.show()