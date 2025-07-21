# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:39:42 2024

@author: Nathan
"""

FEM = 7+17/60
MD = 1.25
print((MD-FEM)/(FEM)*100)
print(FEM/MD)

print(0.86*60)

#%%

t1 = 1.14126
t2 = 1.1738
print((t2-t1)/(t1)*100)

# %% 
import numpy as np
import matplotlib.pyplot as plt

# Define variables
G_Ic = 1.12  # Fracture toughness
b = 25    # Width
h = 2.1     # thickness per DCB arm
E11 = (138300. + 128000.)/2. 
E22 = (10400. + 11500.)/2.
G13 = 5190 # Modulus of elasticity
a = 48    # Crack length
Gamma = 1.18 * np.sqrt(E11 * E22) / G13
chi =  np.sqrt((E11 / (11 * G13)) * (3 - 2 * (Gamma / (1 + Gamma))**2))

P = np.sqrt((G_Ic * b**2 * h**3 * E11) / (12 * (a + chi * h)**2))

n_step = 200
load_disp_1 = np.zeros((n_step,2))
count = 0
for P in np.linspace(0,200,n_step):
    load_disp_1[count,1] = P
    load_disp_1[count,0] = (8 * P * (a + chi * h)**3) / (b * h**3 * E11)
    count += 1

n_step_2 = 200
load_disp_2 = np.zeros((n_step_2,2))
count = 0
for a in np.linspace(30,80,n_step_2):
    P = np.sqrt((G_Ic * b**2 * h**3 * E11) / (12 * (a + chi * h)**2))
    load_disp_2[count,1] = P
    load_disp_2[count,0] = (8 * P * (a + chi * h)**3) / (b * h**3 * E11)
    count += 1

# points of load 1 at disp corresponding to disp 2
load1_disp2 = np.interp(load_disp_2[:,0], load_disp_1[:,0], load_disp_1[:,1])
# plt.plot(load_disp_2[:,0], load1_disp2)

diff_curves = load1_disp2 - load_disp_2[:,1]
pos_intst = np.argmin(np.abs(diff_curves))
disp_intst = load_disp_2[pos_intst,0] # displ at intersection
load_disp = np.vstack((load_disp_1[:np.max(np.where(load_disp_1[:,0] <= disp_intst)),:],
     load_disp_2[np.max(np.where(load_disp_2[:,0] <= disp_intst)):,:]))
# np.max(np.where(load_disp_1[:,0] <= disp_intst)))

plt.figure()
plt.plot(load_disp[:,0], load_disp[:,1], linewidth=3)
# plt.plot(load_disp_1[:,0], load_disp_1[:,1]) # individual curve Linear
# plt.plot(load_disp_2[:,0], load_disp_2[:,1]) # individual curve Propagation
plt.grid()
plt.xlabel('Displacement [mm]', fontsize=14)
plt.ylabel('Force [N]', fontsize=14)
plt.title('Hasehemi LEFM - 10.1098/RSPA.1990.0007')
plt.show()

# %%

fig = plt.figure(figsize=(3,2))
ax = plt.gca()
ax.grid(True)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Normalized Defect Length $\delta/a$')
ax.yaxis.labelpad = 20
ax.set_ylabel('Normalized Critical Buckling Load [$N/m$]', rotation='horizontal')
ax.plot(x, y, 'b-')
# fig.savefig(filename='lb.png', bbox_inches='tight')


# %%
plt.figure(figsize=(6,5))
ax = plt.gca()
plt.plot(x, y, 'b.-', linewidth=3)
plt.xlabel('Normalized Defect Length ($\delta/a$)', fontsize=14)
plt.ylabel('Normalized Critical Buckling Load', fontsize=14)
plt.ylim([-0.05,1.05])
plt.grid()
plt.show()

