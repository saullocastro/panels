# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 14:50:26 2024

@author: Nathan

MANUAL PLOTTING CODE FOR THE RESULTS test_dcb_damage

"""
import numpy as np
import os

# Open images
from matplotlib import image as img

from matplotlib import image as img
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes, inset_axes

# To generate mp4's
import matplotlib
# matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\Users\natha\Downloads\ffmpeg-2024-04-01\ffmpeg\bin\ffmpeg.exe'

# Printing with reduced no of points (ease of viewing) - Suppress this to print in scientific notations and restart the kernel
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})





animate = False

if not animate:
    # Plotting a set of multiple results
    if False:
        # os.chdir('C:/Users/natha/OneDrive - Delft University of Technology/Fokker Internship/_Thesis/_Results/Raw Results/DCB Damage/3 pan/G1c_112e-2-v2_code')
        
        # Load variables
        # ki_all = [1e7, 5e5, 1e5, 1e4]#, 5e3, 1e3, 5e2]
        # all_filename = [f'_ki{ki_iter:.2g}_G1c112e-2' for ki_iter in ki_all]
        # for filename in all_filename:
        #     # print(f'force_intgn_{filename}.npy')
        #     globals()[f'force_intgn_{filename}'] = np.load(f'force_intgn_{filename}.npy')
        # FEM = np.load('FEM.npy')
        
        # Load FEM
        if True:
            os.chdir('C:/Users/natha/OneDrive - Delft University of Technology/Fokker Internship/_Thesis/_Results/Raw Results/DCB Damage/3 pan')
            FEM = np.load('FEM.npy')
        if True:
            os.chdir('C:/Users/natha/OneDrive - Delft University of Technology/Fokker Internship/_Thesis/_Results/Raw Results/DCB Damage/3 pan/v3_p3_m15_10_ki1e4_tauo67_wpts15 - nx_ny')
            all_filename_1 = [f'p3_m15_10_ki1e4_tauo67_nx{nx:.0f}_ny{ny:.0f}_wpts15' for nx, ny in zip([25,30,40,50,60,80],[12,15,20,25,30,40])]
            for filename in all_filename_1:
                globals()[f'force_intgn_{filename}'] = np.load(f'force_intgn_{filename}.npy')
        if False:
            os.chdir('C:/Users/natha/OneDrive - Delft University of Technology/Fokker Internship/_Thesis/_Results/Raw Results/DCB Damage/3 pan/v3_G1c_112e-2_witer_test - tau0 67 NOT 87')
            all_filename_2 = [f'p3_m15_10_ki1e4_tauo87_nx60_ny30_wpts{wpts:.0f}' for wpts in [30,45,60,80,100,120,150,180]]
            for filename in all_filename_2:
                globals()[f'force_intgn_{filename}'] = np.load(f'force_intgn_{filename}.npy')

        all_filename = all_filename_1 #+ all_filename_2
            
        if False:
            os.chdir('C:/Users/natha/OneDrive - Delft University of Technology/Fokker Internship/_Thesis/_Results/Raw Results/DCB Damage/3 pan/v3/m15')
            ki_all = [1e5, 5e4, 1e4, 5e3]
            inp_all = np.zeros((12,6)) # 16=4*4, 6 for 6 col
            m = 15
            count_inp = 0
            for ki in ki_all:
                # 4 bec each inp line as 4 lines
                inp_all[3*count_inp : 3*count_inp+3,:] = np.array((#[3,m,ki,87,150,80],
                           [3,m,ki,67,60,30],
                           [3,m,ki,57,60,30],
                           [3,m,ki,47,60,30]))
                count_inp += 1
            ftn_arg = [(int(inp_i[0]),int(inp_i[1]),f'p{inp_i[0]:.0f}_m{inp_i[1]:.0f}_10_ki{inp_i[2]:.0e}_tauo{inp_i[3]:.0f}_nx{inp_i[4]:.0f}_ny{inp_i[5]:.0f}', float(inp_i[2]), float(inp_i[3]), int(inp_i[4]), int(inp_i[5])) for inp_i in inp_all]
            # Removing +0 from the exponential notation for ki's values
            for i in range(len(ftn_arg)):
                ftn_arg[i] = list(ftn_arg[i]) # convert to list bec tuples are inmutable
                ftn_arg[i][2] = ftn_arg[i][2].replace('+0','')
                ftn_arg[i] = tuple(ftn_arg[i]) # convert back to tuple 
                
            for i_list in range(len(ftn_arg)):
                filename = ftn_arg[i_list][2]
                globals()[f'force_intgn_{filename}'] = np.load(f'force_intgn_{filename}.npy')
                
            all_filename = [el[2] for el in ftn_arg]
        
        # PLOTTING
        # All arrays should be loaded by this point
        # plt.figure(figsize=(10,7))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,7))
        for filename in all_filename:
            plt.plot(locals()[f'force_intgn_{filename}'][:,0], locals()[f'force_intgn_{filename}'][:,1], label=filename[filename.find('tauo67_') +7 : filename.find('_wpts')]) # removing starting _ from filename
        plt.plot(FEM[:,0], FEM[:,1], label='FEM')
        plt.title(r'$G_{1c}$=1.12 $w_{iter}=$15 $k_i$=1e4 $\tau_o$=67', fontsize=14)
        plt.ylabel('Force [N]', fontsize=14)
        plt.xlabel('Displacement [mm]', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid()
        plt.legend(fontsize=14)
        if True:
                ax_inset = inset_axes(ax, width="60%", height="50%",
                                       bbox_to_anchor=(.45 , .1, .8, .8),
                                       bbox_transform=ax.transAxes, loc=3)
                
                # all_filename[1:5:2] + all_filename[4::2]
                for filename in all_filename:
                    ax_inset.plot(locals()[f'force_intgn_{filename}'][:,0], locals()[f'force_intgn_{filename}'][:,1], label=filename[filename.find('ny30_') +5 : ]) # removing starting _ from filename
                ax_inset.set(xlim=[6,8], ylim=[172,202])
                ax_inset.grid()
                # ax_inset.legend()
                mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
        plt.show()
        
        
        
    # Plotting a set of single results    
    if True:
        if True:
            os.chdir('C:/Users/natha/OneDrive - Delft University of Technology/Fokker Internship/_Thesis/_Results/Raw Results/DCB Damage/3 pan')
            FEM = np.load('FEM.npy')
            
        plt.figure(figsize=(11,8))
        
        plt.plot(FEM[:,0], FEM[:,1], label='FEM')
        plt.plot(force_intgn_p3_m15_10_ki1e4_tauo67_nx50_ny25_wreverseloading[:,0], force_intgn_p3_m15_10_ki1e4_tauo67_nx50_ny25_wreverseloading[:,1], label=r'Reverse loading')
        # plt.plot(force_intgn_p3_m15_10_ki1e4_tauo67_nx50_ny25_wpts15[:,0], force_intgn_p3_m15_10_ki1e4_tauo67_nx50_ny25_wpts15[:,1], label=r'$\tau$=67 $n_x$=50 $n_y$=20 $k_i$=1e4')
        
        plt.title(r'$\tau$=67 $n_x$=50 $n_y$=25 $k_i$=1e4')
        plt.ylabel('Force [N]', fontsize=14)
        plt.xlabel('Displacement [mm]', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid()
        plt.legend(fontsize=14)
        plt.show()



# Animate results
if animate:    
    def animate(i):
        curr_res = frames[i]
        max_res = np.max(curr_res)
        min_res = np.min(curr_res)
        if animate_var == 'dmg_index' or animate_var == 'dmg_index__nx150_ny80':
            if min_res == 0:
                vmin = 0.0
                vmax = 1.0
            else: 
                possible_min_cbar = [0,0.5,0.85,0.9,0.95,0.99]
                vmin = max(list(filter(lambda x: x<min_res, possible_min_cbar))) # filters out stuff by iterating each elem of possible_min_cbar
                vmax = 1.0
            
            im = ax.imshow(curr_res)
            fig.colorbar(im, cax=cax, format="{x:.2f}")
        elif animate_var == 'del_d__ki1e+05_G1c112e-2':
            possible_min_cbar = [1e-8, 1e-4,1e-3,0.5, np.max(globals()[animate_var])*1.1]
            vmax = min(list(filter(lambda x: x>max_res, possible_min_cbar)))
            vmin = np.min(globals()[animate_var])
            
            im = ax.imshow(curr_res)
            fig.colorbar(im, cax=cax, format="{x:.2e}")
        else:
            vmin = min_res
            vmax = max_res
        # im = ax.imshow(curr_res)
        # fig.colorbar(im, cax=cax, format="{x:.2f}")
        im.set_data(curr_res)
        im.set_clim(vmin, vmax)
        tx.set_text(f'{animate_var}     -   Disp={w_iter[i]:.2f} mm')
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    for animate_var in ['dmg_index__nx150_ny80']: #["dmg_index"]:#, "del_d"]:
        
        fig = plt.figure()
        ax = fig.add_subplot(111)    
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')
        
        frames = [] # for storing the generated images
        for i in range(np.shape(locals()[animate_var])[2]):
            frames.append(locals()[animate_var][:,:,i])
            
        cv0 = frames[0]
        im = ax.imshow(cv0) 
        cb = fig.colorbar(im, cax=cax)
        tx = ax.set_title('Frame 0')
            
        ani = animation.FuncAnimation(fig, animate, frames=np.shape(locals()[animate_var])[2],
                                      interval = 1000, repeat_delay=1000)
        FFwriter = animation.FFMpegWriter(fps=2)
        ani.save(f'{animate_var}.mp4', writer=FFwriter)
        # ani.save(f'{animate_var}.gif', writer='imagemagick')