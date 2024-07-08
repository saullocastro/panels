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





animate = True

if not animate:
    # Plotting a set of multiple results
    if True:
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
            os.chdir('C:/Users/natha/OneDrive - Delft University of Technology/Fokker Internship/_Thesis/_Results/Raw Results/DCB Damage/v4 code - force crack/p3_m15_8_ki1e4_tauo67_nx60_ny30_wptsITER_G1c112')
            all_filename_1 = [f'p3_m15_8_ki1e4_tauo67_nx60_ny30_wpts{wpts:.0f}_G1c112' for wpts in [25,30,45,50]]
            for filename in all_filename_1:
                globals()[f'force_intgn_{filename}'] = np.load(f'force_intgn_{filename}.npy')
                globals()[f'force_intgn_dmg_{filename}'] = np.load(f'force_intgn_dmg_{filename}.npy')

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
            # plt.plot(locals()[f'force_intgn_{filename}'][:,0], locals()[f'force_intgn_{filename}'][:,1], 
            #          label=filename[filename.find('tauo67_') +7 : filename.find('_wpts')]) # removing starting _ from filename
            if False: # For nx ny iter
                label_unformatted = filename[filename.find('tauo67_') +7 : filename.find('_wpts')]
                label_formatted = label_unformatted.replace('_', ' ')
                label_formatted = label_formatted.replace('nx', '$n_x$=')
                label_formatted = label_formatted.replace('ny', '$n_y$=')
            if True:
                label_unformatted = filename[filename.find('ny30_') +5 : filename.find('_G1c')]
                label_formatted = label_unformatted.replace('_', ' ')
                label_formatted = label_formatted.replace('wpts', '$w_{iter}$=')
            plt.plot(locals()[f'force_intgn_dmg_{filename}'][:,0], 23*locals()[f'force_intgn_dmg_{filename}'][:,1], 
                      label=label_formatted)
        plt.plot(FEM[:,0], FEM[:,1], label='FEM')
        plt.title(r'$p$3 $m$=15,8 $k_i$=1e4 $\tau_o$=67 $G_{1c}$=1.12 $n_x$=60 $n_x$=30 scaled=23', fontsize=14)
        plt.ylabel('Force [N]', fontsize=14)
        plt.xlabel('Displacement [mm]', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid()
        plt.legend(fontsize=14)
        if False:
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
    if False:
        if True:
            os.chdir('C:/Users/natha/OneDrive - Delft University of Technology/Fokker Internship/_Thesis/_Results/Raw Results/DCB Damage/3 pan')
            FEM = np.load('FEM.npy')
            
        plt.figure(figsize=(10,7))
        
        plt.plot(FEM[:,0], FEM[:,1], label='FEM')
        
        plt.plot(force_intgn_dmg_1posve_m15_8_w50_nx60_ny30[:42,0], 22.5*force_intgn_dmg_1posve_m15_8_w50_nx60_ny30[:42,1], 
                 label=r'Area int (scaled 22.5)')
        plt.plot(force_intgn_1posve_m15_8_w50_nx60_ny30[:42,0], force_intgn_1posve_m15_8_w50_nx60_ny30[:42,1], 
                 label=r'Line int')
        
        plt.title(r'+1*fcrack $\tau$=67 $n_x$=60 $n_y$=30 $k_i$=1e4 m =15,8 w=50 - UNCONVERGED')
        plt.ylabel('Force [N]', fontsize=14)
        plt.xlabel('Displacement [mm]', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid()
        plt.legend(fontsize=14)
        plt.show()
      
    # Random plot 2
    if False:
        if True:
            os.chdir('C:/Users/natha/OneDrive - Delft University of Technology/Fokker Internship/_Thesis/_Results/Raw Results/DCB Damage/3 pan')
            FEM = np.load('FEM.npy')
            
        plt.figure(figsize=(10,7))
        
        plt.plot(FEM[:,0], FEM[:,1], label='FEM')
        
        plt.plot(force_intgn_dmg_1posve_m15_8_w50_nx60_ny30[:42,0], 22.5*force_intgn_dmg_1posve_m15_8_w50_nx60_ny30[:42,1], 
                 label=r'Area (scaled 22.5) - $n_x$=60 $n_y$=30 w=50')
        plt.plot(force_intgn_1posve_m15_8_w50_nx60_ny30[:42,0], force_intgn_1posve_m15_8_w50_nx60_ny30[:42,1], 
                 label=r'Line - $n_x$=60 $n_y$=30 w=50')
        
        plt.plot(force_intgn_dmg_1posve_m15_8[:,0], 22.5*force_intgn_dmg_1posve_m15_8[:,1], 
                 label=r'Area (scaled 22.5) - $n_x$=50 $n_y$=25 w=13')
        plt.plot(force_intgn_1posve_m15_8[:42,0], force_intgn_1posve_m15_8[:,1], 
                 label=r'Line - $n_x$=50 $n_y$=25 w=13')
        
        plt.plot(force_intgn_dmg_1posve_m15_8[:,0], 22.5*force_intgn_dmg_1posve_m15_8[:,1], 
                 label=r'Area (scaled 22.5) - $n_x$=50 $n_y$=25 w=13')
        plt.plot(force_intgn_1posve_m15_8[:42,0], force_intgn_1posve_m15_8[:,1], 
                 label=r'Line - $n_x$=50 $n_y$=25 w=13')
        
        plt.plot(force_intgn_p3_m15_10_ki1e4_tauo67_nx60_ny30[:,0], force_intgn_p3_m15_10_ki1e4_tauo67_nx60_ny30[:,1], 
                 label='Previous version')
        
        plt.title(r'+1*fcrack $\tau$=67  $k_i$=1e4 m =15,8 - UNCONVERGED')
        plt.ylabel('Force [N]', fontsize=14)
        plt.xlabel('Displacement [mm]', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid()
        plt.legend(fontsize=12)
        plt.show()
        


# Animate results
if animate:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
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
        elif animate_var == 'tau_p3_m15_8_ki1e4_tauo67_nx60_ny30_wpts50_G1c112':
            vmin = np.min(tau_p3_m15_8_ki1e4_tauo67_nx60_ny30_wpts50_G1c112)
            vmax = np.max(tau_p3_m15_8_ki1e4_tauo67_nx60_ny30_wpts50_G1c112)
            im = ax.imshow(curr_res)
            fig.colorbar(im, cax=cax, format="{x:.2f}")
        else:
            vmin = min_res
            vmax = max_res
        # im = ax.imshow(curr_res)
        # fig.colorbar(im, cax=cax, format="{x:.2f}")
        im.set_data(curr_res)
        im.set_clim(vmin, vmax)
        
        # w_iter needs to be defined
        tx.set_text(f'{animate_var}     -   Disp={w_iter[i]:.2f} mm')
    
    for animate_var in ['tau_p3_m15_8_ki1e4_tauo67_nx60_ny30_wpts50_G1c112']: #["dmg_index"]:#, "del_d"]:
        
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