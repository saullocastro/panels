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
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib.ticker import FormatStrFormatter

# To generate mp4's
import matplotlib
# matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\Users\natha\Downloads\ffmpeg-2024-04-01\ffmpeg\bin\ffmpeg.exe'

# Printing with reduced no of points (ease of viewing) - Suppress this to print in scientific notations and restart the kernel
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

# To open pop up images - Ignore the syntax warning :)
# %matplotlib qt 
# For inline images
# %matplotlib inline

os.chdir('C:/Users/natha/Documents/GitHub/panels')
from panels.legendre_gauss_quadrature import get_points_weights




animate = False

#%% Multiple results for a SINGLE FEM
if not animate:
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
            FEM_foldername = 'All 0s'
            FEM_filename = 'L65_a48'
            os.chdir('C:/Users/natha/OneDrive - Delft University of Technology/Fokker Internship/_Thesis/_Results/Raw Results/DCB Damage/FEM/' + FEM_foldername)
            FEM = np.load(f'FEM_{FEM_filename}.npy')
        if True:
            foldername = 'p3_65_ITERm12_8k1e4_t67_nx50y25_w60_G112'
            os.chdir('C:/Users/natha/OneDrive - Delft University of Technology/Fokker Internship/_Thesis/_Results/Raw Results/DCB Damage/v7 - DI converging/' + foldername)
            # print(os.getcwd())
            
            all_filename_1 = [(f'p3_65_{b:.0f}_{precrack:.0f}m12_8k1e4_t67_nx50y25_w60_G112')
                       for b in [1,5,25,50] for precrack in [48]]

            for i in range(len(all_filename_1)):
                all_filename_1[i] = all_filename_1[i].replace('+0','')

            for filename in all_filename_1:
                globals()[f'Fdmg_{filename}'] = np.load(f'Fdmg_{filename}.npy')
                globals()[f'F_{filename}'] = np.load(f'F_{filename}.npy')

        all_filename = all_filename_1 #+ all_filename_2
        

        # PLOTTING
        # All arrays should be loaded by this point
        # plt.figure(figsize=(10,7))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,7))
        
        title_name = all_filename[0]
        title_name = title_name.replace('m',r' $m$=')
        title_name = title_name.replace('k',r' $k_i$=')
        title_name = title_name.replace('_t',r' $\tau_o$=')
        title_name = title_name.replace('_nx',r' $n_x$=')
        title_name = title_name.replace('y',r' $n_y$=')
        title_name = title_name.replace('_w',r' $w=$')
        title_name = title_name.replace('_G',r' $G_{Ic}$=')
        title_name = title_name.replace('112','1.12')
        
        edit_title = True
        all_labels = []
        filename_count = 0
        
        plt.plot(FEM[:,0], FEM[:,1], label='FEM', linewidth=3)
        # plt.plot(analytical_b25_a48[:,0], analytical_b25_a48[:,1], label='analytical_b25_a48', linewidth=3)
        
        for filename in all_filename:
            # plt.plot(locals()[f'force_intgn_{filename}'][:,0], locals()[f'force_intgn_{filename}'][:,1], 
            #           label='Line b='+filename[filename.find('p3_') + 3 : filename.find('_m')]) # removing starting _ from filename
            
            if False: # For nx ny iter
                label_unformatted = filename[filename.find('_nx')+0 : filename.find('y')+3]
                label_formatted = label_unformatted.replace('_nx', '$n_x$=')
                label_formatted = label_formatted.replace('y', ', $n_y$=')
                if edit_title:
                    title_name = title_name[0:title_name.find('n_x')-2] + title_name[title_name.find('y')+5:]
                    edit_title = False
            if True: # tauo
                label_unformatted = filename[filename.find('_t')+0 : filename.find('_nx')]
                label_formatted = label_unformatted.replace('_t', r"$\tau_o$ = ")
                if edit_title:
                    title_name = title_name[0:title_name.find('tau_o')-2] + title_name[title_name.find('tau_o$')+10:]
                    edit_title = False
            if False: # ki
                label_unformatted = filename[filename.find('k')+0 : filename.find('_t')]
                label_formatted = label_unformatted.replace('k', r"$k_i$ =  ")
                # label_formatted = label_formatted.replace('e', f"$e^{filename[filename.find('_t')-1]}$")
                if edit_title:
                    title_name = title_name[0:title_name.find('k_i')-2] + title_name[title_name.find('k_i$')+8:]
                    edit_title = False
            if False: # b
                label_unformatted = filename[filename.find('p3_') + 2 : filename.find('_m')]
                label_formatted = label_unformatted.replace('_', r"$b$ = ")
            if False: # w
                label_unformatted = filename[filename.find('_w')+0 : filename.find('_G')]
                label_formatted = label_unformatted.replace('_w', r"$w$ = ")
                if edit_title:
                    title_name = title_name[0:title_name.find('w')-2] + title_name[title_name.find('w')+5:]
                    edit_title = False
            if False: # m
                label_unformatted = filename[filename.find('m')+0 : filename.find('k')-2]
                label_formatted = label_unformatted.replace('m', r"$m$=")
                label_formatted = label_formatted.replace('_', r",")
                if edit_title:
                    print(title_name)
                    title_name = title_name[0:title_name.find('m')-2] + title_name[title_name.find('m')+14:]
                    edit_title = False
                
            all_labels.append(label_formatted)
            # scaling = 14
            # scaling_fact = np.array([1/25, 1/5, 1, 2])*14
            # scaling = scaling_fact[filename_count]
            scaling = globals()[f'F_{filename}'][0,1]/globals()[f'Fdmg_{filename}'][0,1]
            print(1/scaling)
            plt.plot(globals()[f'Fdmg_{filename}'][:,0], (scaling)*globals()[f'Fdmg_{filename}'][:,1], 
                      label=label_formatted) # removing starting _ from filename
            # plt.plot(locals()[f'F_{filename}'][:,0], locals()[f'F_{filename}'][:,1], 
            #           label=label_formatted)
            
            filename_count += 1
        
        
        plt.title(f'{title_name} scaled', fontsize=14)
        plt.ylabel('Force [N]', fontsize=14)
        plt.xlabel('Displacement [mm]', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid()
        # plt.xlim(0,4)
        # plt.ylim(0,125)
        plt.legend(fontsize=14, loc='upper left')
        if False:
                ax_inset = inset_axes(ax, width="70%", height="60%",
                                       bbox_to_anchor=(.45 , .1, .8, .8),
                                       bbox_transform=ax.transAxes, loc=3)
                ax_inset.plot(FEM[:,0], FEM[:,1], label='FEM', linewidth=3)
                
                # all_filename[1:5:2] + all_filename[4::2]
                for count in range(0,np.shape(all_filename)[0],1): # change last arg to see every n result
                    filename = all_filename[count]
                    ax_inset.plot(locals()[f'Fdmg_{filename}'][:,0], (1/scaling)*locals()[f'Fdmg_{filename}'][:,1], 
                              label=all_labels[count])
                ax_inset.set(xlim=[6,8], ylim=[155,178])
                ax_inset.grid()
                # ax_inset.legend()
                mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
        plt.show()
        
        
        
    #%% Multiple results for a MULTIPLE FEM
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
            FEM_foldername = 'Theodore'
            FEM_filename = [(f'Theo') for b in [1]]
            os.chdir('C:/Users/natha/OneDrive - Delft University of Technology/Fokker Internship/_Thesis/_Results/Raw Results/DCB Damage/FEM/' + FEM_foldername)
            for all_FEM_filename in FEM_filename:
                globals()[f'FEM_{all_FEM_filename}'] = np.load(f'FEM_{all_FEM_filename}.npy')
        if True:
            foldername = 'Theo_kITR_tITR_p3_90_255_566m10_8_nx60y30_w70_G141'
            os.chdir('C:/Users/natha/OneDrive - Delft University of Technology/Fokker Internship/_Thesis/_Results/Raw Results/DCB Damage/v7 - DI converging/' + foldername)
            # print(os.getcwd())
            
            all_filename_1 = [(f'p3_90_255_566m10_8k{ki:.0e}_t{tau:.0f}_nx60y30_w70_G141')
                      for ki in [5e4] for tau in [47,67,87,107,127,137,147,157] ]
            # [5e3, 1e4, 5e4, 1e5, 5e5, 1e6]
            # [47,67,87,107,127,137,147,157,167,177,187]

            for i in range(len(all_filename_1)):
                all_filename_1[i] = all_filename_1[i].replace('+0','')

            for filename in all_filename_1:
                globals()[f'Fdmg_{filename}'] = np.load(f'Fdmg_{filename}.npy')
                globals()[f'F_{filename}'] = np.load(f'F_{filename}.npy')

        all_filename = all_filename_1 #+ all_filename_2
        

        # PLOTTING
        # All arrays should be loaded by this point
        # plt.figure(figsize=(10,7))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,7))
        
        title_name = all_filename[0]
        title_name = title_name.replace('m',r' $m$=')
        title_name = title_name.replace('k',r' $k_i$=')
        title_name = title_name.replace('_t',r' $\tau_o$=')
        title_name = title_name.replace('_nx',r' $n_x$=')
        title_name = title_name.replace('y',r' $n_y$=')
        title_name = title_name.replace('_w',r' $w$=')
        title_name = title_name.replace('_G',r' $G_{Ic}$=')
        title_name = title_name.replace('112','1.12')
        
        # Removing _ from initial part of the title
        title_name = title_name.replace('p',r' $p$=',1)
        title_name = title_name.replace('_',r' $L$=',1)
        title_name = title_name.replace('_',r' $b$=',1)
        title_name = title_name.replace('_',r' $a_0$=',1)
        
        # Manually correcting for .
        title_name = title_name.replace('255','25.5')
        title_name = title_name.replace('566','56.6')
        title_name = title_name.replace('141','1.41')
        
        edit_title = True
        all_labels = []
        filename_count = 0
        
        for all_FEM_filename in FEM_filename:
            FEM_label = all_FEM_filename.replace('L', '$L$=')
            FEM_label = FEM_label.replace('a', '$a_0$=')
            FEM_label = FEM_label.replace('b', '$b$=')
            
            if False: # a0
                FEM_label = FEM_label[FEM_label.find('a')-2 : FEM_label.find('a')+7]
            if False: # b
                FEM_label = FEM_label[FEM_label.find('b')-1 : FEM_label.find('a')-2]
            if True:
                FEM_label = ''
            
            if False:
                plt.plot(globals()[f'FEM_{all_FEM_filename}'][:,0], globals()[f'FEM_{all_FEM_filename}'][:,1], 
                     label='FEM '+FEM_label, linewidth=3)
        
        # plt.plot(analytical_b25_a48[:,0], analytical_b25_a48[:,1], label='analytical_b25_a48', linewidth=3)
        
        for filename in all_filename:
            # plt.plot(locals()[f'force_intgn_{filename}'][:,0], locals()[f'force_intgn_{filename}'][:,1], 
            #           label='Line b='+filename[filename.find('p3_') + 3 : filename.find('_m')]) # removing starting _ from filename
            
            if False: # For nx ny iter
                label_unformatted = filename[filename.find('_nx')+0 : filename.find('y')+3]
                label_formatted = label_unformatted.replace('_nx', '$n_x$=')
                label_formatted = label_formatted.replace('y', ', $n_y$=')
                if edit_title:
                    title_name = title_name[0:title_name.find('n_x')-2] + title_name[title_name.find('y')+5:]
                    edit_title = False
            if True: # tauo
                label_unformatted = filename[filename.find('_t')+0 : filename.find('_nx')]
                label_formatted = label_unformatted.replace('_t', r"$\tau_o$ = ")
                if edit_title:
                    title_name = title_name[0:title_name.find('tau_o')-2] + title_name[title_name.find('tau_o$')+10:]
                    edit_title = False
            if False: # ki
                label_unformatted = filename[filename.find('k')+0 : filename.find('_t')]
                label_formatted = label_unformatted.replace('k', r"$k_i$ =  ")
                # label_formatted = label_formatted.replace('e', f"$e^{filename[filename.find('_t')-1]}$")
                if edit_title:
                    title_name = title_name[0:title_name.find('k_i')-2] + title_name[title_name.find('k_i$')+8:]
                    edit_title = False
            if False: # b
                label_unformatted = filename[filename.find('p') +6 : filename.find('m')-3]
                label_formatted = r'$b$='+label_unformatted
                if edit_title:
                    title_name = title_name[0:title_name.find('b')-2] + title_name[title_name.find('b$')+4:]
                    edit_title = False
            if False: # w
                label_unformatted = filename[filename.find('_w')+0 : filename.find('_G')]
                label_formatted = label_unformatted.replace('_w', r"$w$ = ")
                if edit_title:
                    title_name = title_name[0:title_name.find('w')-2] + title_name[title_name.find('w')+5:]
                    edit_title = False
            if False: # m
                label_unformatted = filename[filename.find('m')+0 : filename.find('k')-2]
                label_formatted = label_unformatted.replace('m', r"$m$=")
                label_formatted = label_formatted.replace('_', r",")
                if edit_title:
                    title_name = title_name[0:title_name.find('m')-2] + title_name[title_name.find('m')+14:]
                    edit_title = False
            if False: # a0
                label_unformatted = filename[filename.find('m')-3 : filename.find('m')]
                label_formatted = label_unformatted.replace('_', r"$a_0$=")
                if edit_title:
                    title_name = title_name[0:title_name.find('a')-2] + title_name[title_name.find('a')+7:]
                    edit_title = False
                
            all_labels.append(label_formatted)
            # scaling = 1/35
            # scaling_fact = np.array([1/25, 1/5, 1, 2])*14
            # scaling = scaling_fact[filename_count]
            scaling = globals()[f'F_{filename}'][0,1]/globals()[f'Fdmg_{filename}'][0,1]
            plt.plot(globals()[f'Fdmg_{filename}'][:,0], (scaling)*globals()[f'Fdmg_{filename}'][:,1], 
                      label='MD '+label_formatted) # removing starting _ from filename
            # plt.plot(locals()[f'F_{filename}'][:,0], locals()[f'F_{filename}'][:,1], 
            #           label=label_formatted)
            
            filename_count += 1
        
        
        plt.title(f'{title_name} \n scaling included', fontsize=14)
        plt.ylabel('Force [N]', fontsize=14)
        plt.xlabel('Displacement [mm]', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid()
        # plt.xlim(0,4)
        # plt.ylim(0,20)
        plt.legend(fontsize=14, loc='upper left')
        if False: # inset for refinement around prop for 1 FEM
                ax_inset = inset_axes(ax, width="70%", height="60%",
                                       bbox_to_anchor=(.45 , .1, .8, .8),
                                       bbox_transform=ax.transAxes, loc=3)
                ax_inset.plot(FEM[:,0], FEM[:,1], label='FEM', linewidth=3)
                
                # all_filename[1:5:2] + all_filename[4::2]
                for count in range(0,np.shape(all_filename)[0],1): # change last arg to see every n result
                    filename = all_filename[count]
                    ax_inset.plot(locals()[f'Fdmg_{filename}'][:,0], (1/scaling)*locals()[f'Fdmg_{filename}'][:,1], 
                              label=all_labels[count])
                ax_inset.set(xlim=[6,8], ylim=[155,178])
                ax_inset.grid()
                # ax_inset.legend()
                mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
        if False: # Inset for other places
                ax_inset = inset_axes(ax, width="60%", height="20%",
                                       bbox_to_anchor=(.4, .1, 1, 1),
                                       bbox_transform=ax.transAxes, loc=3)
                ax_inset.plot(FEM[:,0], FEM[:,1], label='FEM', linewidth=3)
                ax_inset.set_prop_cycle(None)
                for all_FEM_filename in FEM_filename:
                    plt.plot(globals()[f'FEM_{all_FEM_filename}'][:,0], globals()[f'FEM_{all_FEM_filename}'][:,1], 
                             label='FEM '+FEM_label, linewidth=3)
                # all_filename[1:5:2] + all_filename[4::2]
                for count in range(0,np.shape(all_filename)[0],1): # change last arg to see every n result
                    filename = all_filename[count]
                    scaling = globals()[f'F_{filename}'][0,1]/globals()[f'Fdmg_{filename}'][0,1]
                    ax_inset.plot(globals()[f'Fdmg_{filename}'][:,0], (scaling)*globals()[f'Fdmg_{filename}'][:,1], 
                              label='MD '+label_formatted)
                
                ax_inset.set(xlim=[0,8], ylim=[0,7.5])
                ax_inset.grid()
                # ax_inset.legend()
                mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
        plt.show()
            
        
        
    #%% Plotting a set of single results    
    if False:
        if True:
            os.chdir('C:/Users/natha/OneDrive - Delft University of Technology/Fokker Internship/_Thesis/_Results/Raw Results/DCB Damage/3 pan')
            FEM = np.load('FEM.npy')
            
        plt.figure(figsize=(10,7))
        
        plt.plot(FEM[:,0], FEM[:,1], label='FEM')
        
        var_name = 'p3_65_25_48m12_8k1e4_t67_nx50y25_w60_G112'
        
        plt.plot(globals()['Fdmg_'+var_name][:,0], 
                 (1/14)*globals()['Fdmg_'+var_name][:,1], 
                  label=r'Area int (scaled 1/14.5)')
        
        plt.plot(globals()['Fdmg_'+var_name][:,0], 
                 globals()['Fdmg_'+var_name][:,1], 
                  label=r'Unscaled')
        
        # plt.plot(F_test_12_30_fine[:,0], 
        #          F_test_12_30_fine[:,1], 
        #           label=r'Line int (unscaled)')
        
        plt.title(r'$\tau$=67 $n_x$=60 $n_y$=30 $k_i$=1e4 $m$=15,8 w=30')
        plt.ylabel('Force [N]', fontsize=14)
        plt.xlabel('Displacement [mm]', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid()
        plt.legend(fontsize=14)
        plt.ylim(0,200)
        plt.show()
        
    #%% contourf
    if True:
        var_name = 'DI_noresetdispl_bas'#'kw_tsl_1posve_m15_8_w200_nx60_ny30_tau67_ki1e4'
        
        no_x_gauss = 60
        no_y_gauss = 30
        
        a = 17
        b = 25
        
        xis = np.zeros(no_x_gauss, dtype=np.float64)
        weights_xi = np.zeros(no_x_gauss, dtype=np.float64)
        etas = np.zeros(no_y_gauss, dtype=np.float64)
        weights_eta = np.zeros(no_y_gauss, dtype=np.float64)
        
        get_points_weights(no_x_gauss, xis, weights_xi)
        get_points_weights(no_y_gauss, etas, weights_eta)
        
        xi_grid, eta_grid = np.meshgrid(xis, etas)
        
        x_grid = a*(xi_grid+1)/2
        y_grid = b*(eta_grid+1)/2
        
        fig, ax = plt.subplots()
        # levels = [0,0.3,0.4,0.5,0.55,0.57,np.max(globals()[var_name][:,:,17])]
        levels = None
        plt.contourf(x_grid, y_grid, globals()[var_name][:,:,25], levels=levels)
        cbar = plt.colorbar()
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        # plt.title(r'+1*fcrack $\tau$=67 $n_x$=60 $n_y$=30 $k_i$=1e4 m =15,8 w=200')
        plt.title('Damage Index \n Tip Displacement = 3.94 [mm]', fontsize=14)
        plt.ylabel('y coordinate [mm]', fontsize=14)
        plt.xlabel('x coordinate [mm]', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if True:
            # Y - Ensuring that the ends have tick labels
            org_yticks = ax.get_yticks() # Get the current y-tick positions - numpy array 
            org_ytick_labels = [item.get_text() for item in ax.get_yticklabels()] # Get the current y-tick labels - LIST
            new_yticks = np.sort(np.append(org_yticks, [0,b]))
            new_yticks = np.unique(new_yticks[new_yticks<=b]) # removing all ticks >b
            new_ytick_labels = new_yticks.astype(str).tolist()
            ax.set_yticks(new_yticks)
            ax.set_yticklabels(new_ytick_labels, fontsize=14)
            # X - Ensuring that the ends have tick labels
            org_xticks = ax.get_xticks() # Get the current y-tick positions - numpy array 
            org_xtick_labels = [item.get_text() for item in ax.get_xticklabels()] # Get the current y-tick labels - LIST
            new_xticks = np.sort(np.append(org_xticks, [0,a]))
            new_xticks = np.unique(new_xticks[new_xticks<=a]) # removing all ticks >a
            new_xtick_labels = new_xticks.astype(str).tolist()
            ax.set_xticks(new_xticks)
            ax.set_xticklabels(new_xtick_labels, fontsize=14)
        # plt.grid()
        plt.show()
        
        # >0
        if False:
            fig, ax = plt.subplots()
            plt.contourf(x_grid, y_grid, globals()[var_name][:,:,30]>=0)
            plt.colorbar()
            # plt.title(r'+1*fcrack $\tau$=67 $n_x$=60 $n_y$=30 $k_i$=1e4 m =15,8 w=200')
            plt.title('Relative separtion >= 0 \n Tip Displacement = 4.53 [mm]', fontsize=14)
            plt.ylabel('y coordinate [mm]', fontsize=14)
            plt.xlabel('x coordinate [mm]', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            if True:
                # Y - Ensuring that the ends have tick labels
                org_yticks = ax.get_yticks() # Get the current y-tick positions - numpy array 
                org_ytick_labels = [item.get_text() for item in ax.get_yticklabels()] # Get the current y-tick labels - LIST
                new_yticks = np.sort(np.append(org_yticks, [0,b]))
                new_yticks = new_yticks[new_yticks<=b] # removing all ticks >b
                new_ytick_labels = new_yticks.astype(str).tolist()
                ax.set_yticks(new_yticks)
                ax.set_yticklabels(new_ytick_labels, fontsize=14)
                # X - Ensuring that the ends have tick labels
                org_xticks = ax.get_xticks() # Get the current y-tick positions - numpy array 
                org_xtick_labels = [item.get_text() for item in ax.get_xticklabels()] # Get the current y-tick labels - LIST
                new_xticks = np.sort(np.append(org_xticks, [0,a]))
                new_xticks = new_xticks[new_xticks<=a] # removing all ticks >a
                new_xtick_labels = new_xticks.astype(str).tolist()
                ax.set_xticks(new_xticks)
                ax.set_xticklabels(new_xtick_labels, fontsize=14)
            # plt.grid()
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
        
















#%% Animate results
if animate:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    def animate_contourf(i):
        # i = current frame number
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
        
        w_iter = globals()['force_intgn_dmg'+animate_var][:,0]
        # w_iter needs to be defined
        tx.set_text(f'{animate_var}     -   Disp={w_iter[i]:.2f} mm')
      
    
    def animate_contourf_v2(i):
        # i = current frame number
        
        curr_res = frames[i]
        vmin = np.min(globals()[var_name+animate_var])
        vmax = np.max(globals()[var_name+animate_var])
        im.set_array(curr_res.ravel())
        # im = ax1.pcolormesh(curr_res)
        # im.set_array(curr_res.ravel())
        fig.colorbar(im, cax=cax, format="{x:.2f}", ticks=np.linspace(vmin, vmax, 5))
        # im.set_data(curr_res)
        im.set_clim(vmin, vmax)
        
        # w_iter needs to be defined
        w_iter = globals()['force_intgn_dmg'+animate_var][:,0]
        tx.set_text(f'Interface Traction    -   Disp={w_iter[i]:.2f} mm')
        
        # return scatter_plot
    
        
    def animate_plot_scatter(i):
        scatter_plot.set_offsets(np.stack([globals()[animate_var][i,0], scaling_fct*globals()[animate_var][i,1]]).T)
        
        # line_plot.set_xdata(globals()[animate_var][:i+1,0])
        # line_plot.set_ydata(scaling_fct*globals()[animate_var][:i+1,1])
        
        # return (scatter_plot)#, line_plot)
        
    
    def animate_contourf_scatter_line(i):
        # i = current frame number
        
        scatter_plot.set_offsets(np.stack([globals()['force_intgn_dmg'+animate_var][i,0], 
                                           scaling_fct*globals()['force_intgn_dmg'+animate_var][i,1]]).T)
        
        curr_res = frames[i]
        vmin = np.min(globals()[var_name+animate_var])
        vmax = np.max(globals()[var_name+animate_var])
        im = ax1.imshow(curr_res)
        # im = ax1.pcolormesh(curr_res)
        # im.set_array(curr_res.ravel())
        fig.colorbar(im, cax=cax, format="{x:.2f}", ticks=np.linspace(vmin, vmax, 5))
        # im.set_data(curr_res)
        # im.set_clim(vmin, vmax)
        
        # w_iter needs to be defined
        w_iter = globals()[animate_var][:,0]
        tx.set_text(f'Interface Traction    -   Disp={w_iter[i]:.2f} mm')
        
        # return scatter_plot
    
    # Creates contourf plots
    if False:
        var_name = 'tau'
        for animate_var in ['_p3_m15_8_ki1e4_tauo67_nx60_ny30_wpts50_G1c112']: #["dmg_index"]:#, "del_d"]:
            
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)    
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', '5%', '5%')
            
            frames = [] # for storing the generated images
            for i in range(np.shape(locals()[animate_var])[2]):
                frames.append(locals()[animate_var][:,:,i])
                
            cv0 = frames[0]
            im = ax.imshow(cv0) 
            cb = fig.colorbar(im, cax=cax)
            tx = ax.set_title('Frame 0') # Gets overwritten later
                
            ani = animation.FuncAnimation(fig, animate_contourf, frames=np.shape(locals()[animate_var])[2],
                                          interval = 1000, repeat_delay=1000)
            FFwriter = animation.FFMpegWriter(fps=2)
            ani.save(f'{animate_var}.mp4', writer=FFwriter)
            # ani.save(f'{animate_var}.gif', writer='imagemagick')
    
        
    
    # V2 - Creates contourf plots
    if True:
        no_x_gauss = 60
        no_y_gauss = 30
        
        a = 52
        b = 25
        
        xis = np.zeros(no_x_gauss, dtype=np.float64)
        weights_xi = np.zeros(no_x_gauss, dtype=np.float64)
        etas = np.zeros(no_y_gauss, dtype=np.float64)
        weights_eta = np.zeros(no_y_gauss, dtype=np.float64)
        
        get_points_weights(no_x_gauss, xis, weights_xi)
        get_points_weights(no_y_gauss, etas, weights_eta)
        
        xi_grid, eta_grid = np.meshgrid(xis, etas)
        
        x_grid = a*(xi_grid+1)/2
        y_grid = b*(eta_grid+1)/2
        
        
        def animate_contourf_v2(i):
            # i = current frame number
            
            curr_res = frames[i][:,40:]
            vmin = np.min(globals()[var_name+animate_var])
            vmax = np.max(globals()[var_name+animate_var])
            im.set_array(curr_res.ravel())
            # im = ax1.pcolormesh(curr_res)
            # im.set_array(curr_res.ravel())
            fig.colorbar(im, cax=cax, format="{x:.2f}", ticks=np.linspace(vmin, vmax, 5))
            # im.set_data(curr_res)
            im.set_clim(vmin, vmax)
            
            # w_iter needs to be defined
            w_iter = globals()['force_intgn_dmg'+animate_var][:,0]
            tx.set_text(f'{var_name}   -   Disp={w_iter[i]:.2f} mm')


        var_name = 'tau'
        
        for animate_var in ['_p3_m15_8_ki1e4_tauo67_nx60_ny30_wpts50_G1c112']: 
            
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)    
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', '5%', '5%')
            tx = ax.set_title('Frame 0') # Gets overwritten later
            
            frames = [] # for storing the generated images
            for i in range(np.shape(locals()[var_name+animate_var])[2]):
                frames.append(locals()[var_name+animate_var][:,:,i])
                
            im = ax.pcolormesh(x_grid[:,40:], y_grid[:,40:], frames[0][:,40:])
            cbar = fig.colorbar(im, cax=cax, format="{x:.2f}")
            ax.set_xlim([np.min(x_grid[:,40:]), np.max(x_grid[:,40:])])
            ax.set_ylim([np.min(y_grid[:,40:]), np.max(y_grid[:,40:])])
            
            # plt.suptitle(animate_var, y=0.95)
                
            ani = animation.FuncAnimation(fig, animate_contourf_v2, frames=np.shape(locals()[var_name+animate_var])[2],
                                          interval = 1000, repeat_delay=1000)
            FFwriter = animation.FFMpegWriter(fps=2)
            ani.save(f'{animate_var}.mp4', writer=FFwriter)
            # ani.save(f'{animate_var}.gif', writer='imagemagick')
            
            
    # Creates line and scatter plots
    if False:
        for animate_var in ['_p3_m15_8_ki1e4_tauo67_nx60_ny30_wpts50_G1c112']: #["dmg_index"]:#, "del_d"]:
            
            scaling_fct = 23
            
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)    
            
            scatter_plot = ax.scatter(globals()[animate_var][0,0], scaling_fct*globals()[animate_var][0,1], 
                                      c="r", marker=MarkerStyle('x', transform=Affine2D().rotate_deg(45)), s=50)
            
            # Animate the line as well
            # line_plot = ax.plot(globals()[animate_var][0,0], scaling_fct*globals()[animate_var][0,1])[0]
            
            # Constant line plot
            ax.plot(globals()[animate_var][:,0], scaling_fct*globals()[animate_var][:,1])
            
            # plt.rcParams.update({'font.size': 14})
            ax.set(xlim=[0, 8], ylim=[0,200], xlabel='Displacement [mm]', ylabel='Force [N]')
            
                
            ani = animation.FuncAnimation(fig=fig, func=animate_plot_scatter, frames=np.shape(globals()[animate_var])[0],
                                          interval = 1000, repeat_delay=1000)
            # here frames are number of frames, not actual frames
            FFwriter = animation.FFMpegWriter(fps=2)
            ani.save(f'{animate_var}.mp4', writer=FFwriter)
            # ani.save(f'{animate_var}.gif', writer='imagemagick')
    
    # Creates both contourf and line/scatter plots
    if False:
        no_x_gauss = 60
        no_y_gauss = 30
        
        a = 52
        b = 25
        
        xis = np.zeros(no_x_gauss, dtype=np.float64)
        weights_xi = np.zeros(no_x_gauss, dtype=np.float64)
        etas = np.zeros(no_y_gauss, dtype=np.float64)
        weights_eta = np.zeros(no_y_gauss, dtype=np.float64)
        
        get_points_weights(no_x_gauss, xis, weights_xi)
        get_points_weights(no_y_gauss, etas, weights_eta)
        
        xi_grid, eta_grid = np.meshgrid(xis, etas)
        
        x_grid = a*(xi_grid+1)/2
        y_grid = b*(eta_grid+1)/2
        
        var_name = 'tau'
        for animate_var in ['_p3_m15_8_ki1e4_tauo67_nx60_ny30_wpts50_G1c112']:
            fig, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(10, 4))
            plt.subplots_adjust(bottom=0.05, left=0.05, right=0.95, top=0.95, wspace=0.5, hspace=0)
            
            # 1st plot with contourf plots
            div = make_axes_locatable(ax1)
            cax = div.append_axes('right', '5%', '5%')
            
            frames = [] # for storing the generated images
            for i in range(np.shape(globals()[var_name+animate_var])[2]):
                frames.append(globals()[var_name+animate_var][:,:,i])
                
            cv0 = frames[0]
            im = ax1.imshow(cv0) 
            cb = fig.colorbar(im, cax=cax)
            tx = ax1.set_title('Frame 0') # Gets overwritten later
            scaling_fct = 25
            
            # 2nd plot with the load displacement curve
            scatter_plot = ax2.scatter(globals()['force_intgn_dmg'+animate_var][0,0], 
                                      scaling_fct*globals()['force_intgn_dmg'+animate_var][0,1], 
                                      c="r", marker=MarkerStyle('x', transform=Affine2D().rotate_deg(45)), s=50)
            ax2.set(xlim=[0, 8], ylim=[0,200], xlabel='Displacement [mm]', ylabel='Force [N]')
            ax2.plot(globals()['force_intgn_dmg'+animate_var][:,0], scaling_fct*globals()['force_intgn_dmg'+animate_var][:,1])
            ax2.set_aspect('0.04')
            
            plt.suptitle(animate_var, y=0.95)

# np.shape(globals()['force_intgn_dmg'+animate_var])[0]
            ani = animation.FuncAnimation(fig=fig, func=animate_contourf_scatter_line, frames=np.shape(globals()['force_intgn_dmg'+animate_var])[0],
                                          interval = 1000, repeat_delay=1000)
            # here frames are number of frames, not actual frames
            FFwriter = animation.FFMpegWriter(fps=2)
            ani.save(f'{animate_var}_1.mp4', writer=FFwriter)
            
            
            
# %% Read data from excel

import pandas as pd

save_filename = 'FEM_Theo'
sheet_name = 'Theodores Test Case'
cols_to_use = 'J:K'

os.chdir('C:/Users/natha/OneDrive - Delft University of Technology/Fokker Internship/_Thesis/_Documentation/Results')
res = pd.read_excel('Load-Disp.xlsx', sheet_name=sheet_name, header=0, usecols=cols_to_use, skiprows=[1])
res = res.dropna()
print(res)
globals()[save_filename] = res.to_numpy()

FEM_foldername = 'Theodore'
os.chdir('C:/Users/natha/OneDrive - Delft University of Technology/Fokker Internship/_Thesis/_Results/Raw Results/DCB Damage/FEM/' + FEM_foldername)
np.save(save_filename, globals()[save_filename])

#%% Junkyard code
if False:
        # Importing stuff where everything changes
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
        

        


