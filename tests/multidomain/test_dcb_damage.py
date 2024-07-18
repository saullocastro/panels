local_run = True # True implies running on local device and not the cluster

import sys
import os
if local_run:
    sys.path.append('C:/Users/natha/Documents/GitHub/panels')
    sys.path.append('..\\..')
    os.chdir('C:/Users/natha/Documents/GitHub/panels/tests/multidomain')

import numpy as np
from structsolve import solve
from structsolve.sparseutils import finalize_symmetric_matrix, remove_null_cols
import time
import scipy

from panels import Shell
from panels.multidomain.connections import calc_ku_kv_kw_point_pd
from panels.multidomain.connections import fkCpd, fkCld_xcte, fkCld_ycte
from panels.plot_shell import plot_shell
from panels.multidomain import MultiDomain

# Open images
from matplotlib import image as img

from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

# To generate mp4's
import matplotlib
# matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\Users\natha\Downloads\ffmpeg-2024-04-01\ffmpeg\bin\ffmpeg.exe'

# Printing with reduced no of points (ease of viewing) - Suppress this to print in scientific notations and restart the kernel
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

from multiprocessing import Pool

import sys
import time

# To open pop up images - Ignore the syntax warning :)
# %matplotlib qt 
# For inline images
# %matplotlib inline

# Modified Newton-Raphson method
def scaling(vec, D):
    """
        A. Peano and R. Riccioni, Automated discretisatton error
        control in finite element analysis. In Finite Elements m
        the Commercial Enviror&ent (Editei by J. 26.  Robinson),
        pp. 368-387. Robinson & Assoc., Verwood.  England (1978)
    """
    non_nulls = ~np.isclose(D, 0)
    vec = vec[non_nulls]
    D = D[non_nulls]
    return np.sqrt((vec*np.abs(1/D))@vec)

# import os
# os.chdir('C:/Users/natha/Documents/GitHub/panels/tests/multidomain')

def img_popup(filename, plot_no = None, title = None):
    '''
    plot_no = current plot no 
    '''
    
    # To open pop up images - Ignore the syntax warning :)
    # %matplotlib qt 
    # For inline images
    # %matplotlib inline
     
    image = img.imread(filename)
    if plot_no is None:
        if title is None:
            plt.title(filename)
        else:
            plt.title(title)
        plt.imshow(image)
        plt.show()
    else:
        if title is None:
            plt.subplot(1,2,plot_no).set_title(filename)
        else:
            plt.subplot(1,2,plot_no).set_title(title)
        plt.imshow(image)
      
    

def test_dcb_damage_prop(no_pan, no_terms, filename='', k_i=None, tau_o=None, no_x_gauss=None, no_y_gauss=None, w_iter_no_pts=None, G1c=None):

    '''
        Damage propagation from a DCB with a precrack
        
        Code for 2 panels might not be right
            
        All units in MPa, N, mm
    '''    
    
    print(f'************************** BEGUN {filename} ******************************')
    # Start time
    start_time = time.time()
    
    # Properties
    E1 = (138300. + 128000.)/2. # MPa
    E2 = (10400. + 11500.)/2. # MPa
    G12 = 5190. # MPa
    nu12 = 0.316
    ply_thickness = 0.14 # mm

    # Plate dimensions (overall)
    a = 100 # mm
    b = 25  # mm
    # Dimensions of panel 1 and 2
    a1 = 52
    a2 = 0.015
    a3 = 10

    #others
    m_tsl = no_terms[0]
    n_tsl = no_terms[0]
    m = no_terms[1]
    n = no_terms[1]
    # print(f'no terms : {m}')

    plies = 15
    simple_layup = [0]*plies
    # simple_layup += simple_layup[::-1]
    # print('plies ',np.shape(simple_layup)[0])

    laminaprop = (E1, E2, nu12, G12, G12, G12)
    
    # no_pan = 3
    
    # Top DCB panels
    top1 = Shell(group='top', x0=0, y0=0, a=a1, b=b, m=m_tsl, n=n_tsl, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 2:
        top2 = Shell(group='top', x0=a1, y0=0, a=a-a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 3:
        top2 = Shell(group='top', x0=a1, y0=0, a=a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        top3 = Shell(group='top', x0=a1+a2, y0=0, a=a-a1-a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 4:
        top2 = Shell(group='top', x0=a1, y0=0, a=a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        top3 = Shell(group='top', x0=a1+a2, y0=0, a=a3, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        top4 = Shell(group='top', x0=a1+a2+a3, y0=0, a=a-(a1+a2+a3), b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)

    # Bottom DCB panels
    bot1 = Shell(group='bot', x0=0, y0=0, a=a1, b=b, m=m_tsl, n=n_tsl, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 2:
        bot2 = Shell(group='bot', x0=a1, y0=0, a=a-a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 3:
        bot2 = Shell(group='bot', x0=a1, y0=0, a=a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        bot3 = Shell(group='bot', x0=a1+a2, y0=0, a=a-a1-a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 4:
        bot2 = Shell(group='bot', x0=a1, y0=0, a=a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        bot3 = Shell(group='bot', x0=a1+a2, y0=0, a=a3, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        bot4 = Shell(group='bot', x0=a1+a2+a3, y0=0, a=a-(a1+a2+a3), b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)

    # boundary conditions
    
    BC = 'bot_end_fixed'
    # Possible strs: 'bot_fully_fixed', 'bot_end_fixed'
    
    clamped = False
    ss = True
    
    if clamped:
        bot_r = 0
        bot_t = 0
        top1_x1_wr = 1
    if ss:
        bot_r = 1
        bot_t = 0
        top1_x1_wr = 0
        
    # DCB with only the lower extreme end fixed at the tip. Rest free
    if BC == 'bot_end_fixed':
        top1.x1u = 1 ; top1.x1ur = 1 ; top1.x2u = 1 ; top1.x2ur = 1
        top1.x1v = 1 ; top1.x1vr = 1 ; top1.x2v = 1 ; top1.x2vr = 1 
        top1.x1w = 1 ; top1.x1wr = top1_x1_wr ; top1.x2w = 1 ; top1.x2wr = 1 
        top1.y1u = 1 ; top1.y1ur = 1 ; top1.y2u = 1 ; top1.y2ur = 1
        top1.y1v = 1 ; top1.y1vr = 1 ; top1.y2v = 1 ; top1.y2vr = 1
        top1.y1w = 1 ; top1.y1wr = 1 ; top1.y2w = 1 ; top1.y2wr = 1
        
        top2.x1u = 1 ; top2.x1ur = 1 ; top2.x2u = 1 ; top2.x2ur = 1
        top2.x1v = 1 ; top2.x1vr = 1 ; top2.x2v = 1 ; top2.x2vr = 1 
        top2.x1w = 1 ; top2.x1wr = 1 ; top2.x2w = 1 ; top2.x2wr = 1  
        top2.y1u = 1 ; top2.y1ur = 1 ; top2.y2u = 1 ; top2.y2ur = 1
        top2.y1v = 1 ; top2.y1vr = 1 ; top2.y2v = 1 ; top2.y2vr = 1
        top2.y1w = 1 ; top2.y1wr = 1 ; top2.y2w = 1 ; top2.y2wr = 1
        
        if no_pan == 3:
            top3.x1u = 1 ; top3.x1ur = 1 ; top3.x2u = 1 ; top3.x2ur = 1
            top3.x1v = 1 ; top3.x1vr = 1 ; top3.x2v = 1 ; top3.x2vr = 1 
            top3.x1w = 1 ; top3.x1wr = 1 ; top3.x2w = 1 ; top3.x2wr = 1  
            top3.y1u = 1 ; top3.y1ur = 1 ; top3.y2u = 1 ; top3.y2ur = 1
            top3.y1v = 1 ; top3.y1vr = 1 ; top3.y2v = 1 ; top3.y2vr = 1
            top3.y1w = 1 ; top3.y1wr = 1 ; top3.y2w = 1 ; top3.y2wr = 1
            
        if no_pan == 4:
            top3.x1u = 1 ; top3.x1ur = 1 ; top3.x2u = 1 ; top3.x2ur = 1
            top3.x1v = 1 ; top3.x1vr = 1 ; top3.x2v = 1 ; top3.x2vr = 1 
            top3.x1w = 1 ; top3.x1wr = 1 ; top3.x2w = 1 ; top3.x2wr = 1  
            top3.y1u = 1 ; top3.y1ur = 1 ; top3.y2u = 1 ; top3.y2ur = 1
            top3.y1v = 1 ; top3.y1vr = 1 ; top3.y2v = 1 ; top3.y2vr = 1
            top3.y1w = 1 ; top3.y1wr = 1 ; top3.y2w = 1 ; top3.y2wr = 1
            
            top4.x1u = 1 ; top4.x1ur = 1 ; top4.x2u = 1 ; top4.x2ur = 1
            top4.x1v = 1 ; top4.x1vr = 1 ; top4.x2v = 1 ; top4.x2vr = 1 
            top4.x1w = 1 ; top4.x1wr = 1 ; top4.x2w = 1 ; top4.x2wr = 1  
            top4.y1u = 1 ; top4.y1ur = 1 ; top4.y2u = 1 ; top4.y2ur = 1
            top4.y1v = 1 ; top4.y1vr = 1 ; top4.y2v = 1 ; top4.y2vr = 1
            top4.y1w = 1 ; top4.y1wr = 1 ; top4.y2w = 1 ; top4.y2wr = 1
    
        bot1.x1u = 1 ; bot1.x1ur = 1 ; bot1.x2u = 1 ; bot1.x2ur = 1
        bot1.x1v = 1 ; bot1.x1vr = 1 ; bot1.x2v = 1 ; bot1.x2vr = 1
        bot1.x1w = 1 ; bot1.x1wr = 1 ; bot1.x2w = 1 ; bot1.x2wr = 1
        bot1.y1u = 1 ; bot1.y1ur = 1 ; bot1.y2u = 1 ; bot1.y2ur = 1
        bot1.y1v = 1 ; bot1.y1vr = 1 ; bot1.y2v = 1 ; bot1.y2vr = 1
        bot1.y1w = 1 ; bot1.y1wr = 1 ; bot1.y2w = 1 ; bot1.y2wr = 1
        
        if no_pan == 2:
            bot2.x1u = 1 ; bot2.x1ur = 1 ; bot2.x2u = bot_t ; bot2.x2ur = 1 
            bot2.x1v = 1 ; bot2.x1vr = 1 ; bot2.x2v = bot_t ; bot2.x2vr = 1
            bot2.x1w = 1 ; bot2.x1wr = 1 ; bot2.x2w = bot_t ; bot2.x2wr = bot_r
            bot2.y1u = 1 ; bot2.y1ur = 1 ; bot2.y2u = 1 ; bot2.y2ur = 1
            bot2.y1v = 1 ; bot2.y1vr = 1 ; bot2.y2v = 1 ; bot2.y2vr = 1
            bot2.y1w = 1 ; bot2.y1wr = 1 ; bot2.y2w = 1 ; bot2.y2wr = 1
        
        if no_pan == 3:
            bot2.x1u = 1 ; bot2.x1ur = 1 ; bot2.x2u = 1 ; bot2.x2ur = 1
            bot2.x1v = 1 ; bot2.x1vr = 1 ; bot2.x2v = 1 ; bot2.x2vr = 1 
            bot2.x1w = 1 ; bot2.x1wr = 1 ; bot2.x2w = 1 ; bot2.x2wr = 1 
            bot2.y1u = 1 ; bot2.y1ur = 1 ; bot2.y2u = 1 ; bot2.y2ur = 1
            bot2.y1v = 1 ; bot2.y1vr = 1 ; bot2.y2v = 1 ; bot2.y2vr = 1
            bot2.y1w = 1 ; bot2.y1wr = 1 ; bot2.y2w = 1 ; bot2.y2wr = 1
            
            bot3.x1u = 1 ; bot3.x1ur = 1 ; bot3.x2u = bot_t ; bot3.x2ur = 1 
            bot3.x1v = 1 ; bot3.x1vr = 1 ; bot3.x2v = bot_t ; bot3.x2vr = 1 
            bot3.x1w = 1 ; bot3.x1wr = 1 ; bot3.x2w = bot_t ; bot3.x2wr = bot_r 
            bot3.y1u = 1 ; bot3.y1ur = 1 ; bot3.y2u = 1 ; bot3.y2ur = 1
            bot3.y1v = 1 ; bot3.y1vr = 1 ; bot3.y2v = 1 ; bot3.y2vr = 1
            bot3.y1w = 1 ; bot3.y1wr = 1 ; bot3.y2w = 1 ; bot3.y2wr = 1
            
        if no_pan == 4:
            bot2.x1u = 1 ; bot2.x1ur = 1 ; bot2.x2u = 1 ; bot2.x2ur = 1
            bot2.x1v = 1 ; bot2.x1vr = 1 ; bot2.x2v = 1 ; bot2.x2vr = 1 
            bot2.x1w = 1 ; bot2.x1wr = 1 ; bot2.x2w = 1 ; bot2.x2wr = 1 
            bot2.y1u = 1 ; bot2.y1ur = 1 ; bot2.y2u = 1 ; bot2.y2ur = 1
            bot2.y1v = 1 ; bot2.y1vr = 1 ; bot2.y2v = 1 ; bot2.y2vr = 1
            bot2.y1w = 1 ; bot2.y1wr = 1 ; bot2.y2w = 1 ; bot2.y2wr = 1
            
            bot3.x1u = 1 ; bot3.x1ur = 1 ; bot3.x2u = 1 ; bot3.x2ur = 1 
            bot3.x1v = 1 ; bot3.x1vr = 1 ; bot3.x2v = 1 ; bot3.x2vr = 1 
            bot3.x1w = 1 ; bot3.x1wr = 1 ; bot3.x2w = 1 ; bot3.x2wr = 1 
            bot3.y1u = 1 ; bot3.y1ur = 1 ; bot3.y2u = 1 ; bot3.y2ur = 1
            bot3.y1v = 1 ; bot3.y1vr = 1 ; bot3.y2v = 1 ; bot3.y2vr = 1
            bot3.y1w = 1 ; bot3.y1wr = 1 ; bot3.y2w = 1 ; bot3.y2wr = 1

            bot4.x1u = 1 ; bot4.x1ur = 1 ; bot4.x2u = bot_t ; bot4.x2ur = 1 
            bot4.x1v = 1 ; bot4.x1vr = 1 ; bot4.x2v = bot_t ; bot4.x2vr = 1 
            bot4.x1w = 1 ; bot4.x1wr = 1 ; bot4.x2w = bot_t ; bot4.x2wr = bot_r 
            bot4.y1u = 1 ; bot4.y1ur = 1 ; bot4.y2u = 1 ; bot4.y2ur = 1
            bot4.y1v = 1 ; bot4.y1vr = 1 ; bot4.y2v = 1 ; bot4.y2vr = 1
            bot4.y1w = 1 ; bot4.y1wr = 1 ; bot4.y2w = 1 ; bot4.y2wr = 1
            
    
    if no_x_gauss is None:
        no_x_gauss = 60
    if no_y_gauss is None:
        no_y_gauss = 30
    
    # All connections - list of dict
    if False: # incomplete
        if no_pan == 2:
            conn = [
             # skin-skin
             dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
             dict(p1=bot1, p2=bot2, func='SSxcte', xcte1=bot1.a, xcte2=0),
             dict(p1=top1, p2=bot1, func='SB'), #'_TSL', tsl_type = 'linear'), 
                # dict(p1=top2, p2=bot2, func='SB_TSL', tsl_type = 'linear')
            ]
    if no_pan == 3:
        conn = [
         # skin-skin
           dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
           dict(p1=top2, p2=top3, func='SSxcte', xcte1=top2.a, xcte2=0),
           dict(p1=bot1, p2=bot2, func='SSxcte', xcte1=bot1.a, xcte2=0),
           dict(p1=bot2, p2=bot3, func='SSxcte', xcte1=bot2.a, xcte2=0),
           dict(p1=top1, p2=bot1, func='SB_TSL', tsl_type = 'bilinear', no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, k_i=k_i, tau_o=tau_o, G1c=G1c)
        ]
    if no_pan == 4:
        conn = [
         # skin-skin
           dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
           dict(p1=top2, p2=top3, func='SSxcte', xcte1=top2.a, xcte2=0),
           dict(p1=top3, p2=top4, func='SSxcte', xcte1=top3.a, xcte2=0),
           dict(p1=bot1, p2=bot2, func='SSxcte', xcte1=bot1.a, xcte2=0),
           dict(p1=bot2, p2=bot3, func='SSxcte', xcte1=bot2.a, xcte2=0),
           dict(p1=bot3, p2=bot4, func='SSxcte', xcte1=bot3.a, xcte2=0),
           dict(p1=top1, p2=bot1, func='SB_TSL', tsl_type = 'bilinear', no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, k_i=k_i, tau_o=tau_o, G1c=G1c)
        ]
    
    # This determines the positions of each panel's (sub)matrix in the global matrix when made a MD obj below
    # So changing this changes the placements i.e. starting row and col of each
    if no_pan == 2:
        panels = [bot1, bot2, top1, top2]
    if no_pan == 3:
        panels = [bot1, bot2, bot3, top1, top2, top3]
    if no_pan == 4:
        panels = [bot1, bot2, bot3, bot4, top1, top2, top3, top4]

    assy = MultiDomain(panels=panels, conn=conn) # assy is now an object of the MultiDomain class
    # Here the panels (shell objs) are modified -- their starting positions in the global matrix is assigned etc

    # Panel at which the disp is applied
    if no_pan == 2:
        disp_panel = top2
    if no_pan == 3:
        disp_panel = top3
    if no_pan == 4:
        disp_panel = top4
    



    if True:
        ######## THIS SHOULD BE CHANGED LATER PER DISP TYPE ###########################################
        ku, kv, kw = calc_ku_kv_kw_point_pd(disp_panel)
        
    size = assy.get_size()
        
    # To match the inital increment 
    wp = 0.01   
    
    # --------- IMPROVE THE STARTING GUESS --------------
    if False:
        # Prescribed Displacements
        if True:
            disp_type = 'point' # change based on what's being applied
            
            if disp_type == 'point':
                # Penalty Stiffness
                # Disp in z, so only kw is non zero. ku and kv are zero
                kCp = fkCpd(0., 0., kw, disp_panel, disp_panel.a, disp_panel.b/2, size, disp_panel.row_start, disp_panel.col_start)
                # Point load (added to shell obj)
                disp_panel.add_point_pd(disp_panel.a, disp_panel.b/2, 0., 0., 0., 0., kw, wp)
            if disp_type == 'line_xcte':
                kCp = fkCld_xcte(0., 0., kw, disp_panel, disp_panel.a, size, disp_panel.row_start, disp_panel.col_start)
                disp_panel.add_distr_pd_fixed_x(disp_panel.a, None, None, kw,
                                          funcu=None, funcv=None, funcw = lambda y: wp) #*y/top2.b, cte=True)
            if disp_type == 'line_ycte':
                kCp = fkCld_ycte(0., 0., kw, disp_panel, disp_panel.b, size, disp_panel.row_start, disp_panel.col_start)
                disp_panel.add_distr_pd_fixed_y(disp_panel.b, None, None, kw,
                                          funcu=None, funcv=None, funcw = lambda x: wp) #*x/top2.a, cte=True)
    
        # Stiffness matrix from penalties due to prescribed displacement
        kCp = finalize_symmetric_matrix(kCp)
        
        # Has to be after the forces/loads are added to the panels
        fext = assy.calc_fext()
        
        k0 = kT + kCp
        ci = solve(k0, fext)
        size = k0.shape[0]
    else:
        ci = np.zeros(size) # Only used to calc initial fint
        
        
    # -------------------- INCREMENTATION --------------------------
    # wp_max = 10 # [mm]
    # no_iter_disp = 100
    
    disp_iter_no = 0
    
    # Finding info of the connection
    for conn_list in conn:
        if conn_list['func'] == 'SB_TSL':
            no_x_gauss = conn_list['no_x_gauss']
            no_y_gauss = conn_list['no_y_gauss']
            tsl_type = conn_list['tsl_type']
            p_top = conn_list['p1']
            p_bot = conn_list['p2']
            break # Assuming there is only 1 connection
        
    # Initilaize mat to store results
    if w_iter_no_pts is None:
        w_iter_no_pts = 150
        
    # w_iter = np.unique(np.concatenate((np.linspace(0.01,3,int(0.2*w_iter_no_pts)), np.linspace(3,6,int(0.4*w_iter_no_pts)), 
    #                                    np.linspace(6,8,int(0.4*w_iter_no_pts)))))
    # w_iter = np.linspace(0.01,8,100)
    # w_iter = [0.01, 2]
    # Reverse loading 
    # w_iter = np.concatenate((np.linspace(0.01,3,int(0.1*w_iter_no_pts)), np.linspace(3,6,int(0.25*w_iter_no_pts)), 
    #                                    np.linspace(6,7,int(0.25*w_iter_no_pts)),
    #                                    np.linspace(7,0,int(0.1*w_iter_no_pts)), np.linspace(0,7,int(0.1*w_iter_no_pts)), 
    #                                    np.linspace(7,8,int(0.2*w_iter_no_pts)) ))
    w_iter = np.array([0.01, 1.01, 2.00, 3.00, 3.33, 3.67, 4.00, 4.33, 4.67, 5.00, 5.33,
           5.67, 6.00, 6.11, 6.22, 6.33, 6.44, 6.55, 6.67, 6.77, 6.89, 6.99, 7.00, 4.67, 2.50, 0.01, 2.50,
           4.67, 7.00, 7.14, 7.25, 7.43, 7.57, 7.7, 7.86, 7.92, 8.00])
    
    dmg_index = np.zeros((no_y_gauss,no_x_gauss,np.shape(w_iter)[0]))
    del_d = np.zeros((no_y_gauss,no_x_gauss,np.shape(w_iter)[0]))
    kw_tsl = np.zeros((no_y_gauss,no_x_gauss,np.shape(w_iter)[0]))
    force_intgn = np.zeros((np.shape(w_iter)[0], 2))
    force_intgn_dmg = np.zeros((np.shape(w_iter)[0], 2))
    displ_top_root = np.zeros((50,200,np.shape(w_iter)[0]))
    displ_bot_root = np.zeros((50,200,np.shape(w_iter)[0]))
    c_all = np.zeros((size, np.shape(w_iter)[0]))
    
    
    # Displacement Incrementation
    for wp in w_iter: 
        # print(f'------------ wp = {wp:.3f} ---------------')
        
        # Prescribed Displacements
        if True:
            disp_type = 'line_xcte' # change based on what's being applied
            
            # Clears all previously added displs - otherwise in NL case, theyre readded so you have 2 disps at the tip
            disp_panel.clear_disps()
            
            if disp_type == 'point':
                # Penalty Stiffness
                # Disp in z, so only kw is non zero. ku and kv are zero
                kCp = fkCpd(0., 0., kw, disp_panel, disp_panel.a, disp_panel.b/2, size, disp_panel.row_start, disp_panel.col_start)
                # Point load (added to shell obj)
                disp_panel.add_point_pd(disp_panel.a, disp_panel.b/2, 0., 0., 0., 0., kw, wp)
            if disp_type == 'line_xcte':
                kCp = fkCld_xcte(0., 0., kw, disp_panel, disp_panel.a, size, disp_panel.row_start, disp_panel.col_start)
                disp_panel.add_distr_pd_fixed_x(disp_panel.a, None, None, kw,
                                           funcu=None, funcv=None, funcw = lambda y: wp) #*y/top2.b, cte=True)
            if disp_type == 'line_ycte':
                kCp = fkCld_ycte(0., 0., kw, disp_panel, disp_panel.b, size, disp_panel.row_start, disp_panel.col_start)
                disp_panel.add_distr_pd_fixed_y(disp_panel.b, None, None, kw,
                                           funcu=None, funcv=None, funcw = lambda x: wp) #*x/top2.a, cte=True)

        # Stiffness matrix from penalties due to prescribed displacement
        kCp = finalize_symmetric_matrix(kCp)
        
        # Has to be after the forces/loads are added to the panels
        fext = assy.calc_fext()

        # Inital guess ci and increment dc (same size as fext)
        dc = np.zeros_like(fext)
        
        
        # All subsequent load steps
        if disp_iter_no != 0:
            # Doing this to avoid recalc kc_conn and help pass the correct k_i
            kC_conn = assy.get_kC_conn(c=c)
            # Inital fint (0 bec c=0)
            fint = np.asarray(assy.calc_fint(c=c, kC_conn=kC_conn))
            # Residual with disp terms
            Ri = fint - fext + kCp*c
            kT = assy.calc_kT(c=c, kC_conn=kC_conn) 
        # Initial step where ci = zeros
        if disp_iter_no == 0 and np.max(ci) == 0: 
            # max(ci) bec if its already calc for an initial guess above, no need to repeat it - 
            #                   only when c is zero (or randomly initialized - modify 'if' then)
            
            # Doing this to avoid recalc kc_conn and help pass the correct k_i
            kC_conn = assy.get_kC_conn(c=ci)
            # Inital fint (0 bec c=0)
            fint = np.asarray(assy.calc_fint(c=ci, kC_conn=kC_conn))
            # Residual with disp terms
            Ri = fint - fext + kCp*ci
            kT = assy.calc_kT(c=ci, kC_conn=kC_conn)
        k0 = kT + kCp
        
        # print(f'kT {np.max(kT):.3e} kcp {np.max(kCp):.3e} k0 {np.max(k0):.3e}')
        
        epsilon = 5.e-6 # Convergence criteria
        D = k0.diagonal() # For convergence - calc at beginning of load increment
        
        count = 0 # Tracks number of NR iterations 

        # Modified Newton Raphson Iteration
        while True:
            # print()
            # print(f"------------ NR start {count+1}--------------")
            dc = solve(k0, -Ri, silent=True)
            c = ci + dc
            
            kC_conn = assy.get_kC_conn(c=c)
            fint = np.asarray(assy.calc_fint(c=c, kC_conn=kC_conn))
            # Might need to calc fext here again if it changes per iteration when fext changes when not using kc_conn for SB
            Ri = fint - fext + kCp*c
            # print(f'Ri {np.linalg.norm(Ri)}')
            
            # Extracting data out of the MD class for plotting etc 
                # Damage considerations are already implemented in the Multidomain class functions kT, fint - no need to include anything externally

            # kw_tsl_iter, dmg_index_iter, del_d_iter = assy.calc_k_dmg(c=c, pA=p_top, pB=p_bot, no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, tsl_type=tsl_type)
            # print(f"             NR end {count} -- wp={wp:.3f}  max dmg {np.max(dmg_index_iter):.3f}  ---") # min del_d {np.min(del_d_iter):.2e}---------")
            # print(f'    del_d min {np.min(del_d_iter)}  -- max {np.max(del_d_iter):.4f}')
            
            crisfield_test = scaling(Ri, D)/max(scaling(fext, D), scaling(fint + kCp*c, D))
            # print(f'    crisfield {crisfield_test:.4f}')
            if crisfield_test < epsilon:
                break
            
            count += 1
            kT = assy.calc_kT(c=c, kC_conn=kC_conn) 
            k0 = kT + kCp
            ci = c.copy()
            
            # ------------------ SAVING VARIABLES --------------------
            if True:
                np.save(f'dmg_index_{filename}', dmg_index)
                np.save(f'del_d_{filename}', del_d)
                np.save(f'kw_tsl_{filename}', kw_tsl)
                np.save(f'force_intgn_{filename}', force_intgn)
                np.save(f'force_intgn_dmg_{filename}', force_intgn_dmg)
                # np.save(f'displ_top_root_{filename}', displ_top_root)
                # np.save(f'displ_bot_root_{filename}', displ_bot_root)
                np.save(f'c_all_{filename}', c_all)
            
            # Saving results incase cluster time is exceeded
            if (time.time() - start_time)/(60*60) > 70: # Max wall time (hrs) on the cluster
                np.save(f'dmg_index_{filename}', dmg_index)
                np.save(f'del_d_{filename}', del_d)
                np.save(f'kw_tsl_{filename}', kw_tsl)
                np.save(f'force_intgn_{filename}', force_intgn)
                np.save(f'force_intgn_dmg_{filename}', force_intgn_dmg)
                # np.save(f'displ_top_root_{filename}', displ_top_root)
                # np.save(f'displ_bot_root_{filename}', displ_bot_root)
                np.save(f'c_all_{filename}', c_all)
            
            if count > 1000:
                print('Unconverged Results !!!!!!!!!!!!!!!!!!!')
                # return None, dmg_index, del_d, kw_tsl
                if True:
                    np.save(f'dmg_index_{filename}', dmg_index)
                    np.save(f'del_d_{filename}', del_d)
                    np.save(f'kw_tsl_{filename}', kw_tsl)
                    np.save(f'force_intgn_{filename}', force_intgn)
                    np.save(f'force_intgn_dmg_{filename}', force_intgn_dmg)
                    # np.save(f'displ_top_root_{filename}', displ_top_root)
                    # np.save(f'displ_bot_root_{filename}', displ_bot_root)
                    np.save(f'c_all_{filename}', c_all)
                raise RuntimeError('NR didnt converged :(') 
        
        # Edit: should be correct as it is 
                # Ques: should first update max del d then calc kw_tsl etc ??
        if hasattr(assy, "del_d"):
            kw_tsl[:,:,disp_iter_no], dmg_index[:,:,disp_iter_no], del_d[:,:,disp_iter_no]  = assy.calc_k_dmg(c=c, pA=p_top, pB=p_bot, 
                                                 no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, tsl_type=tsl_type, 
                                                 prev_max_del_d=assy.del_d, k_i=k_i, tau_o=tau_o)
        else:
            kw_tsl[:,:,disp_iter_no], dmg_index[:,:,disp_iter_no], del_d[:,:,disp_iter_no]  = assy.calc_k_dmg(c=c, pA=p_top, pB=p_bot, 
                                                 no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, tsl_type=tsl_type,
                                                 prev_max_del_d=None, k_i=k_i, tau_o=tau_o)
            
        c_all[:,disp_iter_no] = c
            
        # Update max del_d AFTER a converged NR iteration
        assy.update_max_del_d(curr_max_del_d=del_d[:,:,disp_iter_no])
        
        # kw_tsl[:,:,disp_iter_no], dmg_index[:,:,disp_iter_no], del_d[:,:,disp_iter_no] = assy.calc_k_dmg(c=c, pA=p_top, pB=p_bot, no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, tsl_type=tsl_type)
        print(f'            {filename} -- wp={wp:.3f} {disp_iter_no/np.shape(w_iter)[0]*100:.1f}% -- max DI {np.max(dmg_index[:,:,disp_iter_no]):.4f}')
        sys.stdout.flush()
        # print(f'        max del_d {np.max(del_d[:,:,disp_iter_no])}')
        # print(f'       min del_d {np.min(del_d[:,:,disp_iter_no])}')
        # print(f'        max kw_tsl {np.max(kw_tsl[:,:,disp_iter_no])}')
        
        
        # Force - TEMP - CHECK LATER AND EDIT/REMOVE
        if True:
            force_intgn[disp_iter_no, 0] = wp
            force_intgn[disp_iter_no, 1] = assy.force_out_plane(c, group=None, eval_panel=disp_panel, x_cte_force=None, y_cte_force=None,
                      gridx=100, gridy=50, NLterms=True, no_x_gauss=128, no_y_gauss=128)
        else: 
            force_intgn = None
        
        # Force by area integral of traction in the damaged region
        if True:
            force_intgn_dmg[disp_iter_no, 0] = wp
            force_intgn_dmg[disp_iter_no, 1] = assy.force_out_plane_damage(conn=conn, c=c)
        
        # Calc displ of top and bottom panels at each increment
        if True:
            res_pan_top = assy.calc_results(c=c, eval_panel=top1, vec='w', 
                                    no_x_gauss=200, no_y_gauss=50)
            res_pan_bot = assy.calc_results(c=c, eval_panel=bot1, vec='w', 
                                    no_x_gauss=200, no_y_gauss=50)
            displ_top_root[:,:,disp_iter_no] = res_pan_top['w'][0]
            displ_bot_root[:,:,disp_iter_no] = res_pan_bot['w'][0]
        else:
            res_pan_top = None
            res_pan_bot = None
        
        disp_iter_no += 1
        print()
        
        sys.stdout.flush()
        
        if np.all(dmg_index == 1):
            print('Cohesive Zone has failed')
            break
        
        
        
    # ------------------ SAVING VARIABLES --------------------
    if True:
        np.save(f'dmg_index_{filename}', dmg_index)
        np.save(f'del_d_{filename}', del_d)
        np.save(f'kw_tsl_{filename}', kw_tsl)
        np.save(f'force_intgn_{filename}', force_intgn)
        np.save(f'force_intgn_dmg_{filename}', force_intgn_dmg)
        # np.save(f'displ_top_root_{filename}', displ_top_root)
        # np.save(f'displ_bot_root_{filename}', displ_bot_root)
        np.save(f'c_all_{filename}', c_all)
    
    print(f'************************** COMPLETED {filename} ******************************')
    sys.stdout.flush()
    
    # ------------------ RESULTS AND POST PROCESSING --------------------
    
    c0 = c.copy()

    generate_plots = False
    
    final_res = None
    # Plotting results
    if False:
        for vec in ['w']:#, 'Mxx']:#, 'Myy', 'Mxy']:#, 'Nxx', 'Nyy']:
            res_bot = assy.calc_results(c=c0, group='bot', vec=vec, no_x_gauss=None, no_y_gauss=None)
            res_top = assy.calc_results(c=c0, group='top', vec=vec, no_x_gauss=None, no_y_gauss=None)
            vecmin = min(np.min(np.array(res_top[vec])), np.min(np.array(res_bot[vec])))
            vecmax = max(np.max(np.array(res_top[vec])), np.max(np.array(res_bot[vec])))
            if vec != 'w':
                print(f'{vec} :: {vecmin:.3f}  {vecmax:.3f}')
            # if vec == 'w':
            if True:
                # Printing max min per panel
                if False:
                    for pan in range(0,np.shape(res_bot[vec])[0]):
                        print(f'{vec} top{pan+1} :: {np.min(np.array(res_top[vec][pan])):.3f}  {np.max(np.array(res_top[vec][pan])):.3f}')
                    print('------------------------------')
                    for pan in range(0,np.shape(res_bot[vec])[0]): 
                        print(f'{vec} bot{pan+1} :: {np.min(np.array(res_bot[vec][pan])):.3f}  {np.max(np.array(res_bot[vec][pan])):.3f}')
                    print('------------------------------')
                print(f'Global TOP {vec} :: {np.min(np.array(res_top[vec])):.3f}  {np.max(np.array(res_top[vec])):.3f}')
                print(f'Global BOT {vec} :: {np.min(np.array(res_bot[vec])):.3f}  {np.max(np.array(res_bot[vec])):.3f}')
                # print(res_bot[vec][1][:,-1]) # disp at the tip
                final_res = np.min(np.array(res_top[vec]))
            
            if generate_plots:
                # if vec == 'w':
                if True:
                    assy.plot(c=c0, group='bot', vec=vec, filename='test_dcb_before_opening_bot_tsl.png', show_boundaries=True,
                                                colorbar=True, res = res_bot, vecmax=vecmax, vecmin=vecmin, display_zero=True)
                    
                    assy.plot(c=c0, group='top', vec=vec, filename='test_dcb_before_opening_top_tsl.png', show_boundaries=True,
                                              colorbar=True, res = res_top, vecmax=vecmax, vecmin=vecmin, display_zero=True)
            
            # Open images
            if generate_plots:
                img_popup('test_dcb_before_opening_top_tsl.png',1, f"{vec} top")
                img_popup('test_dcb_before_opening_bot_tsl.png',2, f"{vec} bot")
                plt.show()
    
    animate = False
    if animate:    
        def animate(i):
            curr_res = frames[i]
            max_res = np.max(curr_res)
            min_res = np.min(curr_res)
            if animate_var == 'dmg_index':
                if min_res == 0:
                    vmin = 0.0
                    vmax = 1.0
                else: 
                    possible_min_cbar = [0,0.5,0.85,0.9,0.95,0.99]
                    vmin = max(list(filter(lambda x: x<min_res, possible_min_cbar)))
                    vmax = 1.0
            else:
                vmin = min_res
                vmax = max_res
            im = ax.imshow(curr_res)
            fig.colorbar(im, cax=cax)
            im.set_data(curr_res)
            im.set_clim(vmin, vmax)
            tx.set_text(f'{animate_var}     -   Disp={w_iter[i]:.2f} mm')
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        for animate_var in ["dmg_index", "del_d"]:#, 'kw_tsl']:
            
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
                                          interval = 200, repeat_delay=1000)
            FFwriter = animation.FFMpegWriter(fps=5)
            ani.save(f'{animate_var}.mp4', writer=FFwriter)
            # ani.save(f'{animate_var}.gif', writer='imagemagick')
        
    return dmg_index, del_d, kw_tsl, force_intgn, displ_top_root, displ_bot_root, c_all




def test_dcb_damage_prop_fcrack(phy_dim, no_terms, filename='', k_i=None, tau_o=None, no_x_gauss=None, no_y_gauss=None, w_iter_no_pts=None, G1c=None):

    '''
        Damage propagation from a DCB with a precrack
        
        Code for 2 panels might not be right
            
        All units in MPa, N, mm
    '''    
    
    foldername = 'p3_25_m15_8_kiITER_tauo77_nx60_ny30_wpts50_G1c112'
    generation_code = 'for ki in [1e4, 5e4, 1e5, 5e5, 1e6]]'
    
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    os.chdir(foldername)
                    
    # Delete previous log file - later when appending the file, if the file doesnt exist, it is created
    if os.path.isfile(f'./{filename}.txt'):
        os.remove(f'{filename}.txt')
    
    # Log to check on multiple runs at once 
    if not local_run:
        log_file = open(f'{filename}.txt', 'a')
        log_file.write(f'************************** IT HAS BEGUN - {filename} ******************************\n')
        log_file.write(f'generation code:   {generation_code} \n')
        log_file.close()
    
    print(f'************************** IT HAS BEGUN - {filename} ******************************')
    sys.stdout.flush()
    # Start time
    start_time = time.time()
    
    no_pan = phy_dim[0]
    b = phy_dim[1]
    
    # Properties
    E1 = (138300. + 128000.)/2. # MPa
    E2 = (10400. + 11500.)/2. # MPa
    G12 = 5190. # MPa
    nu12 = 0.316
    ply_thickness = 0.14 # mm

    # Plate dimensions (overall)
    a = 100 # mm
    # b = 25  # mm
    # Dimensions of panel 1 and 2
    a1 = 52
    a2 = 0.015
    a3 = 10

    #others
    m_tsl = no_terms[0]
    n_tsl = no_terms[0]
    m = no_terms[1]
    n = no_terms[1]
    # print(f'no terms : {m}')

    plies = 15
    simple_layup = [0]*plies
    # simple_layup += simple_layup[::-1]
    # print('plies ',np.shape(simple_layup)[0])

    laminaprop = (E1, E2, nu12, G12, G12, G12)
    
    # no_pan = 3
    
    # Top DCB panels
    top1 = Shell(group='top', x0=0, y0=0, a=a1, b=b, m=m_tsl, n=n_tsl, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 2:
        top2 = Shell(group='top', x0=a1, y0=0, a=a-a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 3:
        top2 = Shell(group='top', x0=a1, y0=0, a=a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        top3 = Shell(group='top', x0=a1+a2, y0=0, a=a-a1-a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 4:
        top2 = Shell(group='top', x0=a1, y0=0, a=a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        top3 = Shell(group='top', x0=a1+a2, y0=0, a=a3, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        top4 = Shell(group='top', x0=a1+a2+a3, y0=0, a=a-(a1+a2+a3), b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)

    # Bottom DCB panels
    bot1 = Shell(group='bot', x0=0, y0=0, a=a1, b=b, m=m_tsl, n=n_tsl, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 2:
        bot2 = Shell(group='bot', x0=a1, y0=0, a=a-a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 3:
        bot2 = Shell(group='bot', x0=a1, y0=0, a=a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        bot3 = Shell(group='bot', x0=a1+a2, y0=0, a=a-a1-a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 4:
        bot2 = Shell(group='bot', x0=a1, y0=0, a=a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        bot3 = Shell(group='bot', x0=a1+a2, y0=0, a=a3, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        bot4 = Shell(group='bot', x0=a1+a2+a3, y0=0, a=a-(a1+a2+a3), b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)


    # boundary conditions
    
    BC = 'bot_end_fixed'
    # Possible strs: 'bot_fully_fixed', 'bot_end_fixed'
    
    clamped = False
    ss = True
    
    if clamped:
        bot_r = 0
        bot_t = 0
        top1_x1_wr = 1
    if ss:
        bot_r = 1
        bot_t = 0
        top1_x1_wr = 0
        
    # DCB with only the lower extreme end fixed at the tip. Rest free
    if BC == 'bot_end_fixed':
        top1.x1u = 1 ; top1.x1ur = 1 ; top1.x2u = 1 ; top1.x2ur = 1
        top1.x1v = 1 ; top1.x1vr = 1 ; top1.x2v = 1 ; top1.x2vr = 1 
        top1.x1w = 1 ; top1.x1wr = top1_x1_wr ; top1.x2w = 1 ; top1.x2wr = 1 
        top1.y1u = 1 ; top1.y1ur = 1 ; top1.y2u = 1 ; top1.y2ur = 1
        top1.y1v = 1 ; top1.y1vr = 1 ; top1.y2v = 1 ; top1.y2vr = 1
        top1.y1w = 1 ; top1.y1wr = 1 ; top1.y2w = 1 ; top1.y2wr = 1
        
        top2.x1u = 1 ; top2.x1ur = 1 ; top2.x2u = 1 ; top2.x2ur = 1
        top2.x1v = 1 ; top2.x1vr = 1 ; top2.x2v = 1 ; top2.x2vr = 1 
        top2.x1w = 1 ; top2.x1wr = 1 ; top2.x2w = 1 ; top2.x2wr = 1  
        top2.y1u = 1 ; top2.y1ur = 1 ; top2.y2u = 1 ; top2.y2ur = 1
        top2.y1v = 1 ; top2.y1vr = 1 ; top2.y2v = 1 ; top2.y2vr = 1
        top2.y1w = 1 ; top2.y1wr = 1 ; top2.y2w = 1 ; top2.y2wr = 1
        
        if no_pan == 3:
            top3.x1u = 1 ; top3.x1ur = 1 ; top3.x2u = 1 ; top3.x2ur = 1
            top3.x1v = 1 ; top3.x1vr = 1 ; top3.x2v = 1 ; top3.x2vr = 1 
            top3.x1w = 1 ; top3.x1wr = 1 ; top3.x2w = 1 ; top3.x2wr = 1  
            top3.y1u = 1 ; top3.y1ur = 1 ; top3.y2u = 1 ; top3.y2ur = 1
            top3.y1v = 1 ; top3.y1vr = 1 ; top3.y2v = 1 ; top3.y2vr = 1
            top3.y1w = 1 ; top3.y1wr = 1 ; top3.y2w = 1 ; top3.y2wr = 1
            
        if no_pan == 4:
            top3.x1u = 1 ; top3.x1ur = 1 ; top3.x2u = 1 ; top3.x2ur = 1
            top3.x1v = 1 ; top3.x1vr = 1 ; top3.x2v = 1 ; top3.x2vr = 1 
            top3.x1w = 1 ; top3.x1wr = 1 ; top3.x2w = 1 ; top3.x2wr = 1  
            top3.y1u = 1 ; top3.y1ur = 1 ; top3.y2u = 1 ; top3.y2ur = 1
            top3.y1v = 1 ; top3.y1vr = 1 ; top3.y2v = 1 ; top3.y2vr = 1
            top3.y1w = 1 ; top3.y1wr = 1 ; top3.y2w = 1 ; top3.y2wr = 1
            
            top4.x1u = 1 ; top4.x1ur = 1 ; top4.x2u = 1 ; top4.x2ur = 1
            top4.x1v = 1 ; top4.x1vr = 1 ; top4.x2v = 1 ; top4.x2vr = 1 
            top4.x1w = 1 ; top4.x1wr = 1 ; top4.x2w = 1 ; top4.x2wr = 1  
            top4.y1u = 1 ; top4.y1ur = 1 ; top4.y2u = 1 ; top4.y2ur = 1
            top4.y1v = 1 ; top4.y1vr = 1 ; top4.y2v = 1 ; top4.y2vr = 1
            top4.y1w = 1 ; top4.y1wr = 1 ; top4.y2w = 1 ; top4.y2wr = 1
    
        bot1.x1u = 1 ; bot1.x1ur = 1 ; bot1.x2u = 1 ; bot1.x2ur = 1
        bot1.x1v = 1 ; bot1.x1vr = 1 ; bot1.x2v = 1 ; bot1.x2vr = 1
        bot1.x1w = 1 ; bot1.x1wr = 1 ; bot1.x2w = 1 ; bot1.x2wr = 1
        bot1.y1u = 1 ; bot1.y1ur = 1 ; bot1.y2u = 1 ; bot1.y2ur = 1
        bot1.y1v = 1 ; bot1.y1vr = 1 ; bot1.y2v = 1 ; bot1.y2vr = 1
        bot1.y1w = 1 ; bot1.y1wr = 1 ; bot1.y2w = 1 ; bot1.y2wr = 1
        
        if no_pan == 2:
            bot2.x1u = 1 ; bot2.x1ur = 1 ; bot2.x2u = bot_t ; bot2.x2ur = 1 
            bot2.x1v = 1 ; bot2.x1vr = 1 ; bot2.x2v = bot_t ; bot2.x2vr = 1
            bot2.x1w = 1 ; bot2.x1wr = 1 ; bot2.x2w = bot_t ; bot2.x2wr = bot_r
            bot2.y1u = 1 ; bot2.y1ur = 1 ; bot2.y2u = 1 ; bot2.y2ur = 1
            bot2.y1v = 1 ; bot2.y1vr = 1 ; bot2.y2v = 1 ; bot2.y2vr = 1
            bot2.y1w = 1 ; bot2.y1wr = 1 ; bot2.y2w = 1 ; bot2.y2wr = 1
        
        if no_pan == 3:
            bot2.x1u = 1 ; bot2.x1ur = 1 ; bot2.x2u = 1 ; bot2.x2ur = 1
            bot2.x1v = 1 ; bot2.x1vr = 1 ; bot2.x2v = 1 ; bot2.x2vr = 1 
            bot2.x1w = 1 ; bot2.x1wr = 1 ; bot2.x2w = 1 ; bot2.x2wr = 1 
            bot2.y1u = 1 ; bot2.y1ur = 1 ; bot2.y2u = 1 ; bot2.y2ur = 1
            bot2.y1v = 1 ; bot2.y1vr = 1 ; bot2.y2v = 1 ; bot2.y2vr = 1
            bot2.y1w = 1 ; bot2.y1wr = 1 ; bot2.y2w = 1 ; bot2.y2wr = 1
            
            bot3.x1u = 1 ; bot3.x1ur = 1 ; bot3.x2u = bot_t ; bot3.x2ur = 1 
            bot3.x1v = 1 ; bot3.x1vr = 1 ; bot3.x2v = bot_t ; bot3.x2vr = 1 
            bot3.x1w = 1 ; bot3.x1wr = 1 ; bot3.x2w = bot_t ; bot3.x2wr = bot_r 
            bot3.y1u = 1 ; bot3.y1ur = 1 ; bot3.y2u = 1 ; bot3.y2ur = 1
            bot3.y1v = 1 ; bot3.y1vr = 1 ; bot3.y2v = 1 ; bot3.y2vr = 1
            bot3.y1w = 1 ; bot3.y1wr = 1 ; bot3.y2w = 1 ; bot3.y2wr = 1
            
        if no_pan == 4:
            bot2.x1u = 1 ; bot2.x1ur = 1 ; bot2.x2u = 1 ; bot2.x2ur = 1
            bot2.x1v = 1 ; bot2.x1vr = 1 ; bot2.x2v = 1 ; bot2.x2vr = 1 
            bot2.x1w = 1 ; bot2.x1wr = 1 ; bot2.x2w = 1 ; bot2.x2wr = 1 
            bot2.y1u = 1 ; bot2.y1ur = 1 ; bot2.y2u = 1 ; bot2.y2ur = 1
            bot2.y1v = 1 ; bot2.y1vr = 1 ; bot2.y2v = 1 ; bot2.y2vr = 1
            bot2.y1w = 1 ; bot2.y1wr = 1 ; bot2.y2w = 1 ; bot2.y2wr = 1
            
            bot3.x1u = 1 ; bot3.x1ur = 1 ; bot3.x2u = 1 ; bot3.x2ur = 1 
            bot3.x1v = 1 ; bot3.x1vr = 1 ; bot3.x2v = 1 ; bot3.x2vr = 1 
            bot3.x1w = 1 ; bot3.x1wr = 1 ; bot3.x2w = 1 ; bot3.x2wr = 1 
            bot3.y1u = 1 ; bot3.y1ur = 1 ; bot3.y2u = 1 ; bot3.y2ur = 1
            bot3.y1v = 1 ; bot3.y1vr = 1 ; bot3.y2v = 1 ; bot3.y2vr = 1
            bot3.y1w = 1 ; bot3.y1wr = 1 ; bot3.y2w = 1 ; bot3.y2wr = 1

            bot4.x1u = 1 ; bot4.x1ur = 1 ; bot4.x2u = bot_t ; bot4.x2ur = 1 
            bot4.x1v = 1 ; bot4.x1vr = 1 ; bot4.x2v = bot_t ; bot4.x2vr = 1 
            bot4.x1w = 1 ; bot4.x1wr = 1 ; bot4.x2w = bot_t ; bot4.x2wr = bot_r 
            bot4.y1u = 1 ; bot4.y1ur = 1 ; bot4.y2u = 1 ; bot4.y2ur = 1
            bot4.y1v = 1 ; bot4.y1vr = 1 ; bot4.y2v = 1 ; bot4.y2vr = 1
            bot4.y1w = 1 ; bot4.y1wr = 1 ; bot4.y2w = 1 ; bot4.y2wr = 1
            
    
    if no_x_gauss is None:
        no_x_gauss = 60
    if no_y_gauss is None:
        no_y_gauss = 30
    
    # All connections - list of dict
    if False: # incomplete
        if no_pan == 2:
            conn = [
             # skin-skin
             dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
             dict(p1=bot1, p2=bot2, func='SSxcte', xcte1=bot1.a, xcte2=0),
             dict(p1=top1, p2=bot1, func='SB'), #'_TSL', tsl_type = 'linear'), 
                # dict(p1=top2, p2=bot2, func='SB_TSL', tsl_type = 'linear')
            ]
    if no_pan == 3:
        conn = [
           # skin-skin
           dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
           dict(p1=top2, p2=top3, func='SSxcte', xcte1=top2.a, xcte2=0),
           dict(p1=bot1, p2=bot2, func='SSxcte', xcte1=bot1.a, xcte2=0),
           dict(p1=bot2, p2=bot3, func='SSxcte', xcte1=bot2.a, xcte2=0),
           dict(p1=top1, p2=bot1, func='SB_TSL', tsl_type = 'bilinear', no_x_gauss=no_x_gauss, 
                no_y_gauss=no_y_gauss, k_i=k_i, tau_o=tau_o, G1c=G1c, k_o=k_i, del_o=tau_o/k_i, del_f=2*G1c/tau_o)
        ]
    if no_pan == 4:
        conn = [
           # skin-skin
           dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
           dict(p1=top2, p2=top3, func='SSxcte', xcte1=top2.a, xcte2=0),
           dict(p1=top3, p2=top4, func='SSxcte', xcte1=top3.a, xcte2=0),
           dict(p1=bot1, p2=bot2, func='SSxcte', xcte1=bot1.a, xcte2=0),
           dict(p1=bot2, p2=bot3, func='SSxcte', xcte1=bot2.a, xcte2=0),
           dict(p1=bot3, p2=bot4, func='SSxcte', xcte1=bot3.a, xcte2=0),
           dict(p1=top1, p2=bot1, func='SB_TSL', tsl_type = 'bilinear', no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, k_i=k_i, tau_o=tau_o, G1c=G1c)
        ]
    
    # This determines the positions of each panel's (sub)matrix in the global matrix when made a MD obj below
    # So changing this changes the placements i.e. starting row and col of each
    if no_pan == 2:
        panels = [bot1, bot2, top1, top2]
    if no_pan == 3:
        panels = [bot1, bot2, bot3, top1, top2, top3]
    if no_pan == 4:
        panels = [bot1, bot2, bot3, bot4, top1, top2, top3, top4]

    assy = MultiDomain(panels=panels, conn=conn) # assy is now an object of the MultiDomain class
    # Here the panels (shell objs) are modified -- their starting positions in the global matrix is assigned etc

    # Panel at which the disp is applied
    if no_pan == 2:
        disp_panel = top2
    if no_pan == 3:
        disp_panel = top3
    if no_pan == 4:
        disp_panel = top4
 


    if True:
        ######## THIS SHOULD BE CHANGED LATER PER DISP TYPE ###########################################
        ku, kv, kw = calc_ku_kv_kw_point_pd(disp_panel)
        kw = 1e6
        
    size = assy.get_size()
        
    # To match the inital increment 
    wp = 0.01   
    
    # --------- IMPROVE THE STARTING GUESS --------------
    if False:
        # Prescribed Displacements
        if True:
            disp_type = 'point' # change based on what's being applied
            
            if disp_type == 'point':
                # Penalty Stiffness
                # Disp in z, so only kw is non zero. ku and kv are zero
                kCp = fkCpd(0., 0., kw, disp_panel, disp_panel.a, disp_panel.b/2, size, disp_panel.row_start, disp_panel.col_start)
                # Point load (added to shell obj)
                disp_panel.add_point_pd(disp_panel.a, disp_panel.b/2, 0., 0., 0., 0., kw, wp)
            if disp_type == 'line_xcte':
                kCp = fkCld_xcte(0., 0., kw, disp_panel, disp_panel.a, size, disp_panel.row_start, disp_panel.col_start)
                disp_panel.add_distr_pd_fixed_x(disp_panel.a, None, None, kw,
                                          funcu=None, funcv=None, funcw = lambda y: wp) #*y/top2.b, cte=True)
            if disp_type == 'line_ycte':
                kCp = fkCld_ycte(0., 0., kw, disp_panel, disp_panel.b, size, disp_panel.row_start, disp_panel.col_start)
                disp_panel.add_distr_pd_fixed_y(disp_panel.b, None, None, kw,
                                          funcu=None, funcv=None, funcw = lambda x: wp) #*x/top2.a, cte=True)
    
        # Stiffness matrix from penalties due to prescribed displacement
        kCp = finalize_symmetric_matrix(kCp)
        
        # Has to be after the forces/loads are added to the panels
        fext = assy.calc_fext()
        
        k0 = kT + kCp
        ci = solve(k0, fext)
        size = k0.shape[0]
    else:
        ci = np.zeros(size) # Only used to calc initial fint
        
        
    # -------------------- INCREMENTATION --------------------------
    # wp_max = 10 # [mm]
    # no_iter_disp = 100
    
    disp_iter_no = 0
    
    # Finding info of the connection
    for conn_list in conn:
        if conn_list['func'] == 'SB_TSL':
            no_x_gauss = conn_list['no_x_gauss']
            no_y_gauss = conn_list['no_y_gauss']
            tsl_type = conn_list['tsl_type']
            p_top = conn_list['p1']
            p_bot = conn_list['p2']
            break # Assuming there is only 1 connection
        
    # Initilaize mat to store results
    if w_iter_no_pts is None:
        w_iter_no_pts = 150
        
    w_iter = np.unique(np.concatenate((np.linspace(0.01,3,int(0.3*w_iter_no_pts)), np.linspace(3,5,int(0.3*w_iter_no_pts)), 
                                        np.linspace(5,8,int(0.4*w_iter_no_pts)))))
    # w_iter = np.linspace(0.01,8,100)
    # w_iter = [0.01, 2]
    # Reverse loading 
    # w_iter = np.concatenate((np.linspace(0.01,3,int(0.1*w_iter_no_pts)), np.linspace(3,6,int(0.25*w_iter_no_pts)), 
    #                                    np.linspace(6,7,int(0.25*w_iter_no_pts)),
    #                                    np.linspace(7,0,int(0.1*w_iter_no_pts)), np.linspace(0,7,int(0.1*w_iter_no_pts)), 
    #                                    np.linspace(7,8,int(0.2*w_iter_no_pts)) ))
    # w_iter = np.array([0.01, 1.01, 2.00, 3.00, 3.33, 3.67, 4.00, 4.33, 4.67, 5.00, 5.33,
    #        5.67, 6.00, 6.11, 6.22, 6.33, 6.44, 6.55, 6.67, 6.77, 6.89, 6.99, 7.00, 4.67, 2.50, 0.01, 2.50,
    #        4.67, 7.00, 7.14, 7.25, 7.43, 7.57, 7.7, 7.86, 7.92, 8.00])
    
    dmg_index = np.zeros((no_y_gauss,no_x_gauss,np.shape(w_iter)[0]))
    del_d = np.zeros((no_y_gauss,no_x_gauss,np.shape(w_iter)[0]))
    kw_tsl = np.zeros((no_y_gauss,no_x_gauss,np.shape(w_iter)[0]))
    force_intgn = np.zeros((np.shape(w_iter)[0], 2))
    force_intgn_dmg = np.zeros((np.shape(w_iter)[0], 2))
    displ_top_root = np.zeros((50,200,np.shape(w_iter)[0]))
    displ_bot_root = np.zeros((50,200,np.shape(w_iter)[0]))
    c_all = np.zeros((size, np.shape(w_iter)[0]))
    
    quasi_NR = True
    NR_kT_update = 3 # After how many iterations should kT be updated
    crisfield_test_fail = False
    

    
    # Displacement Incrementation
    for wp in w_iter: 
        # print(f'------------ wp = {wp:.3f} ---------------')
        log_file = open(f'{filename}.txt', 'a')
        log_file.write(f'------------ wp = {wp:.3f} ---------------\n')
        log_file.close()
        
        # Prescribed Displacements
        if True:
            disp_type = 'line_xcte' # change based on what's being applied
            
            # Clears all previously added displs - otherwise in NL case, theyre readded so you have 2 disps at the tip
            disp_panel.clear_disps()
            
            if disp_type == 'point':
                # Penalty Stiffness
                # Disp in z, so only kw is non zero. ku and kv are zero
                kCp = fkCpd(0., 0., kw, disp_panel, disp_panel.a, disp_panel.b/2, size, disp_panel.row_start, disp_panel.col_start)
                # Point load (added to shell obj)
                disp_panel.add_point_pd(disp_panel.a, disp_panel.b/2, 0., 0., 0., 0., kw, wp)
            if disp_type == 'line_xcte':
                kCp = fkCld_xcte(0., 0., kw, disp_panel, disp_panel.a, size, disp_panel.row_start, disp_panel.col_start)
                disp_panel.add_distr_pd_fixed_x(disp_panel.a, None, None, kw,
                                           funcu=None, funcv=None, funcw = lambda y: wp) #*y/top2.b, cte=True)
            if disp_type == 'line_ycte':
                kCp = fkCld_ycte(0., 0., kw, disp_panel, disp_panel.b, size, disp_panel.row_start, disp_panel.col_start)
                disp_panel.add_distr_pd_fixed_y(disp_panel.b, None, None, kw,
                                           funcu=None, funcv=None, funcw = lambda x: wp) #*x/top2.a, cte=True)

        # Stiffness matrix from penalties due to prescribed displacement
        kCp = finalize_symmetric_matrix(kCp)
        
        # Has to be after the forces/loads are added to the panels
        fext = assy.calc_fext()

        # Inital guess ci and increment dc (same size as fext)
        dc = np.zeros_like(fext)
        
        
        # All subsequent load steps
        if disp_iter_no != 0:
            # Doing this to avoid recalc kc_conn and help pass the correct k_i
            kC_conn = assy.get_kC_conn(c=c)
            # Inital fint (0 bec c=0)
            fint = np.asarray(assy.calc_fint(c=c, kC_conn=kC_conn))
            # Stiffness matrix for the crack creation 
            if disp_iter_no != 1:
                kcrack = assy.calc_kcrack(conn=conn, c_i=c, kw_tsl_i_1=kw_tsl[:,:,disp_iter_no-1], 
                                          del_d_i_1=del_d[:,:,disp_iter_no-1])
            else:
                kcrack = np.zeros_like(kC_conn, dtype=np.float64)
            # force vector due to the crack creation
            fcrack = assy.calc_fcrack(c_i=c, kw_tsl_i_1=kw_tsl[:,:,disp_iter_no-1], 
                                      del_d_i_1=del_d[:,:,disp_iter_no-1], conn=conn, kcrack=kcrack)
            # Residual with disp terms
            Ri = fint - fext + kCp*c + fcrack
            
            kT = assy.calc_kT(c=c, kC_conn=kC_conn) + kcrack
        # Initial step where ci = zeros
        if disp_iter_no == 0 and np.max(ci) == 0: 
            # max(ci) bec if its already calc for an initial guess above, no need to repeat it - 
            #                   only when c is zero (or randomly initialized - modify 'if' then)
            
            # Doing this to avoid recalc kc_conn and help pass the correct k_i
            kC_conn = assy.get_kC_conn(c=ci)
            # Inital fint (0 bec c=0)
            fint = np.asarray(assy.calc_fint(c=ci, kC_conn=kC_conn))
            # Residual with disp terms
            Ri = fint - fext + kCp*ci
            kT = assy.calc_kT(c=ci, kC_conn=kC_conn)
            
            # Setting the max displ field to be 0
            assy.update_max_del_d(curr_max_del_d=np.zeros((no_y_gauss, no_x_gauss)))
        k0 = kT + kCp
        
        
        # print(f'kT {np.max(kT):.3e} kcp {np.max(kCp):.3e} k0 {np.max(k0):.3e}')
        
        epsilon = 1.e-4 # Convergence criteria
        D = k0.diagonal() # For convergence - calc at beginning of load increment
        
        count = 0 # Tracks number of NR iterations 
        
        crisfield_test_res = np.zeros(1000)

        # Modified Newton Raphson Iteration
        while True:
            # print()
            # print(f"------------ NR start {count+1}--------------")
            dc = solve(k0, -Ri, silent=True)
            c = ci + dc
            
            if quasi_NR:
                if count == 0 or count % NR_kT_update == 0:
                    kC_conn = assy.get_kC_conn(c=c)
                    if disp_iter_no != 0:
                        kcrack = assy.calc_kcrack(conn=conn, c_i=c, kw_tsl_i_1=kw_tsl[:,:,disp_iter_no-1], 
                                              del_d_i_1=del_d[:,:,disp_iter_no-1])
                    else:
                        kcrack = assy.calc_kcrack(conn=conn, c_i=c, kw_tsl_i_1=k_i*np.ones((no_y_gauss, no_x_gauss)), 
                                              del_d_i_1=np.zeros((no_y_gauss, no_x_gauss)))
            else:
                kC_conn = assy.get_kC_conn(c=c)
                if disp_iter_no != 0:
                    kcrack = assy.calc_kcrack(conn=conn, c_i=c, kw_tsl_i_1=kw_tsl[:,:,disp_iter_no-1], 
                                          del_d_i_1=del_d[:,:,disp_iter_no-1])
                else:
                    kcrack = assy.calc_kcrack(conn=conn, c_i=c, kw_tsl_i_1=k_i*np.ones((no_y_gauss, no_x_gauss)), 
                                          del_d_i_1=np.zeros((no_y_gauss, no_x_gauss)))
                    
            fint = np.asarray(assy.calc_fint(c=c, kC_conn=kC_conn))
            # force vector due to the crack creation
            if disp_iter_no != 0:
                fcrack = assy.calc_fcrack(c_i=c, kw_tsl_i_1=kw_tsl[:,:,disp_iter_no-1], 
                                      del_d_i_1=del_d[:,:,disp_iter_no-1], conn=conn, kcrack=kcrack)
            else:
                fcrack = assy.calc_fcrack(c_i=c, kw_tsl_i_1=k_i*np.ones((no_y_gauss, no_x_gauss)), 
                                      del_d_i_1=np.zeros((no_y_gauss, no_x_gauss)), conn=conn, kcrack=kcrack)
            
            # Might need to calc fext here again if it changes per iteration when fext changes when not using kc_conn for SB
            # CAN PROBABLY REMOVE IF !!!!!!!!!!!!
            if disp_iter_no != 0:
                Ri = fint - fext + kCp*c + fcrack
            else:
                Ri = fint - fext + kCp*c + fcrack
            if local_run:
                print(f'Ri {np.linalg.norm(Ri):.2e}')
                # print(f'pos sun:{np.linalg.norm(fint) + np.linalg.norm(kCp*c) + np.linalg.norm(fcrack):.2e} fint {np.linalg.norm(fint):.2e} kcp {np.linalg.norm(kCp*c):.2e} fcrack {np.linalg.norm(fcrack):.2e}')
                # print(f'fext {np.linalg.norm(fext):.2e}')
            # Extracting data out of the MD class for plotting etc 
                # Damage considerations are already implemented in the Multidomain class functions kT, fint - no need to include anything externally

            # kw_tsl_iter, dmg_index_iter, del_d_iter = assy.calc_k_dmg(c=c, pA=p_top, pB=p_bot, no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, tsl_type=tsl_type)
            # print(f"             NR end {count} -- wp={wp:.3f}  max dmg {np.max(dmg_index_iter):.3f}  ---") # min del_d {np.min(del_d_iter):.2e}---------")
            # print(f'    del_d min {np.min(del_d_iter)}  -- max {np.max(del_d_iter):.4f}')
            
            # CAN PROBABLY REMOVE IF !!!!!!!!!!!!
            if disp_iter_no != 0:
                crisfield_test = scaling(Ri, D)/max(scaling(fext, D), scaling(fint + kCp*c + fcrack, D))
            else:
                crisfield_test = scaling(Ri, D)/max(scaling(fext, D), scaling(fint + kCp*c + fcrack, D))
            crisfield_test_res[count] = crisfield_test
            # print(np.shape(k0))
            # print(np.linalg.det(k0))
            if local_run:
                print(f'    crisfield {crisfield_test:.4e} ')
            else:
                log_file = open(f'{filename}.txt', 'a')
                log_file.write(f'       crisfield {crisfield_test:.4e}\n')
                log_file.close()
            if crisfield_test < epsilon:
                break
            
            if quasi_NR:
                if count == 0 or count % NR_kT_update == 0:
                    kT = assy.calc_kT(c=c, kC_conn=kC_conn) 
                    # CAN PROBABLY REMOVE IF !!!!!!!!!!!!
                    if disp_iter_no != 0:
                        k0 = kT + kCp + kcrack
                    else:
                        k0 = kT + kCp + kcrack
            else:
                kT = assy.calc_kT(c=c, kC_conn=kC_conn) 
                # CAN PROBABLY REMOVE IF !!!!!!!!!!!!
                if disp_iter_no != 0:
                    k0 = kT + kCp + kcrack
                else:
                    k0 = kT + kCp + kcrack
            ci = c.copy()
            
            count += 1
            
            # Changing convergence criteria to prevent solutions from diverging
                # Context: Current soltuions get close to epsilon before diverging
            if False:
                if crisfield_test_res[count-1] > crisfield_test_res[0] and crisfield_test_res[count-2] > crisfield_test_res[0]:
                # Two consecutive crisfield values are larger than the starting guess and epsilon is still small enough, reset c and try again
                    if epsilon < 1e-3:
                        epsilon = epsilon*1.2
                        count = 0
                        # Updating c to converged c of the previous displacement step
                        c = c_all[:,disp_iter_no-1]
                        Ri = fint - fext + kCp*c + fcrack
                        kT = assy.calc_kT(c=c, kC_conn=kC_conn) 
                        k0 = kT + kCp
                    else:
                        log_file = open(f'{filename}.txt', 'a')
                        log_file.write(f'ABORTED DUE TO DIVERGING RESULTS {filename}\n')
                        log_file.close()
                        crisfield_test_fail = True
                        break
            

            
            # ------------------ SAVING VARIABLES --------------------
            if True:
                np.save(f'dmg_index_{filename}', dmg_index)
                np.save(f'del_d_{filename}', del_d)
                np.save(f'kw_tsl_{filename}', kw_tsl)
                np.save(f'force_intgn_{filename}', force_intgn)
                np.save(f'force_intgn_dmg_{filename}', force_intgn_dmg)
                # np.save(f'displ_top_root_{filename}', displ_top_root)
                # np.save(f'displ_bot_root_{filename}', displ_bot_root)
                np.save(f'c_all_{filename}', c_all)
            
            # Saving results incase cluster time is exceeded
            if (time.time() - start_time)/(60*60) > 70: # Max wall time (hrs) on the cluster
                np.save(f'dmg_index_{filename}', dmg_index)
                np.save(f'del_d_{filename}', del_d)
                np.save(f'kw_tsl_{filename}', kw_tsl)
                np.save(f'force_intgn_{filename}', force_intgn)
                np.save(f'force_intgn_dmg_{filename}', force_intgn_dmg)
                # np.save(f'displ_top_root_{filename}', displ_top_root)
                # np.save(f'displ_bot_root_{filename}', displ_bot_root)
                np.save(f'c_all_{filename}', c_all)
            
            # Kills this run but prevents other runs in parallel from being killed due to errors
            if False:
                if crisfield_test >= 1:
                    if True:
                        np.save(f'dmg_index_{filename}', dmg_index)
                        np.save(f'del_d_{filename}', del_d)
                        np.save(f'kw_tsl_{filename}', kw_tsl)
                        np.save(f'force_intgn_{filename}', force_intgn)
                        np.save(f'force_intgn_dmg_{filename}', force_intgn_dmg)
                        # np.save(f'displ_top_root_{filename}', displ_top_root)
                        # np.save(f'displ_bot_root_{filename}', displ_bot_root)
                        np.save(f'c_all_{filename}', c_all)
                    
                    crisfield_test_fail = True
                    break
            
            if count > 1000:
                log_file = open(f'{filename}.txt', 'a')
                log_file.write('Unconverged Results !!!!!!!!!!!!!!!!!!!\n')
                log_file.close()
                # return None, dmg_index, del_d, kw_tsl
                if True:
                    np.save(f'dmg_index_{filename}', dmg_index)
                    np.save(f'del_d_{filename}', del_d)
                    np.save(f'kw_tsl_{filename}', kw_tsl)
                    np.save(f'force_intgn_{filename}', force_intgn)
                    np.save(f'force_intgn_dmg_{filename}', force_intgn_dmg)
                    # np.save(f'displ_top_root_{filename}', displ_top_root)
                    # np.save(f'displ_bot_root_{filename}', displ_bot_root)
                    np.save(f'c_all_{filename}', c_all)
                # raise RuntimeError('NR didnt converged :(') 
                print(f'{filename} -- NR didnt converged :(')
                crisfield_test_fail = True
                break
        
        if crisfield_test_fail is True:
            print(f'????????????????????? ABORTED {filename} ?????????????????????????????')
            log_file = open(f'{filename}.txt', 'a')
            log_file.write(f'????????????????????? ABORTED {filename} ?????????????????????????????\n')
            log_file.close()
            sys.stdout.flush()
            break
        
        # Edit: should be correct as it is 
                # Ques: should first update max del d then calc kw_tsl etc ??
        if hasattr(assy, "del_d"):
            kw_tsl[:,:,disp_iter_no], dmg_index[:,:,disp_iter_no], del_d[:,:,disp_iter_no]  = assy.calc_k_dmg(c=c, pA=p_top, pB=p_bot, 
                                                 no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, tsl_type=tsl_type, 
                                                 prev_max_del_d=assy.del_d, k_i=k_i, tau_o=tau_o)
        else:
            kw_tsl[:,:,disp_iter_no], dmg_index[:,:,disp_iter_no], del_d[:,:,disp_iter_no]  = assy.calc_k_dmg(c=c, pA=p_top, pB=p_bot, 
                                                 no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, tsl_type=tsl_type,
                                                 prev_max_del_d=None, k_i=k_i, tau_o=tau_o)
            
        c_all[:,disp_iter_no] = c
            
        # Update max del_d AFTER a converged NR iteration
        assy.update_max_del_d(curr_max_del_d=del_d[:,:,disp_iter_no])
        
        # kw_tsl[:,:,disp_iter_no], dmg_index[:,:,disp_iter_no], del_d[:,:,disp_iter_no] = assy.calc_k_dmg(c=c, pA=p_top, pB=p_bot, no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, tsl_type=tsl_type)
        if not local_run:
            log_file = open(f'{filename}.txt', 'a')
            log_file.write(f'{disp_iter_no/np.shape(w_iter)[0]*100:.1f}% - {filename} -- wp={wp:.3f} -- max DI {np.max(dmg_index[:,:,disp_iter_no]):.4f}\n')
            log_file.close()
            print(f'            {filename} {disp_iter_no/np.shape(w_iter)[0]*100:.1f}%')
            sys.stdout.flush()
        else:
            print(f'{disp_iter_no/np.shape(w_iter)[0]*100:.1f}% - {filename} -- wp={wp:.3f} -- max DI {np.max(dmg_index[:,:,disp_iter_no]):.4f}\n')

        # print(f'        max del_d {np.max(del_d[:,:,disp_iter_no])}')
        # print(f'       min del_d {np.min(del_d[:,:,disp_iter_no])}')
        # print(f'        max kw_tsl {np.max(kw_tsl[:,:,disp_iter_no])}')
        

        
        # Force - TEMP - CHECK LATER AND EDIT/REMOVE
        if True:
            # force_intgn[disp_iter_no, 0] = wp
            force_intgn[disp_iter_no, 1] = assy.force_out_plane(c, group=None, eval_panel=disp_panel, x_cte_force=None, y_cte_force=None,
                      gridx=100, gridy=50, NLterms=True, no_x_gauss=128, no_y_gauss=128)
        else: 
            force_intgn = None
        
        # Force by area integral of traction in the damaged region
        if True:
            # force_intgn_dmg[disp_iter_no, 0] = wp
            force_intgn_dmg[disp_iter_no, 1] = assy.force_out_plane_damage(conn=conn, c=c)
            
        if not local_run:
            log_file = open(f'{filename}.txt', 'a')
            log_file.write(f'       Force - Line: {force_intgn[disp_iter_no, 1]:.3f} -- Area: {force_intgn_dmg[disp_iter_no, 1]:.3f}\n')
            log_file.close()
        else:
            print(f'       Force - Line: {force_intgn[disp_iter_no, 1]:.3f} -- Area: {force_intgn_dmg[disp_iter_no, 1]:.3f}')
            
            plt.figure(figsize=(10,7))
            plt.plot(force_intgn[:disp_iter_no+1, 0], force_intgn[:disp_iter_no+1, 1], label = 'Line')
            plt.plot(force_intgn_dmg[:disp_iter_no+1, 0], 4/52*force_intgn_dmg[:disp_iter_no+1, 1], label='Area Int')
            plt.plot(FEM[:,0],FEM[:,1], label='FEM')
            plt.ylabel('Force [N]', fontsize=14)
            plt.xlabel('Displacement [mm]', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.title(f'{filename}')
            plt.grid()
            plt.legend(fontsize=14)
            plt.show()
        
        # Calc displ of top and bottom panels at each increment
        if True:
            res_pan_top = assy.calc_results(c=c, eval_panel=disp_panel, vec='w', 
                                    no_x_gauss=200, no_y_gauss=50)
            max_displ_w = np.max(res_pan_top['w'][-1])
            
            force_intgn[disp_iter_no, 0] = max_displ_w
            force_intgn_dmg[disp_iter_no, 0] = max_displ_w
            
            # print(f'Calc W_TOP: {max_displ_w}')
            # res_pan_bot = assy.calc_results(c=c, eval_panel=disp_panel, vec='w', 
            #                         no_x_gauss=200, no_y_gauss=50)
            # displ_top_root[:,:,disp_iter_no] = res_pan_top['w'][0]
            # displ_bot_root[:,:,disp_iter_no] = res_pan_bot['w'][0]
        else:
            res_pan_top = None
            res_pan_bot = None
        
        disp_iter_no += 1
        print()
        
        sys.stdout.flush()
        
        if np.all(dmg_index == 1):
            print('Cohesive Zone has failed')
            break
        
        
        
    # ------------------ SAVING VARIABLES --------------------
    if True:
        np.save(f'dmg_index_{filename}', dmg_index)
        np.save(f'del_d_{filename}', del_d)
        np.save(f'kw_tsl_{filename}', kw_tsl)
        np.save(f'force_intgn_{filename}', force_intgn)
        np.save(f'force_intgn_dmg_{filename}', force_intgn_dmg)
        # np.save(f'displ_top_root_{filename}', displ_top_root)
        # np.save(f'displ_bot_root_{filename}', displ_bot_root)
        np.save(f'c_all_{filename}', c_all)
    
    print(f'************************** WAKE UP! ITS DONE! - {filename} ******************************')
    sys.stdout.flush()
    
    log_file = open(f'{filename}.txt', 'a')
    log_file.write((f'************************** WAKE UP! ITS DONE! - {filename} ******************************\n'))
    log_file.close()
    
    # ------------------ RESULTS AND POST PROCESSING --------------------
    
    c0 = c.copy()

    generate_plots = False
    
    final_res = None
    # Plotting results
    if False:
        for vec in ['w']:#, 'Mxx']:#, 'Myy', 'Mxy']:#, 'Nxx', 'Nyy']:
            res_bot = assy.calc_results(c=c0, group='bot', vec=vec, no_x_gauss=None, no_y_gauss=None)
            res_top = assy.calc_results(c=c0, group='top', vec=vec, no_x_gauss=None, no_y_gauss=None)
            vecmin = min(np.min(np.array(res_top[vec])), np.min(np.array(res_bot[vec])))
            vecmax = max(np.max(np.array(res_top[vec])), np.max(np.array(res_bot[vec])))
            if vec != 'w':
                print(f'{vec} :: {vecmin:.3f}  {vecmax:.3f}')
            # if vec == 'w':
            if True:
                # Printing max min per panel
                if False:
                    for pan in range(0,np.shape(res_bot[vec])[0]):
                        print(f'{vec} top{pan+1} :: {np.min(np.array(res_top[vec][pan])):.3f}  {np.max(np.array(res_top[vec][pan])):.3f}')
                    print('------------------------------')
                    for pan in range(0,np.shape(res_bot[vec])[0]): 
                        print(f'{vec} bot{pan+1} :: {np.min(np.array(res_bot[vec][pan])):.3f}  {np.max(np.array(res_bot[vec][pan])):.3f}')
                    print('------------------------------')
                # print(f'Global TOP {vec} :: {np.min(np.array(res_top[vec])):.3f}  {np.max(np.array(res_top[vec])):.3f}')
                # print(f'Global BOT {vec} :: {np.min(np.array(res_bot[vec])):.3f}  {np.max(np.array(res_bot[vec])):.3f}')
                # print(res_bot[vec][1][:,-1]) # disp at the tip
                max_displ_w = np.max(res_top['w'][-1])
                print(f'Calc W_TOP: {max_displ_w}')
                final_res = np.min(np.array(res_top[vec]))
            
            if generate_plots:
                # if vec == 'w':
                if True:
                    assy.plot(c=c0, group='bot', vec=vec, filename='test_dcb_before_opening_bot_tsl.png', show_boundaries=True,
                                                colorbar=True, res = res_bot, vecmax=vecmax, vecmin=vecmin, display_zero=True)
                    
                    assy.plot(c=c0, group='top', vec=vec, filename='test_dcb_before_opening_top_tsl.png', show_boundaries=True,
                                              colorbar=True, res = res_top, vecmax=vecmax, vecmin=vecmin, display_zero=True)
            
            # Open images
            if generate_plots:
                img_popup('test_dcb_before_opening_top_tsl.png',1, f"{vec} top")
                img_popup('test_dcb_before_opening_bot_tsl.png',2, f"{vec} bot")
                plt.show()
    
    animate = False
    if animate:    
        def animate(i):
            curr_res = frames[i]
            max_res = np.max(curr_res)
            min_res = np.min(curr_res)
            if animate_var == 'dmg_index':
                if min_res == 0:
                    vmin = 0.0
                    vmax = 1.0
                else: 
                    possible_min_cbar = [0,0.5,0.85,0.9,0.95,0.99]
                    vmin = max(list(filter(lambda x: x<min_res, possible_min_cbar)))
                    vmax = 1.0
            else:
                vmin = min_res
                vmax = max_res
            im = ax.imshow(curr_res)
            fig.colorbar(im, cax=cax)
            im.set_data(curr_res)
            im.set_clim(vmin, vmax)
            tx.set_text(f'{animate_var}     -   Disp={w_iter[i]:.2f} mm')
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        for animate_var in ["dmg_index", "del_d"]:#, 'kw_tsl']:
            
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
                                          interval = 200, repeat_delay=1000)
            FFwriter = animation.FFMpegWriter(fps=5)
            ani.save(f'{animate_var}.mp4', writer=FFwriter)
            # ani.save(f'{animate_var}.gif', writer='imagemagick')
        
    return dmg_index, del_d, kw_tsl, force_intgn, displ_top_root, displ_bot_root, c_all



if __name__ == "__main__":
    
    # Connection as penalty stiffness
    if True:
        # Single test run
        if False: # wo fcrack
            # test_dcb_non_linear(3, 4, 1)
            # kw_tsl, dmg_index = test_kw_tsl(1, 6, 1)
            dmg_index, del_d, kw_tsl, force_intgn, displ_top_root, displ_bot_root, c_all = test_dcb_damage_prop(no_pan=3, no_terms=[10,8], filename='', k_i=1e4, tau_o=67, no_x_gauss=50, no_y_gauss=25, w_iter_no_pts=15, G1c=1.12)
        if True: # with fcrack
            dmg_index, del_d, kw_tsl, force_intgn, displ_top_root, displ_bot_root, c_all = test_dcb_damage_prop_fcrack(phy_dim=[3,25], 
                no_terms=[15,8], filename='1posve_m15_8_w200_nx60_ny30_tau67_ki1e4', k_i=1e4, tau_o=67, no_x_gauss=60, 
                no_y_gauss=30, w_iter_no_pts=200, G1c=1.12)
        
        
        # Parallelization of multiple runs
        if False:
            # test_dcb_damage_prop(no_pan, no_terms, filename='', k_i=None, tau_o=None, 
                                # no_x_gauss=None, no_y_gauss=None, w_iter_no_pts=None):                
            if False:
                ki_all = [1e5, 5e4, 1e4, 5e3]
                inp_all = np.zeros((16,6)) # 16=4*4, 6 for 6 col
                m = 10
                count_inp = 0
                for ki in ki_all:
                    # 4 bec each inp line as 4 lines
                    inp_all[4*count_inp : 4*count_inp+4,:] = np.array(([3,m,ki,87,150,80],
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
                    
                    # for nx, ny in zip([30,50,60,80,100,30,50,60,80,100],[15,25,30,40,50,8,13,15,20,25]
            
            ftn_arg = [([3,25],[15,8],f'p3_25_m15_8_ki{ki:.0e}_tauo77_nx60_ny30_wpts50_G1c112',ki,77,60,30,40,1.12) for ki in [1e4, 5e4, 1e5, 5e5, 1e6]]
            for i in range(len(ftn_arg)):
                ftn_arg[i] = list(ftn_arg[i]) # convert to list bec tuples are inmutable
                ftn_arg[i][2] = ftn_arg[i][2].replace('+0','')
                ftn_arg[i] = tuple(ftn_arg[i]) # convert back to tuple
            
            # create the process pool
            with Pool(processes=20) as pool: # using 20 cpus on the cluster
                pool.starmap(test_dcb_damage_prop_fcrack, ftn_arg)
                # the pool is processing its inputs in parallel, close() and join() 
                #can be used to synchronize the main process 
                #with the task processes to ensure proper cleanup.
                pool.close()
                pool.join()
         

                
                
