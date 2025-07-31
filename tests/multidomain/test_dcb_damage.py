import sys
sys.path.append('../..')
import os

import numpy as np
from structsolve import solve
from structsolve.sparseutils import finalize_symmetric_matrix, remove_null_cols
import time
import scipy

from panels.shell import Shell
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


def img_popup(filename, plot_no = None, title = None):
    '''
    plot_no = current plot no
    '''
    image = img.imread(filename)
    if plot_no is None:
        if title is None:
            plt.title(filename)
        else:
            plt.title(title)
        plt.imshow(image)
        #plt.show()
    else:
        if title is None:
            plt.subplot(1,2,plot_no).set_title(filename)
        else:
            plt.subplot(1,2,plot_no).set_title(title)
        plt.imshow(image)


def dcb_damage_prop_no_f_kcrack(phy_dim, nr_terms, k_i=None, tau_o=None, nr_x_gauss=None, nr_y_gauss=None, w_iter_info=None, G1c=None, name=''):
    r"""Damage propagation from a DCB with a precrack

        Code for 2 panels might not be right

        All units in MPa, N, mm
    """
    filename=name[0]
    foldername=name[1]

    # Delete previous log file - later when appending the file, if the file doesnt exist, it is created
    if os.path.isfile(f'./{filename}.txt'):
        os.remove(f'{filename}.txt')

    print(f'************************** HELLO THERE! IT HAS BEGUN - {filename} ******************************')
    sys.stdout.flush()
    # Start time
    start_time = time.time()

    nr_pan = phy_dim[0]
    a = phy_dim[1]
    b = phy_dim[2]
    precrack_len = phy_dim[3]

    # Properties
    if True: # Bas's paper (TP - 65,25,48)
        E1 = (138300. + 128000.)/2. # MPa
        E2 = (10400. + 11500.)/2. # MPa
        G12 = 5190. # MPa
        nu12 = 0.316
        ply_thickness = 0.14 # mm
        plies = 15
        simple_layup = [0]*plies
    if False: # Bas's paper with QI laminate
        E1 = (138300. + 128000.)/2. # MPa
        E2 = (10400. + 11500.)/2. # MPa
        G12 = 5190. # MPa
        nu12 = 0.316
        ply_thickness = 0.2625 # mm
        simple_layup = [0,90,45,-45]
        simple_layup += simple_layup[::-1]
    if False: # Theodores thesis
        E1 = 126100. # MPa
        E2 = 11200. # MPa
        G12 = 5460. # MPa
        nu12 = 0.319
        ply_thickness = 0.14 # mm
        plies = 15
        simple_layup = [0]*plies
    if False: # Thermoset
        E1 = (137100. + 114300.)/2. # MPa
        E2 = (8800. + 10100.)/2. # MPa
        G12 = 4900. # MPa
        nu12 = 0.314
        ply_thickness = 0.14 # mm
        plies = 15
        simple_layup = [0]*plies

    # Plate dimensions (overall)
    # a = 65 # mm
    # b = 25  # mm
    # Dimensions of panel 1 and 2
    a1 = a - precrack_len
    a2 = 0.015
    a3 = 10

    #others
    m_tsl = nr_terms[0]
    n_tsl = nr_terms[1]
    m = nr_terms[2]
    n = nr_terms[3]

    # simple_layup += simple_layup[::-1]
    # print('plies ',np.shape(simple_layup)[0])

    laminaprop = (E1, E2, nu12, G12, G12, G12)

    # nr_pan = 3

    # Top DCB panels
    top1 = Shell(group='top', x0=0, y0=0, a=a1, b=b, m=m_tsl, n=n_tsl, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if nr_pan == 2:
        top2 = Shell(group='top', x0=a1, y0=0, a=a-a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    elif nr_pan == 3:
        top2 = Shell(group='top', x0=a1, y0=0, a=a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        top3 = Shell(group='top', x0=a1+a2, y0=0, a=a-a1-a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    elif nr_pan == 4:
        top2 = Shell(group='top', x0=a1, y0=0, a=a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        top3 = Shell(group='top', x0=a1+a2, y0=0, a=a3, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        top4 = Shell(group='top', x0=a1+a2+a3, y0=0, a=a-(a1+a2+a3), b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)

    # Bottom DCB panels
    bot1 = Shell(group='bot', x0=0, y0=0, a=a1, b=b, m=m_tsl, n=n_tsl, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if nr_pan == 2:
        bot2 = Shell(group='bot', x0=a1, y0=0, a=a-a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    elif nr_pan == 3:
        bot2 = Shell(group='bot', x0=a1, y0=0, a=a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        bot3 = Shell(group='bot', x0=a1+a2, y0=0, a=a-a1-a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    elif nr_pan == 4:
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

        if nr_pan == 3:
            top3.x1u = 1 ; top3.x1ur = 1 ; top3.x2u = 1 ; top3.x2ur = 1
            top3.x1v = 1 ; top3.x1vr = 1 ; top3.x2v = 1 ; top3.x2vr = 1
            top3.x1w = 1 ; top3.x1wr = 1 ; top3.x2w = 1 ; top3.x2wr = 1
            top3.y1u = 1 ; top3.y1ur = 1 ; top3.y2u = 1 ; top3.y2ur = 1
            top3.y1v = 1 ; top3.y1vr = 1 ; top3.y2v = 1 ; top3.y2vr = 1
            top3.y1w = 1 ; top3.y1wr = 1 ; top3.y2w = 1 ; top3.y2wr = 1

        elif nr_pan == 4:
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

        if nr_pan == 2:
            bot2.x1u = 1 ; bot2.x1ur = 1 ; bot2.x2u = bot_t ; bot2.x2ur = 1
            bot2.x1v = 1 ; bot2.x1vr = 1 ; bot2.x2v = bot_t ; bot2.x2vr = 1
            bot2.x1w = 1 ; bot2.x1wr = 1 ; bot2.x2w = bot_t ; bot2.x2wr = bot_r
            bot2.y1u = 1 ; bot2.y1ur = 1 ; bot2.y2u = 1 ; bot2.y2ur = 1
            bot2.y1v = 1 ; bot2.y1vr = 1 ; bot2.y2v = 1 ; bot2.y2vr = 1
            bot2.y1w = 1 ; bot2.y1wr = 1 ; bot2.y2w = 1 ; bot2.y2wr = 1

        elif nr_pan == 3:
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

        elif nr_pan == 4:
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


    if nr_x_gauss is None:
        nr_x_gauss = 60
    if nr_y_gauss is None:
        nr_y_gauss = 30

    # All connections - list of dict
    if False: # incomplete
        if nr_pan == 2:
            conn = [
             # skin-skin
             dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
             dict(p1=bot1, p2=bot2, func='SSxcte', xcte1=bot1.a, xcte2=0),
             dict(p1=top1, p2=bot1, func='SB'), #'_TSL', tsl_type = 'linear'),
                # dict(p1=top2, p2=bot2, func='SB_TSL', tsl_type = 'linear')
            ]
    if nr_pan == 3:
        conn = [
           # skin-skin
           dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
           dict(p1=top2, p2=top3, func='SSxcte', xcte1=top2.a, xcte2=0),
           dict(p1=bot1, p2=bot2, func='SSxcte', xcte1=bot1.a, xcte2=0),
           dict(p1=bot2, p2=bot3, func='SSxcte', xcte1=bot2.a, xcte2=0),
           dict(p1=top1, p2=bot1, func='SB_TSL', tsl_type = 'bilinear', nr_x_gauss=nr_x_gauss,
                nr_y_gauss=nr_y_gauss, tau_o=tau_o, G1c=G1c, k_o=k_i, del_o=tau_o/k_i, del_f=2*G1c/tau_o)
        ]
    elif nr_pan == 4:
        conn = [
           # skin-skin
           dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
           dict(p1=top2, p2=top3, func='SSxcte', xcte1=top2.a, xcte2=0),
           dict(p1=top3, p2=top4, func='SSxcte', xcte1=top3.a, xcte2=0),
           dict(p1=bot1, p2=bot2, func='SSxcte', xcte1=bot1.a, xcte2=0),
           dict(p1=bot2, p2=bot3, func='SSxcte', xcte1=bot2.a, xcte2=0),
           dict(p1=bot3, p2=bot4, func='SSxcte', xcte1=bot3.a, xcte2=0),
           dict(p1=top1, p2=bot1, func='SB_TSL', tsl_type = 'bilinear', nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss, k_i=k_i, tau_o=tau_o, G1c=G1c)
        ]

    # This determines the positions of each panel's (sub)matrix in the global matrix when made a MD obj below
    # So changing this changes the placements i.e. starting row and col of each
    if nr_pan == 2:
        panels = [bot1, bot2, top1, top2]
    elif nr_pan == 3:
        panels = [bot1, bot2, bot3, top1, top2, top3]
    elif nr_pan == 4:
        panels = [bot1, bot2, bot3, bot4, top1, top2, top3, top4]

    assy = MultiDomain(panels=panels, conn=conn) # assy is now an object of the MultiDomain class
    # Here the panels (shell objs) are modified -- their starting positions in the global matrix is assigned etc

    # Panel at which the disp is applied
    if nr_pan == 2:
        disp_panel = top2
    elif nr_pan == 3:
        disp_panel = top3
    elif nr_pan == 4:
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
    # nr_iter_disp = 100

    disp_iter_no = 0

    # Finding info of the connection
    for conn_list in conn:
        if conn_list['func'] == 'SB_TSL':
            nr_x_gauss = conn_list['nr_x_gauss']
            nr_y_gauss = conn_list['nr_y_gauss']
            tsl_type = conn_list['tsl_type']
            p_top = conn_list['p1']
            p_bot = conn_list['p2']
            break # Assuming there is only 1 connection

    w_iter_nr_pts = w_iter_info[0]
    w_max = w_iter_info[1]

    # Initilaize mat to store results
    if w_iter_nr_pts is None:
        w_iter_nr_pts = 50

    load_reversal = True
    if not load_reversal:
        w_iter = np.unique(np.concatenate((np.linspace(0.01,0.375*w_max,int(0.3*w_iter_nr_pts)), np.linspace(0.375*w_max,0.625*w_max,int(0.3*w_iter_nr_pts)),
                                        np.linspace(0.625*w_max,w_max,int(0.4*w_iter_nr_pts)))))
    else:
        w_iter = np.concatenate((np.linspace(0.01,0.375*w_max,int(0.3*w_iter_nr_pts)), np.linspace(0.375*w_max,0.625*w_max,int(0.3*w_iter_nr_pts)),
                                 np.linspace(0.625*w_max,0.9166*w_max,int(0.3*w_iter_nr_pts)), np.array([1]),
                                 np.linspace(0.9166*w_max,1*w_max,int(0.1*w_iter_nr_pts)) ))

    # Initilize variables to store results
    dmg_index = np.zeros((nr_y_gauss,nr_x_gauss,np.shape(w_iter)[0]))
    del_d = np.zeros((nr_y_gauss,nr_x_gauss,np.shape(w_iter)[0]))
    kw_tsl = np.zeros((nr_y_gauss,nr_x_gauss,np.shape(w_iter)[0]))
    force_intgn = np.zeros((np.shape(w_iter)[0], 2))
    force_intgn_dmg = np.zeros((np.shape(w_iter)[0], 2))
    c_all = np.zeros((size, np.shape(w_iter)[0]))

    quasi_NR = True
    NR_kT_update = 3 # After how many iterations should kT be updated
    crisfield_test_fail = False # keeps track if the crisfield test has failed or not and prevents the run from aborting

    # Displacement Incrementation
    for wp in w_iter:
        print(f'------------ wp = {wp:.3f} ---------------\n')

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

            k0 = kT + kCp

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

            # Contains both fint and kC_conn contribution
            kT = assy.calc_kT(c=ci, kC_conn=kC_conn)

            # Setting the max DI to be 0 - needed to calc kcrack
            assy.update_TSL_history(curr_max_dmg_index=np.zeros((nr_y_gauss, nr_x_gauss)))

            k0 = kT + kCp


        epsilon = 1.e-4 # Convergence criteria
        D = k0.diagonal() # For convergence - calc at beginning of load increment

        count = 0 # Tracks number of NR iterations

        crisfield_test_res = np.zeros(1000)

        # Modified Newton Raphson Iteration
        while True:
            dc = solve(k0, -Ri, silent=True)
            c = ci + dc

            # Run it everytime if not quasi_NR or run it selectively if quasi_NR
            if not quasi_NR or (quasi_NR and (count == 0 or count % NR_kT_update == 0)):
                kC_conn = assy.get_kC_conn(c=c)

            fint = np.asarray(assy.calc_fint(c=c, kC_conn=kC_conn))

            # Might need to calc fext here again if it changes per iteration when fext changes when not using kc_conn for SB
            Ri = fint - fext + kCp*c

            print()
            print(f'Ri {np.linalg.norm(Ri):.2e}')
            print(f'fint {np.linalg.norm(fint):.2e}')
            print(f'-fext {np.linalg.norm(fext):.2e}')
            print(f'kCp*c {np.linalg.norm(kCp*c):.2e}')

            crisfield_test = scaling(Ri, D)/max(scaling(fext, D), scaling(fint + kCp*c, D))
            crisfield_test_res[count] = crisfield_test

            print(f'    crisfield {crisfield_test:.4e} ')

            if crisfield_test < epsilon:
                break

            if not quasi_NR or (quasi_NR and (count == 0 or count % NR_kT_update == 0)):
                kT = assy.calc_kT(c=c, kC_conn=kC_conn)
                k0 = kT + kCp
                print(f'        kT {np.max(kT):.2e}')
                print(f'        kCp {np.max(kCp):.2e}')

            # Update for next starting guess
            ci = c.copy()

            count += 1


            # Kills this run but prevents other runs in parallel from being killed due to errors
            if True:
                if crisfield_test >= 1:
                    crisfield_test_fail = True
                    break

            if count > 1000:
                log_file = open(f'{filename}.txt', 'a')
                log_file.write('Unconverged Results !!!!!!!!!!!!!!!!!!!\n')
                log_file.close()
                # raise RuntimeError('NR didnt converged :(')
                print(f'{filename} -- NR didnt converged :(')
                crisfield_test_fail = True
                break

        # ------------------ SAVING VARIABLES (after a completed NR) --------------------
        if True:
            np.save(f'DI_{filename}', dmg_index)
            np.save(f'del_{filename}', del_d)
            np.save(f'k_{filename}', kw_tsl)
            np.save(f'F_{filename}', force_intgn)
            np.save(f'Fdmg_{filename}', force_intgn_dmg)
            np.save(f'c_{filename}', c_all)

        # Saving results incase cluster time is exceeded
        if (time.time() - start_time)/(60*60) > 70: # Max wall time (hrs) on the cluster
            np.save(f'DI_{filename}', dmg_index)
            np.save(f'del_{filename}', del_d)
            np.save(f'k_{filename}', kw_tsl)
            np.save(f'F_{filename}', force_intgn)
            np.save(f'Fdmg_{filename}', force_intgn_dmg)
            np.save(f'c_{filename}', c_all)

        if crisfield_test_fail is True:
            print(f'????????????????????? ABORTED {filename} ?????????????????????????????')
            log_file = open(f'{filename}.txt', 'a')
            log_file.write(f'????????????????????? ABORTED {filename} ?????????????????????????????\n')
            log_file.close()
            sys.stdout.flush()
            break

        # Edit: should be correct as it is
            # Ques: should first update max del d then calc kw_tsl etc ??
        if hasattr(assy, "dmg_index"):
            kw_tsl[:,:,disp_iter_no], dmg_index[:,:,disp_iter_no], del_d[:,:,disp_iter_no], dmg_index_curr  = assy.calc_k_dmg(c=c, pA=p_top, pB=p_bot,
                                                 nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss, tsl_type=tsl_type,
                                                 prev_max_dmg_index=assy.dmg_index, k_i=k_i, tau_o=tau_o)
        else:
            kw_tsl[:,:,disp_iter_no], dmg_index[:,:,disp_iter_no], del_d[:,:,disp_iter_no], dmg_index_curr  = assy.calc_k_dmg(c=c, pA=p_top, pB=p_bot,
                                                 nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss, tsl_type=tsl_type,
                                                 prev_max_dmg_index=None, k_i=k_i, tau_o=tau_o)

        c_all[:,disp_iter_no] = c

        # Update max del_d AFTER a converged NR iteration
        assy.update_TSL_history(curr_max_dmg_index=dmg_index[:,:,disp_iter_no])

        print(f'{disp_iter_no/np.shape(w_iter)[0]*100:.1f}% - {filename} -- wp={wp:.3f} -- max DI {np.max(dmg_index[:,:,disp_iter_no]):.4f}\n')

        # Force - TEMP - CHECK LATER AND EDIT/REMOVE
        if True:
            # force_intgn[disp_iter_no, 0] = wp
            force_intgn[disp_iter_no, 1] = assy.force_out_plane(c, group=None, eval_panel=disp_panel, x_cte_force=None, y_cte_force=None,
                      gridx=100, gridy=50, NLterms=True, nr_x_gauss=128, nr_y_gauss=128)

        # Force by area integral of traction in the damaged region
        if True:
            # force_intgn_dmg[disp_iter_no, 0] = wp
            force_intgn_dmg[disp_iter_no, 1] = assy.force_out_plane_damage(conn=conn, c=c)

        print(f'       Force - Line: {force_intgn[disp_iter_no, 1]:.3f} -- Area: {force_intgn_dmg[disp_iter_no, 1]:.3f}')

        # Calc displ of top and bottom panels at each increment
        res_pan_top = assy.calc_results(c=c, eval_panel=disp_panel, vec='w',
                                nr_x_gauss=200, nr_y_gauss=50)
        max_displ_w = np.max(res_pan_top['w'][-1])

        # Storing actual panel displ instead of disp that 'should' be applied
        force_intgn[disp_iter_no, 0] = max_displ_w
        force_intgn_dmg[disp_iter_no, 0] = max_displ_w

        plt.figure(figsize=(10,7))
        plt.plot(force_intgn[:disp_iter_no+1, 0], force_intgn[:disp_iter_no+1, 1], label = 'Line')
        scaling_F_plot = force_intgn[0,1]/force_intgn_dmg[0,1]
        plt.plot(force_intgn_dmg[:disp_iter_no+1, 0], scaling_F_plot*force_intgn_dmg[:disp_iter_no+1, 1], label='Area Int')
        #plt.plot(FEM[:,0],FEM[:,1], label='FEM')
        plt.ylabel('Force [N]', fontsize=14)
        plt.xlabel('Displacement [mm]', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(f'{filename}')
        plt.grid()
        plt.legend(fontsize=14)
        plt.savefig('test_dcb_damage.png')

        disp_iter_no += 1
        print()

        sys.stdout.flush()

        if np.all(dmg_index == 1):
            print('Cohesive Zone has failed')
            break

    # ------------------ SAVING VARIABLES --------------------
    if True:
        np.save(f'DI_{filename}', dmg_index)
        np.save(f'del_{filename}', del_d)
        np.save(f'k_{filename}', kw_tsl)
        np.save(f'F_{filename}', force_intgn)
        np.save(f'Fdmg_{filename}', force_intgn_dmg)
        np.save(f'c_{filename}', c_all)

    print(f'************************** WAKE UP! ITS DONE! - {filename} ******************************')
    sys.stdout.flush()

    log_file = open(f'{filename}.txt', 'a')
    log_file.write(f'It ONLY took {(time.time() - start_time)/(60*60)} hrs\n')
    log_file.write(f'************************** WAKE UP! ITS DONE! - {filename} ******************************\n')
    log_file.close()

    # ------------------ RESULTS AND POST PROCESSING --------------------

    generate_plots = False

    final_res = None
    # Plotting results
    if True:
        for vec in ['w']:#, 'Mxx']:#, 'Myy', 'Mxy']:#, 'Nxx', 'Nyy']:
            res_bot = assy.calc_results(c=c, group='bot', vec=vec, nr_x_gauss=None, nr_y_gauss=None)
            res_top = assy.calc_results(c=c, group='top', vec=vec, nr_x_gauss=None, nr_y_gauss=None)
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
                max_displ_w = np.max(res_top['w'][-1])
                print(f'Calc W_TOP: {max_displ_w}')
                final_res = np.min(np.array(res_top[vec]))

            if generate_plots:
                # if vec == 'w':
                if True:
                    assy.plot(c=c, group='bot', vec=vec, filename='test_dcb_before_opening_bot_tsl.png', show_boundaries=True,
                                                colorbar=True, res = res_bot, vecmax=vecmax, vecmin=vecmin, display_zero=True)

                    assy.plot(c=c, group='top', vec=vec, filename='test_dcb_before_opening_top_tsl.png', show_boundaries=True,
                                              colorbar=True, res = res_top, vecmax=vecmax, vecmin=vecmin, display_zero=True)

            # Open images
            if generate_plots:
                img_popup('test_dcb_before_opening_top_tsl.png',1, f"{vec} top")
                img_popup('test_dcb_before_opening_bot_tsl.png',2, f"{vec} bot")

    return dmg_index, del_d, kw_tsl, force_intgn, c_all


def test_dcb_damage_prop_no_f_kcrack():
    # p3_70_25_m18_8_ki1e5_tauo67_nx60_ny30_wpts30_G1c112
    dmg_index, del_d, kw_tsl, force_intgn, c_all = dcb_damage_prop_no_f_kcrack(phy_dim=[3,65,25,48],
        nr_terms=[8, 8, 6, 6], name=['test_nokcrack',''], k_i=1e4, tau_o=87, nr_x_gauss=50,
        nr_y_gauss=25, w_iter_info=[10,8], G1c=1.12)


if __name__ == "__main__":
    # Single run tests
    test_dcb_damage_prop_no_f_kcrack()
