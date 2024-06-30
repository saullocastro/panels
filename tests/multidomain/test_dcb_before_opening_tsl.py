import sys
sys.path.append('C:/Users/natha/Documents/GitHub/panels')
# sys.path.append('..\\..')
import os
os.chdir('C:/Users/natha/Documents/GitHub/panels/tests/multidomain')

import numpy as np
from structsolve import solve
from structsolve.sparseutils import finalize_symmetric_matrix

from panels import Shell
from panels.multidomain.connections import calc_ku_kv_kw_point_pd
from panels.multidomain.connections import fkCpd, fkCld_xcte, fkCld_ycte
from panels.plot_shell import plot_shell
from panels.multidomain import MultiDomain

# Open images
from matplotlib import image as img
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes, inset_axes

# Printing with reduced no of points (ease of viewing) - Suppress this to print in scientific notations and restart the kernel
np.set_printoptions(formatter={'float': lambda x: "{0:0.3e}".format(x)})


# import os
# os.chdir('C:/Users/natha/Documents/GitHub/panels/tests/multidomain')

def img_popup(filename, plot_no = None, title = None):
    
    # plot_no = current plot no 
    
    # To open pop up images - Ignore the syntax warning :)
    # %matplotlib qt 
    # For inline images
    # %matplotlib inline
    
    # 
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
      
        
def convergence():
    i = 0 
    final_res = np.zeros((26,2))
    for no_terms in range(5,31):
        final_res[i,0] = test_dcb_vs_fem(2, no_terms)
        print('------------------------------------')
        final_res[i,1] = test_dcb_vs_fem(3, no_terms)
        print('====================================')
        i += 1
    plt.figure()
    plt.plot(range(5,31), final_res[:,0], label = '2 panels' )
    plt.plot(range(5,31), final_res[:,1], label = '3 panels')
    plt.legend()
    plt.grid()
    plt.title('80 Plies - Clamped')
    plt.xlabel('No of terms in shape function')
    plt.ylabel('w [mm]')
    plt.yticks(np.arange(np.min(final_res), np.max(final_res), 0.01))
    # plt.ylim([np.min(final_res), np.max(final_res)])
    plt.show()
    

def monotonicity_check_displ(check_matrix):
    monotonicity = np.all(check_matrix[:, 1:] >= check_matrix[:, :-1], axis=1)
    
    return monotonicity



def test_dcb_bending_pd_tsl():

    '''
        Test function for a DBC with different BCs made with 3 panels per section
    '''    

    # Properties
    E1 = 127560 # MPa
    E2 = 13030. # MPa
    G12 = 6410. # MPa
    nu12 = 0.3
    ply_thickness = 0.127#e-3 # mm

    # Plate dimensions (overall)
    a = 1181.1#e-3 # mm
    b = 746.74#e-3 # mm
    # Dimensions of panel 1 and 2
    a1 = 0.9*a
    a2 = 0.3*a
    
    no_pan = 2 # no of panels per structure
    print('No of panels = ', no_pan)

    #others
    m = 8
    n = 8

    simple_layup = [+45, -45]*20 + [0, 90]*20
    # simple_layup = [0, 0]*20 + [0, 0]*20
    simple_layup += simple_layup[::-1]

    laminaprop = (E1, E2, nu12, G12, G12, G12)
     
    # Top DCB panels
    top1 = Shell(group='top', x0=0, y0=0, a=a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 2:
        top2 = Shell(group='top', x0=a1, y0=0, a=a-a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 3:
        top2 = Shell(group='top', x0=a1, y0=0, a=a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        top3 = Shell(group='top', x0=a1+a2, y0=0, a=a-a1-a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    # Bottom DCB panels
    bot1 = Shell(group='bot', x0=0, y0=0, a=a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 2:
        bot2 = Shell(group='bot', x0=a1, y0=0, a=a-a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 3:
        bot2 = Shell(group='bot', x0=a1, y0=0, a=a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        bot3 = Shell(group='bot', x0=a1+a2, y0=0, a=a-a1-a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    
    # boundary conditions
    
    BC = 'bot_end_fixed'
    # Possible strs: 'bot_fully_fixed', 'bot_end_fixed'
    # DCB with bottom fixed
    if BC == 'bot_fully_fixed':
        raise ValueError('Development incomplete :(')
        top1.x1u = 1 ; top1.x1ur = 1 ; top1.x2u = 1 ; top1.x2ur = 1
        top1.x1v = 1 ; top1.x1vr = 1 ; top1.x2v = 1 ; top1.x2vr = 1 
        top1.x1w = 1 ; top1.x1wr = 1 ; top1.x2w = 1 ; top1.x2wr = 1 
        top1.y1u = 1 ; top1.y1ur = 1 ; top1.y2u = 1 ; top1.y2ur = 1
        top1.y1v = 1 ; top1.y1vr = 1 ; top1.y2v = 1 ; top1.y2vr = 1
        top1.y1w = 1 ; top1.y1wr = 1 ; top1.y2w = 1 ; top1.y2wr = 1
        
        top2.x1u = 1 ; top2.x1ur = 1 ; top2.x2u = 1 ; top2.x2ur = 1
        top2.x1v = 1 ; top2.x1vr = 1 ; top2.x2v = 1 ; top2.x2vr = 1 
        top2.x1w = 1 ; top2.x1wr = 1 ; top2.x2w = 1 ; top2.x2wr = 1  
        top2.y1u = 1 ; top2.y1ur = 1 ; top2.y2u = 1 ; top2.y2ur = 1
        top2.y1v = 1 ; top2.y1vr = 1 ; top2.y2v = 1 ; top2.y2vr = 1
        top2.y1w = 1 ; top2.y1wr = 1 ; top2.y2w = 1 ; top2.y2wr = 1
    
        bot1.x1u = 0 ; bot1.x1ur = 0 ; bot1.x2u = 0 ; bot1.x2ur = 0
        bot1.x1v = 0 ; bot1.x1vr = 0 ; bot1.x2v = 0 ; bot1.x2vr = 0 
        bot1.x1w = 0 ; bot1.x1wr = 0 ; bot1.x2w = 0 ; bot1.x2wr = 0 
        bot1.y1u = 0 ; bot1.y1ur = 0 ; bot1.y2u = 0 ; bot1.y2ur = 0
        bot1.y1v = 0 ; bot1.y1vr = 0 ; bot1.y2v = 0 ; bot1.y2vr = 0
        bot1.y1w = 0 ; bot1.y1wr = 0 ; bot1.y2w = 0 ; bot1.y2wr = 0
        
        bot2.x1u = 0 ; bot2.x1ur = 0 ; bot2.x2u = 0 ; bot2.x2ur = 0
        bot2.x1v = 0 ; bot2.x1vr = 0 ; bot2.x2v = 0 ; bot2.x2vr = 0 
        bot2.x1w = 0 ; bot2.x1wr = 0 ; bot2.x2w = 0 ; bot2.x2wr = 0 
        bot2.y1u = 0 ; bot2.y1ur = 0 ; bot2.y2u = 0 ; bot2.y2ur = 0
        bot2.y1v = 0 ; bot2.y1vr = 0 ; bot2.y2v = 0 ; bot2.y2vr = 0
        bot2.y1w = 0 ; bot2.y1wr = 0 ; bot2.y2w = 0 ; bot2.y2wr = 0
    
    # DCB with only the lower extreme end fixed at the tip. Rest free
    if BC == 'bot_end_fixed':
        top1.x1u = 1 ; top1.x1ur = 1 ; top1.x2u = 1 ; top1.x2ur = 1
        top1.x1v = 1 ; top1.x1vr = 1 ; top1.x2v = 1 ; top1.x2vr = 1 
        top1.x1w = 1 ; top1.x1wr = 1 ; top1.x2w = 1 ; top1.x2wr = 1 
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
    
        bot1.x1u = 1 ; bot1.x1ur = 1 ; bot1.x2u = 1 ; bot1.x2ur = 1
        bot1.x1v = 1 ; bot1.x1vr = 1 ; bot1.x2v = 1 ; bot1.x2vr = 1 
        bot1.x1w = 1 ; bot1.x1wr = 1 ; bot1.x2w = 1 ; bot1.x2wr = 1 
        bot1.y1u = 1 ; bot1.y1ur = 1 ; bot1.y2u = 1 ; bot1.y2ur = 1
        bot1.y1v = 1 ; bot1.y1vr = 1 ; bot1.y2v = 1 ; bot1.y2vr = 1
        bot1.y1w = 1 ; bot1.y1wr = 1 ; bot1.y2w = 1 ; bot1.y2wr = 1
        
        if no_pan == 2:
            bot2.x1u = 1 ; bot2.x1ur = 1 ; bot2.x2u = 0 ; bot2.x2ur = 0 # only right extreme of plate 3 with line ll to x is fixed
            bot2.x1v = 1 ; bot2.x1vr = 1 ; bot2.x2v = 0 ; bot2.x2vr = 0 
            bot2.x1w = 1 ; bot2.x1wr = 1 ; bot2.x2w = 0 ; bot2.x2wr = 0 
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
            
            bot3.x1u = 1 ; bot3.x1ur = 1 ; bot3.x2u = 0 ; bot3.x2ur = 0 # only right extreme of plate 3 with line ll to x is fixed
            bot3.x1v = 1 ; bot3.x1vr = 1 ; bot3.x2v = 0 ; bot3.x2vr = 0 
            bot3.x1w = 1 ; bot3.x1wr = 1 ; bot3.x2w = 0 ; bot3.x2wr = 0 
            bot3.y1u = 1 ; bot3.y1ur = 1 ; bot3.y2u = 1 ; bot3.y2ur = 1
            bot3.y1v = 1 ; bot3.y1vr = 1 ; bot3.y2v = 1 ; bot3.y2vr = 1
            bot3.y1w = 1 ; bot3.y1wr = 1 ; bot3.y2w = 1 ; bot3.y2wr = 1

    # All connections - list of dict
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
          dict(p1=top1, p2=bot1, func='SB'), #'_TSL', tsl_type = 'linear'), 
           # dict(p1=top2, p2=bot2, func='SB_TSL', tsl_type = 'linear')
        ]
    
    # This determines the positions of each panel's (sub)matrix in the global matrix when made a MD obj below
    # So changing this changes the placements i.e. starting row and col of each
    if no_pan == 2:
        panels = [top1, top2, bot1, bot2]
    if no_pan == 3:
        panels = [top1, top2, top3, bot1, bot2, bot3]

    assy = MultiDomain(panels) # assy is now an object of the MultiDomain class
    # Here the panels (shell objs) are modified -- their starting positions in the global matrix is assigned etc

    # print('bot', bot1.row_start, bot1.col_start, bot2.row_start, bot2.col_start, bot3.col_start, bot3.row_start)
    # print('top', top1.row_start, top1.col_start, top2.row_start, top2.col_start, top3.col_start, top3.row_start)
    k0 = assy.calc_kC(conn)
    
    size = k0.shape[0]

    # Panel at which the disp is applied
    if no_pan == 2:
        disp_panel = top2
    if no_pan == 3:
        disp_panel = top3

    if True:
        ku, kv, kw = calc_ku_kv_kw_point_pd(disp_panel)
        # kw = 1e4*top1.a*top1.b
    # Prescribed Displacements
    if True:
        disp_type = 'line_xcte' # change based on what's being applied
        
        if disp_type == 'point':
            # Penalty Stiffness
            # Disp in z, so only kw is non zero. ku and kv are zero
            kCp = fkCpd(0., 0., kw, disp_panel, disp_panel.a, disp_panel.b/2, size, disp_panel.row_start, disp_panel.col_start)
            # Point load (added to shell obj)
            wp = 5#e-3 # mm
            disp_panel.add_point_pd(disp_panel.a, disp_panel.b/2, 0., 0., 0., 0., kw, wp)
        if disp_type == 'line_xcte':
            kCp = fkCld_xcte(0., 0., kw, disp_panel, disp_panel.a, size, disp_panel.row_start, disp_panel.col_start)
            disp_panel.add_distr_pd_fixed_x(disp_panel.a, None, None, kw,
                                      funcu=None, funcv=None, funcw = lambda y: 5) #*y/top2.b, cte=True)
        if disp_type == 'line_ycte':
            kCp = fkCld_ycte(0., 0., kw, disp_panel, disp_panel.b, size, disp_panel.row_start, disp_panel.col_start)
            disp_panel.add_distr_pd_fixed_y(disp_panel.b, None, None, kw,
                                      funcu=None, funcv=None, funcw = lambda x: 1) #*x/top2.a, cte=True)
        
        kCp = finalize_symmetric_matrix(kCp)
        
    # Tangent (complete) stiffness matrix
    k0 = k0 + kCp
        
    fext = disp_panel.calc_fext(size=size, col0=disp_panel.col_start)
    c0 = solve(k0, fext)
    # print('shape c0: ', np.shape(c0))
    
    # Plotting results
    if True:
        res_bot = assy.calc_results(c=c0, group='bot', vec='w')
        res_top = assy.calc_results(c=c0, group='top', vec='w')
        vecmin = min(np.min(np.array(res_top['w'])), np.min(np.array(res_bot['w'])))
        vecmax = max(np.max(np.array(res_top['w'])), np.max(np.array(res_bot['w'])))
        
        assy.plot(c=c0, group='bot', vec='w', filename='test_dcb_before_opening_bot_tsl.png', show_boundaries=True,
                                  colorbar=True, res = res_bot, vecmin=vecmin, vecmax=vecmax, display_zero=True)
        
        assy.plot(c=c0, group='top', vec='w', filename='test_dcb_before_opening_top_tsl.png', show_boundaries=True,
                                  colorbar=True, res = res_top, vecmin=vecmin, vecmax=vecmax, display_zero=True)
    
    # Open images
    if True:
        img_popup('test_dcb_before_opening_top_tsl.png')
        img_popup('test_dcb_before_opening_bot_tsl.png')



def test_junk():

    '''
        Dummy function to test
    '''    
    print('DEBUGGING FTN')

    # Properties
    E1 = 127560 # MPa
    E2 = 13030. # MPa
    G12 = 6410. # MPa
    nu12 = 0.3
    ply_thickness = 0.127#e-3 # mm

    # Plate dimensions (overall)
    a = 1181.1#e-3 # mm
    b = 746.74#e-3 # mm
    # Dimensions of panel 1 and 2
    a1 = 0.5*a
    a2 = 0.3*a
    
    no_pan = 2 # no of panels per structure
    print('No of panels = ', no_pan)

    #others
    m = 8
    n = 8

    simple_layup = [+45, -45]*20 + [0, 90]*20
    # simple_layup = [0, 0]*20 + [0, 0]*20
    simple_layup += simple_layup[::-1]

    laminaprop = (E1, E2, nu12, G12, G12, G12)
     
    # Top DCB panels
    top1 = Shell(group='top', x0=0, y0=0, a=a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 2:
        top2 = Shell(group='top', x0=a1, y0=0, a=a-a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 3:
        top2 = Shell(group='top', x0=a1, y0=0, a=a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        top3 = Shell(group='top', x0=a1+a2, y0=0, a=a-a1-a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    # Bottom DCB panels
    bot1 = Shell(group='bot', x0=0, y0=0, a=a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 2:
        bot2 = Shell(group='bot', x0=a1, y0=0, a=a-a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 3:
        bot2 = Shell(group='bot', x0=a1, y0=0, a=a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        bot3 = Shell(group='bot', x0=a1+a2, y0=0, a=a-a1-a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    
    # boundary conditions
    
    BC = 'bot_end_fixed'
    # Possible strs: 'bot_fully_fixed', 'bot_end_fixed'

    # DCB with only the lower extreme end fixed at the tip. Rest free
    if BC == 'bot_end_fixed':
        top1.x1u = 1 ; top1.x1ur = 1 ; top1.x2u = 1 ; top1.x2ur = 1
        top1.x1v = 1 ; top1.x1vr = 1 ; top1.x2v = 1 ; top1.x2vr = 1 
        top1.x1w = 1 ; top1.x1wr = 1 ; top1.x2w = 1 ; top1.x2wr = 1 
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
    
        # bot1.x1u = 0 ; bot1.x1ur = 0 ; bot1.x2u = 1 ; bot1.x2ur = 1
        # bot1.x1v = 0 ; bot1.x1vr = 0 ; bot1.x2v = 1 ; bot1.x2vr = 1
        # bot1.x1w = 0 ; bot1.x1wr = 0 ; bot1.x2w = 1 ; bot1.x2wr = 1
        # bot1.y1u = 1 ; bot1.y1ur = 1 ; bot1.y2u = 1 ; bot1.y2ur = 1
        # bot1.y1v = 1 ; bot1.y1vr = 1 ; bot1.y2v = 1 ; bot1.y2vr = 1
        # bot1.y1w = 1 ; bot1.y1wr = 1 ; bot1.y2w = 1 ; bot1.y2wr = 1
        
        bot1.x1u = 1 ; bot1.x1ur = 1 ; bot1.x2u = 1 ; bot1.x2ur = 1
        bot1.x1v = 1 ; bot1.x1vr = 1 ; bot1.x2v = 1 ; bot1.x2vr = 1
        bot1.x1w = 1 ; bot1.x1wr = 1 ; bot1.x2w = 1 ; bot1.x2wr = 1
        bot1.y1u = 1 ; bot1.y1ur = 1 ; bot1.y2u = 1 ; bot1.y2ur = 1
        bot1.y1v = 1 ; bot1.y1vr = 1 ; bot1.y2v = 1 ; bot1.y2vr = 1
        bot1.y1w = 1 ; bot1.y1wr = 1 ; bot1.y2w = 1 ; bot1.y2wr = 1
        
        if no_pan == 2:
            bot2.x1u = 1 ; bot2.x1ur = 1 ; bot2.x2u = 0 ; bot2.x2ur = 0 
            bot2.x1v = 1 ; bot2.x1vr = 1 ; bot2.x2v = 0 ; bot2.x2vr = 0 
            bot2.x1w = 1 ; bot2.x1wr = 1 ; bot2.x2w = 0 ; bot2.x2wr = 0
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
            
            bot3.x1u = 1 ; bot3.x1ur = 1 ; bot3.x2u = 0 ; bot3.x2ur = 0 
            bot3.x1v = 1 ; bot3.x1vr = 1 ; bot3.x2v = 0 ; bot3.x2vr = 0 
            bot3.x1w = 1 ; bot3.x1wr = 1 ; bot3.x2w = 0 ; bot3.x2wr = 0 
            bot3.y1u = 1 ; bot3.y1ur = 1 ; bot3.y2u = 1 ; bot3.y2ur = 1
            bot3.y1v = 1 ; bot3.y1vr = 1 ; bot3.y2v = 1 ; bot3.y2vr = 1
            bot3.y1w = 1 ; bot3.y1wr = 1 ; bot3.y2w = 1 ; bot3.y2wr = 1

    # All connections - list of dict
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
           dict(p1=top1, p2=bot1, func='SB'), #'_TSL', tsl_type = 'linear'), 
           # dict(p1=top2, p2=bot2, func='SB_TSL', tsl_type = 'linear')
        ]
    
    # This determines the positions of each panel's (sub)matrix in the global matrix when made a MD obj below
    # So changing this changes the placements i.e. starting row and col of each
    if no_pan == 2:
        panels = [bot1, bot2, top1, top2]
    if no_pan == 3:
        panels = [bot1, bot2, bot3, top1, top2, top3]

    assy = MultiDomain(panels) # assy is now an object of the MultiDomain class
    # Here the panels (shell objs) are modified -- their starting positions in the global matrix is assigned etc

    # print('bot', bot1.row_start, bot1.col_start, bot2.row_start, bot2.col_start, bot3.col_start, bot3.row_start)
    # print('top', top1.row_start, top1.col_start, top2.row_start, top2.col_start, top3.col_start, top3.row_start)
    k0 = assy.calc_kC(conn)
    
    size = k0.shape[0]

    # Panel at which the disp is applied
    if no_pan == 2:
        disp_panel = top2
    if no_pan == 3:
        disp_panel = top3

    if True:
        ku, kv, kw = calc_ku_kv_kw_point_pd(disp_panel)
        # kw = 1e4*top1.a*top1.b
    # Prescribed Displacements
    if True:
        # print('called first')
        disp_type = 'line_xcte' # change based on what's being applied
        
        if disp_type == 'point':
            # Penalty Stiffness
            # Disp in z, so only kw is non zero. ku and kv are zero
            kCp = fkCpd(0., 0., kw, disp_panel, disp_panel.a, disp_panel.b/2, size, disp_panel.row_start, disp_panel.col_start)
            # Point load (added to shell obj)
            wp = 5#e-3 # mm
            disp_panel.add_point_pd(disp_panel.a, disp_panel.b/2, 0., 0., 0., 0., kw, wp)
        if disp_type == 'line_xcte':
            kCp = fkCld_xcte(0., 0., kw, disp_panel, disp_panel.a, size, disp_panel.row_start, disp_panel.col_start)
            disp_panel.add_distr_pd_fixed_x(disp_panel.a, None, None, kw,
                                      funcu=None, funcv=None, funcw = lambda y: 5) #*y/top2.b, cte=True)
        if disp_type == 'line_ycte':
            kCp = fkCld_ycte(0., 0., kw, disp_panel, disp_panel.b, size, disp_panel.row_start, disp_panel.col_start)
            disp_panel.add_distr_pd_fixed_y(disp_panel.b, None, None, kw,
                                      funcu=None, funcv=None, funcw = lambda x: 1) #*x/top2.a, cte=True)
    kCp = finalize_symmetric_matrix(kCp)
    fext = disp_panel.calc_fext(size=size, col0=disp_panel.col_start)
        
    # Tangent (complete) stiffness matrix
    k0 = k0 + kCp

    # For 2 loads
    # if True:        
    #     # Panel at which the disp is applied
    #     if no_pan == 2:
    #         disp_panel = bot2
    #     if no_pan == 3:
    #         disp_panel = bot3
    
    #     if True:
    #         ku, kv, kw = calc_ku_kv_kw_point_pd(disp_panel)
    #         # kw = 1e4*top1.a*top1.b
    #     # Prescribed Displacements
    #     if True:
    #         print('called second')
    #         disp_type = 'line_xcte' # change based on what's being applied
            
    #         if disp_type == 'point':
    #             # Penalty Stiffness
    #             # Disp in z, so only kw is non zero. ku and kv are zero
    #             kCp = fkCpd(0., 0., kw, disp_panel, disp_panel.a, disp_panel.b/2, size, disp_panel.row_start, disp_panel.col_start)
    #             # Point load (added to shell obj)
    #             wp = 5#e-3 # mm
    #             disp_panel.add_point_pd(disp_panel.a, disp_panel.b/2, 0., 0., 0., 0., kw, wp)
    #         if disp_type == 'line_xcte':
    #             kCp = fkCld_xcte(0., 0., kw, disp_panel, disp_panel.a, size, disp_panel.row_start, disp_panel.col_start)
    #             disp_panel.add_distr_pd_fixed_x(disp_panel.a, None, None, kw,
    #                                       funcu=None, funcv=None, funcw = lambda y: -10) #*y/top2.b, cte=True)
    #         if disp_type == 'line_ycte':
    #             kCp = fkCld_ycte(0., 0., kw, disp_panel, disp_panel.b, size, disp_panel.row_start, disp_panel.col_start)
    #             disp_panel.add_distr_pd_fixed_y(disp_panel.b, None, None, kw,
    #                                       funcu=None, funcv=None, funcw = lambda x: 1) #*x/top2.a, cte=True)
                
    #         fext = fext + disp_panel.calc_fext(size=size, col0=disp_panel.col_start)
            
    #     kCp = finalize_symmetric_matrix(kCp)
            
    #     # Tangent (complete) stiffness matrix
    #     k0 = k0 + kCp
        
    # fext = disp_panel.calc_fext(size=size, col0=disp_panel.col_start)
    c0 = solve(k0, fext)
    # print('shape c0: ', np.shape(c0))
    
    # Plotting results
    if True:
        vec = 'Nxx'
        res_bot = assy.calc_results(c=c0, group='bot', vec=vec, no_x_gauss=50, no_y_gauss=50)
        # print(type(res_bot))
        # print(np.shape(res_bot['x']))
        # print(res_bot['x'][0])
        # print(res_bot['y'][0])
        # print(res_bot['x'][0][:,1])
        # print(np.shape(res_bot['x'][0][:,1])[0])
        # [temp_row, temp_col] = np.where( np.isclose(res_bot['x'][0], 136.2784744))
        # print(temp_row)
        # print(temp_col)
        
        res_top = assy.calc_results(c=c0, group='top', vec=vec, no_x_gauss=50, no_y_gauss=50)
        vecmin = min(np.min(np.array(res_top[vec])), np.min(np.array(res_bot[vec])))
        vecmax = max(np.max(np.array(res_top[vec])), np.max(np.array(res_bot[vec])))
        print(vecmin, vecmax)
        
        assy.plot(c=c0, group='bot', vec=vec, filename='test_dcb_before_opening_bot_tsl.png', show_boundaries=True,
                                    colorbar=True, res = res_bot, vecmax=vecmax, vecmin=vecmin, display_zero=True)
        
        assy.plot(c=c0, group='top', vec=vec, filename='test_dcb_before_opening_top_tsl.png', show_boundaries=True,
                                  colorbar=True, res = res_top, vecmax=vecmax, vecmin=vecmin, display_zero=True)
    
    # Test for force
    if True:
        vec = 'Fxx'
        res_bot = assy.calc_results(c=c0, group='bot', vec=vec, no_x_gauss=5, no_y_gauss=6,
                                    cte_panel_force=bot2, y_cte_force=bot2.b)
        print(res_bot)
        # print(type(res_bot))
        # print(np.shape(res_bot['x']))
        # print(res_bot['x'][0])
        # print(res_bot['y'][0])
        # print(res_bot['x'][0][:,1])
        # print(np.shape(res_bot['x'][0][:,1])[0])
        # [temp_row, temp_col] = np.where( np.isclose(res_bot['x'][0], 136.2784744))
        # print(temp_row)
        # print(temp_col)
        
        # res_top = assy.calc_results(c=c0, group='top', vec=vec, no_x_gauss=5, no_y_gauss=5)
        # vecmin = min(np.min(np.array(res_top[vec])), np.min(np.array(res_bot[vec])))
        # vecmax = max(np.max(np.array(res_top[vec])), np.max(np.array(res_bot[vec])))
        # print(vecmin, vecmax)
        # [vecmin, vecmax] = None
        
        # assy.plot(c=c0, group='bot', vec=vec, filename='test_dcb_before_opening_bot_tsl.png', show_boundaries=True,
        #                             colorbar=True, res = res_bot, vecmax=vecmax, vecmin=vecmin, display_zero=True)
        
        # assy.plot(c=c0, group='top', vec=vec, filename='test_dcb_before_opening_top_tsl.png', show_boundaries=True,
        #                           colorbar=True, res = res_top, vecmax=vecmax, vecmin=vecmin, display_zero=True)
    
    # Open images
    if True:
        img_popup('test_dcb_before_opening_top_tsl.png')
        img_popup('test_dcb_before_opening_bot_tsl.png')




def test_dcb_vs_fem(no_pan, no_terms, plies, disp_mag, a2, no_y_gauss, grid_x, kw=None):

    '''
        Full DCB
        Does not have TSL - Just testing bending for now 
        
        Testing it with FEM
        Values are taken from 'Characterization and analysis of the interlaminar behavior of thermoplastic
        composites considering fiber bridging and R-curve effects'
            https://doi.org/10.1016/j.compositesa.2022.107101
            
        All units in MPa, N, mm
    '''    
    
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
    a1 = 52 #0.5*a
    a2 = a2 #0.3*a
    a3 = 4
    
    # no_pan = 3 # no of panels per structure
    # print('No of panels = ', no_pan)

    #others
    m = no_terms
    n = no_terms
    
    m_conn = 10
    n_conn = 10

    # simple_layup = [+45, -45]*plies + [0, 90]*plies
    # simple_layup = [0, 0]*10 + [0, 0]*10
    simple_layup = [0]*15
    # simple_layup += simple_layup[::-1]
    # simple_layup += simple_layup[::-1]
    # print('plies ',np.shape(simple_layup)[0])

    laminaprop = (E1, E2, nu12, G12, G12, G12)
     
    # Top DCB panels
    top1 = Shell(group='top', x0=0, y0=0, a=a1, b=b, m=m_conn, n=n_conn, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
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
    bot1 = Shell(group='bot', x0=0, y0=0, a=a1, b=b, m=m_conn, n=n_conn, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
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
    
        # bot1.x1u = 0 ; bot1.x1ur = 0 ; bot1.x2u = 1 ; bot1.x2ur = 1
        # bot1.x1v = 0 ; bot1.x1vr = 0 ; bot1.x2v = 1 ; bot1.x2vr = 1
        # bot1.x1w = 0 ; bot1.x1wr = 0 ; bot1.x2w = 1 ; bot1.x2wr = 1
        # bot1.y1u = 1 ; bot1.y1ur = 1 ; bot1.y2u = 1 ; bot1.y2ur = 1
        # bot1.y1v = 1 ; bot1.y1vr = 1 ; bot1.y2v = 1 ; bot1.y2vr = 1
        # bot1.y1w = 1 ; bot1.y1wr = 1 ; bot1.y2w = 1 ; bot1.y2wr = 1
        
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

    # All connections - list of dict
    if no_pan == 2:
        conn = [
         # skin-skin
         dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
         dict(p1=bot1, p2=bot2, func='SSxcte', xcte1=bot1.a, xcte2=0),
         dict(p1=top1, p2=bot1, func='SB'),  
            # dict(p1=top2, p2=bot2, func='SB_TSL', tsl_type = 'linear')
        ]
    if no_pan == 3:
        conn = [
         # skin-skin
           dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
           dict(p1=top2, p2=top3, func='SSxcte', xcte1=top2.a, xcte2=0),
           dict(p1=bot1, p2=bot2, func='SSxcte', xcte1=bot1.a, xcte2=0),
           dict(p1=bot2, p2=bot3, func='SSxcte', xcte1=bot2.a, xcte2=0),
           dict(p1=top1, p2=bot1, func='SB'), 
            # dict(p1=top2, p2=bot2, func='SB_TSL', tsl_type = 'linear')
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
           dict(p1=top1, p2=bot1, func='SB'), 
            # dict(p1=top2, p2=bot2, func='SB_TSL', tsl_type = 'linear')
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

    k0 = assy.calc_kC(conn)
    
    size = k0.shape[0]
    
    # Prescribed Displacements
    if True:
        ######## THIS SHOULD BE CHANGED LATER PER DISP TYPE ###########################################
        ku, kv, kw = calc_ku_kv_kw_point_pd(disp_panel)
        
        # print('called first')
        disp_type = 'line_xcte' # change based on what's being applied
        # print('disp_type : ', disp_type)
        # disp_mag = 3.0
        print(f'applied disp = {disp_mag}')
        
        if disp_type == 'point':
            # Penalty Stiffness
            # Disp in z, so only kw is non zero. ku and kv are zero
            kCp = fkCpd(0., 0., kw, disp_panel, disp_panel.a, disp_panel.b/2, size, disp_panel.row_start, disp_panel.col_start)
            # Point load (added to shell obj)
            wp = disp_mag # mm
            disp_panel.add_point_pd(disp_panel.a, disp_panel.b/2, 0., 0., 0., 0., kw, wp)
        if disp_type == 'line_xcte':
            kCp = fkCld_xcte(0., 0., kw, disp_panel, disp_panel.a, size, disp_panel.row_start, disp_panel.col_start)
            disp_panel.add_distr_pd_fixed_x(disp_panel.a, None, None, kw,
                                      funcu=None, funcv=None, funcw = lambda y: disp_mag) #*y/top2.b, cte=True)
        if disp_type == 'line_ycte':
            kCp = fkCld_ycte(0., 0., kw, disp_panel, disp_panel.b, size, disp_panel.row_start, disp_panel.col_start)
            disp_panel.add_distr_pd_fixed_y(disp_panel.b, None, None, kw,
                                      funcu=None, funcv=None, funcw = lambda x: 1) #*x/top2.a, cte=True)
        kCp = finalize_symmetric_matrix(kCp)
    
    # Applied Load
    else:
        kCp = np.zeros_like(k0)
        # here disp_mag is mag of applied force but to make it easier and not complicate the inputs, its just used here temporarily 
        disp_panel.add_distr_load_fixed_x(disp_panel.a, funcx=None, funcy=None, funcz=lambda y: disp_mag/disp_panel.b)
        print(f'applied load = {disp_mag}')

    # fext = disp_panel.calc_fext(size=size, col0=disp_panel.col_start)
    fext = assy.calc_fext()
        
    # Tangent (complete) stiffness matrix
    k0 = k0 + kCp

    # fext = disp_panel.calc_fext(size=size, col0=disp_panel.col_start)
    c0 = solve(k0, fext, silent=True, **dict())

        

    # Testing Mx at the tip
    if False:
        stress = assy.stress(c0, group=None, gridx=100, gridy=100, NLterms=True, no_x_gauss=None, no_y_gauss=None,
                   eval_panel=top1, x_cte_force=None, y_cte_force=None)
        Mxx_end = stress["Mxx"][0][:,-1]
        dy_Mxx = np.linspace(0,disp_panel.b, 100)
        # print(np.shape(final_res))
    
    # TESTING Qx, Qy 
    if True:
        force = 0
        for panels_i in [disp_panel]:
            force = assy.force_out_plane(c0, group=None, eval_panel=panels_i, x_cte_force=panels_i.a, y_cte_force=None,
                      gridx=grid_x, gridy=None, NLterms=True, no_x_gauss=300, no_y_gauss=no_y_gauss)
        # print(f'Force {force}')
    
    generate_plots = False
    
    # Plotting results
    if True:
        for vec in ['w']:#, 'w', 'Myy', 'Mxy']:#, 'Nxx', 'Nyy']:
            res_bot = assy.calc_results(c=c0, group='bot', vec=vec, gridx=50, gridy=50)
            res_top = assy.calc_results(c=c0, group='top', vec=vec, gridx=50, gridy=50)
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
                print(f'Global TOP {vec} :: {np.min(np.array(res_top[vec])):.4f}  {np.max(np.array(res_top[vec])):.4f}')
                print(f'Global BOT {vec} :: {np.min(np.array(res_bot[vec])):.4f}  {np.max(np.array(res_bot[vec])):.4f}')
                # print(res_bot[vec][1][:,-1]) # disp at the tip
                final_res = res_top
            
            if generate_plots:
                # if vec == 'w':
                if True:
                    assy.plot(c=c0, group='bot', vec=vec, filename='test_dcb_before_opening_bot_tsl.png', show_boundaries=True,
                                                colorbar=True, res = res_bot, vecmax=vecmax, vecmin=vecmin, display_zero=True, 
                                                flip_plot=False)
                    
                    assy.plot(c=c0, group='top', vec=vec, filename='test_dcb_before_opening_top_tsl.png', show_boundaries=True,
                                              colorbar=True, res = res_top, vecmax=vecmax, vecmin=vecmin, display_zero=True,
                                              flip_plot=False)
            
            
            # Open images
            if generate_plots:
                img_popup('test_dcb_before_opening_top_tsl.png',1, f"{vec} top")
                img_popup('test_dcb_before_opening_bot_tsl.png',2, f"{vec} bot")
                plt.show()
        
    # Calcuate separation
    if False:
        res_pan_top = assy.calc_results(c=c0, eval_panel=top1, vec='w', 
                                no_x_gauss=200, no_y_gauss=50)
        res_pan_bot = assy.calc_results(c=c0, eval_panel=bot1, vec='w', 
                                no_x_gauss=200, no_y_gauss=50)
        del_d = assy.calc_separation(res_pan_top, res_pan_bot)
        
        monoton_top = monotonicity_check_displ(res_pan_top['w'][0])
        monoton_bot = monotonicity_check_displ(res_pan_bot['w'][0])
        
        if np.all(monoton_top):
            print('Top displ is monotonic')
            top_mon = 'Monotonic'
        else:
            top_mon = 'NOT monotonic'
        if np.all(monoton_bot):
            print('Bot displ is monotonic')
            bot_mon = 'Monotonic'
        else:
            bot_mon = 'NOT monotonic'
        
        
        # Plotting separation and displacements
        vmin=min(np.min(res_pan_top['w'][0]), np.min(res_pan_bot['w'][0]))
        vmax=max(np.max(res_pan_top['w'][0]), np.max(res_pan_bot['w'][0]))
        print(np.min(res_pan_top['w'][0]), np.max(res_pan_top['w'][0]))
        print(np.min(res_pan_bot['w'][0]), np.max(res_pan_bot['w'][0]))
        levels = np.linspace(vmin, vmax, 20)
        plt.figure(figsize=(8,14))
        plt.subplot(3,1,1)
        plt.contourf(del_d, cmap='jet')
        plt.gca().set_title(f'Full DCB - Terms: {m}, {m_conn} \n  Separation between plates in contact [mm]')
        plt.xlabel('x direction')
        plt.ylabel('y direction')
        plt.colorbar()
        plt.subplot(3,1,2)
        plt.contourf(res_pan_top['w'][0], cmap='jet', levels=levels)
        plt.gca().set_title(f'Top displ [mm] - {top_mon}')
        plt.xlabel('x direction')
        plt.ylabel('y direction')
        plt.colorbar()
        plt.subplot(3,1,3)
        plt.contourf(res_pan_bot['w'][0], cmap='jet', levels=levels)
        plt.gca().set_title(f'Bot displ [mm] - {bot_mon}')
        plt.xlabel('x direction')
        plt.ylabel('y direction')
        plt.colorbar()
        plt.show()
    else:
        monoton_top = None
        monoton_bot = None
        
        

    
    final_res = None

    # return res_pan_top, res_pan_bot
    return force



def single_panel_bending(no_pan, no_terms, plies, disp_mag, a1, no_y_gauss, grid_x):
    
        
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
    a1 = a1 #0.5*a
    # a2 = a2

    #others
    m = no_terms
    n = no_terms

    simple_layup = [0]*15

    laminaprop = (E1, E2, nu12, G12, G12, G12)
     
    # Top DCB panels
    top1 = Shell(group='top', x0=0, y0=0, a=a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 2:
        top2 = Shell(group='top', x0=a1, y0=0, a=a-a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 3:
        top2 = Shell(group='top', x0=a1, y0=0, a=a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        top3 = Shell(group='top', x0=a1+a2, y0=0, a=a-a1-a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)

    # boundary conditions
    
    BC = 'bot_end_fixed'
    # Possible strs: 'bot_fully_fixed', 'bot_end_fixed'
    
    clamped = True
    ss = False
    
    if clamped:
        top_r = 0
        top_t = 0
    if ss:
        top_r = 1
        top_t = 0

    # DCB with only the lower extreme end fixed at the tip. Rest free
    if BC == 'bot_end_fixed':
        top1.x1u = top_t ; top1.x1ur = top_r ; top1.x2u = 1 ; top1.x2ur = 1
        top1.x1v = top_t ; top1.x1vr = top_r ; top1.x2v = 1 ; top1.x2vr = 1 
        top1.x1w = top_t ; top1.x1wr = top_r ; top1.x2w = 1 ; top1.x2wr = 1 
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
            raise ValueError('Check if its correct')
            top3.x1u = 1 ; top3.x1ur = 1 ; top3.x2u = 1 ; top3.x2ur = 1
            top3.x1v = 1 ; top3.x1vr = 1 ; top3.x2v = 1 ; top3.x2vr = 1 
            top3.x1w = 1 ; top3.x1wr = 1 ; top3.x2w = 1 ; top3.x2wr = 1  
            top3.y1u = 1 ; top3.y1ur = 1 ; top3.y2u = 1 ; top3.y2ur = 1
            top3.y1v = 1 ; top3.y1vr = 1 ; top3.y2v = 1 ; top3.y2vr = 1
            top3.y1w = 1 ; top3.y1wr = 1 ; top3.y2w = 1 ; top3.y2wr = 1
    
    # All connections - list of dict
    if no_pan == 2:
        conn = [
         # skin-skin
         dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
        ]
    if no_pan == 3:
        conn = [
         # skin-skin
           dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
           dict(p1=top2, p2=top3, func='SSxcte', xcte1=top2.a, xcte2=0),
           dict(p1=bot1, p2=bot2, func='SSxcte', xcte1=bot1.a, xcte2=0),
           dict(p1=bot2, p2=bot3, func='SSxcte', xcte1=bot2.a, xcte2=0),
           dict(p1=top1, p2=bot1, func='SB'), #'_TSL', tsl_type = 'linear'), 
            # dict(p1=top2, p2=bot2, func='SB_TSL', tsl_type = 'linear')
        ]
    
    # This determines the positions of each panel's (sub)matrix in the global matrix when made a MD obj below
    # So changing this changes the placements i.e. starting row and col of each
    if no_pan == 2:
        panels = [top1, top2]
    if no_pan == 3:
        panels = [bot1, bot2, bot3, top1, top2, top3]

    assy = MultiDomain(panels=panels, conn=conn) # assy is now an object of the MultiDomain class
    # Here the panels (shell objs) are modified -- their starting positions in the global matrix is assigned etc

    # Panel at which the disp is applied
    if no_pan == 2:
        disp_panel = top2
    if no_pan == 3:
        disp_panel = top3

    k0 = assy.calc_kC(conn)
    
    size = k0.shape[0]
    
    # Prescribed Displacements
    if False:
        ku, kv, kw = calc_ku_kv_kw_point_pd(disp_panel)
        
        disp_type = 'point' # change based on what's being applied

        if disp_type == 'point':
            # Penalty Stiffness
            # Disp in z, so only kw is non zero. ku and kv are zero
            kCp = fkCpd(0., 0., kw, disp_panel, disp_panel.a, disp_panel.b/2, size, disp_panel.row_start, disp_panel.col_start)
            # Point load (added to shell obj)
            wp = disp_mag # mm
            disp_panel.add_point_pd(disp_panel.a, disp_panel.b/2, 0., 0., 0., 0., kw, wp)
        if disp_type == 'line_xcte':
            kCp = fkCld_xcte(0., 0., kw, disp_panel, disp_panel.a, size, disp_panel.row_start, disp_panel.col_start)
            disp_panel.add_distr_pd_fixed_x(disp_panel.a, None, None, kw,
                                      funcu=None, funcv=None, funcw = lambda y: disp_mag) #*y/top2.b, cte=True)
        if disp_type == 'line_ycte':
            kCp = fkCld_ycte(0., 0., kw, disp_panel, disp_panel.b, size, disp_panel.row_start, disp_panel.col_start)
            disp_panel.add_distr_pd_fixed_y(disp_panel.b, None, None, kw,
                                      funcu=None, funcv=None, funcw = lambda x: 1) #*x/top2.a, cte=True)
            
        kCp = finalize_symmetric_matrix(kCp)
    
    # Prescribed loads
    else:
        kCp = np.zeros_like(k0)
        # here disp_mag is mag of applied force but to make it easier and not complicate the inputs, its just used here temporarily 
        disp_panel.add_distr_load_fixed_x(disp_panel.a, funcx=None, funcy=None, funcz=lambda y: disp_mag/disp_panel.b)
    
    # fext = disp_panel.calc_fext(size=size, col0=disp_panel.col_start)
    fext = assy.calc_fext()

    # Tangent (complete) stiffness matrix
    k0 = k0 + kCp

    # fext = disp_panel.calc_fext(size=size, col0=disp_panel.col_start)
    c0 = solve(k0, fext, silent=True, **dict())
    
    f_calc = k0@c0
    # return f_calc

        
    # Calc field results
    if True:
        for vec in ['w']:#, 'w', 'Myy', 'Mxy']:#, 'Nxx', 'Nyy']:
            # res_bot = assy.calc_results(c=c0, group='bot', vec=vec, gridx=50, gridy=50)
            res_top = assy.calc_results(c=c0, group='top', vec=vec, gridx=50, gridy=50)
            # vecmin = min(np.min(np.array(res_top[vec])), np.min(np.array(res_bot[vec])))
            # vecmax = max(np.max(np.array(res_top[vec])), np.max(np.array(res_bot[vec])))
            
            temp = np.max(res_top['w'][1])
            print(f'w {temp}')
            # return res_top
        
        if False:
            # Returning Mxx
            Mxx_end = assy.calc_results(c=c0, group='top', vec='Mxx', gridx=50, gridy=50)
            dy_Mxx = np.linspace(0,disp_panel.b, 50)
    
    
    # TESTING Qx, Qy - REMOVE LATER !!!!!!!!!!!!!!!!!!!!!!!
    if True:
        final_res = 0
        for panels_i in [disp_panel]:
            Qxx_end = assy.force_out_plane(c0, group=None, eval_panel=panels_i, x_cte_force=panels_i.a, y_cte_force=None,
                      gridx=grid_x, gridy=None, NLterms=True, no_x_gauss=no_y_gauss, no_y_gauss=no_y_gauss)
            # final_res += force_intgn
        # print(final_res)
    
    
    return Qxx_end
    # return final_res



def dcb_one_and_half(no_pan, no_terms, plies, disp_mag, a2, no_y_gauss, grid_x, kw):

    '''
    ONE HALF PANEL or HALF PANEL 
    
        DCB with the bottom arm only having the contact region - no free part 
        Like: ____.__
              ____
        
        Testing it with FEM
        Values are taken from 'Characterization and analysis of the interlaminar behavior of thermoplastic
        composites considering fiber bridging and R-curve effects'
            https://doi.org/10.1016/j.compositesa.2022.107101
            
        All units in MPa, N, mm
    '''    
    
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
    a1 = 52 #0.5*a
    a2 = a2 #0.3*a

    #others
    m = no_terms
    n = no_terms
    
    m_conn = 30
    n_conn = 30

    simple_layup = [0]*15
    # simple_layup += simple_layup[::-1]

    laminaprop = (E1, E2, nu12, G12, G12, G12)
     
    # Top DCB panels
    top1 = Shell(group='top', x0=0, y0=0, a=a1, b=b, m=m_conn, n=n_conn, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 2:
        top2 = Shell(group='top', x0=a1, y0=0, a=a-a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    if no_pan == 3:
        top2 = Shell(group='top', x0=a1, y0=0, a=a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        top3 = Shell(group='top', x0=a1+a2, y0=0, a=a-a1-a2, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    # Bottom DCB panels
    bot1 = Shell(group='bot', x0=0, y0=0, a=a1, b=b, m=m_conn, n=n_conn, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
 
    # boundary conditions
    
    BC = 'bot_end_fixed'
    # Possible strs: 'bot_fully_fixed', 'bot_end_fixed'
    
    clamped = True
    
    if clamped:
        bot_r = 0
        bot_t = 0
        top1_x1_wr = 1
        
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

        bot1.x1u = 1 ; bot1.x1ur = 1 ; bot1.x2u = bot_t ; bot1.x2ur = bot_r
        bot1.x1v = 1 ; bot1.x1vr = 1 ; bot1.x2v = bot_t ; bot1.x2vr = bot_r
        bot1.x1w = 1 ; bot1.x1wr = 1 ; bot1.x2w = bot_t ; bot1.x2wr = bot_r
        bot1.y1u = 1 ; bot1.y1ur = 1 ; bot1.y2u = 1 ; bot1.y2ur = 1
        bot1.y1v = 1 ; bot1.y1vr = 1 ; bot1.y2v = 1 ; bot1.y2vr = 1
        bot1.y1w = 1 ; bot1.y1wr = 1 ; bot1.y2w = 1 ; bot1.y2wr = 1
        
        
    # All connections - list of dict
    if no_pan == 2:
        conn = [
         # skin-skin
         dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
         dict(p1=top1, p2=bot1, func='SB'), #'_TSL', tsl_type = 'linear'), 
            # dict(p1=top2, p2=bot2, func='SB_TSL', tsl_type = 'linear')
        ]
    if no_pan == 3:
        conn = [
         # skin-skin
           dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
           dict(p1=top2, p2=top3, func='SSxcte', xcte1=top2.a, xcte2=0),
           dict(p1=top1, p2=bot1, func='SB'), #'_TSL', tsl_type = 'linear'), 
            # dict(p1=top2, p2=bot2, func='SB_TSL', tsl_type = 'linear')
        ]
    
    # This determines the positions of each panel's (sub)matrix in the global matrix when made a MD obj below
    # So changing this changes the placements i.e. starting row and col of each
    if no_pan == 2:
        panels = [bot1, top1, top2]
    if no_pan == 3:
        panels = [bot1, top1, top2, top3]

    assy = MultiDomain(panels=panels, conn=conn) # assy is now an object of the MultiDomain class
    # Here the panels (shell objs) are modified -- their starting positions in the global matrix is assigned etc


    # Panel at which the disp is applied
    if no_pan == 2:
        disp_panel = top2
    if no_pan == 3:
        disp_panel = top3

    k0 = assy.calc_kC(conn)
    
    size = k0.shape[0]
    
    # Prescribed Displacements
    if False:
        ######## THIS SHOULD BE CHANGED LATER PER DISP TYPE ###########################################
        ku, kv, kw = calc_ku_kv_kw_point_pd(disp_panel)
        
        # print('called first')
        disp_type = 'line_xcte' # change based on what's being applied
        
        if disp_type == 'point':
            # Penalty Stiffness
            # Disp in z, so only kw is non zero. ku and kv are zero
            kCp = fkCpd(0., 0., kw, disp_panel, disp_panel.a, disp_panel.b/2, size, disp_panel.row_start, disp_panel.col_start)
            # Point load (added to shell obj)
            wp = disp_mag # mm
            disp_panel.add_point_pd(disp_panel.a, disp_panel.b/2, 0., 0., 0., 0., kw, wp)
        if disp_type == 'line_xcte':
            kCp = fkCld_xcte(0., 0., kw, disp_panel, disp_panel.a, size, disp_panel.row_start, disp_panel.col_start)
            disp_panel.add_distr_pd_fixed_x(disp_panel.a, None, None, kw,
                                      funcu=None, funcv=None, funcw = lambda y: disp_mag) #*y/top2.b, cte=True)
        if disp_type == 'line_ycte':
            kCp = fkCld_ycte(0., 0., kw, disp_panel, disp_panel.b, size, disp_panel.row_start, disp_panel.col_start)
            disp_panel.add_distr_pd_fixed_y(disp_panel.b, None, None, kw,
                                      funcu=None, funcv=None, funcw = lambda x: 1) #*x/top2.a, cte=True)
        kCp = finalize_symmetric_matrix(kCp)
    
    # Applied Load
    else:
        kCp = np.zeros_like(k0)
        # here disp_mag is mag of applied force but to make it easier and not complicate the inputs, its just used here temporarily 
        disp_panel.add_distr_load_fixed_x(disp_panel.a, funcx=None, funcy=None, funcz=lambda y: disp_mag/disp_panel.b)
        

    # fext = disp_panel.calc_fext(size=size, col0=disp_panel.col_start)
    fext = assy.calc_fext()
        
    # Tangent (complete) stiffness matrix
    k0 = k0 + kCp

    # fext = disp_panel.calc_fext(size=size, col0=disp_panel.col_start)
    c0 = solve(k0, fext, silent=True, **dict())


    generate_plots = False
    
    # Calc w
    if True:
        for vec in ['w']:#, 'w', 'Myy', 'Mxy']:#, 'Nxx', 'Nyy']:
            res_bot = assy.calc_results(c=c0, group='bot', vec=vec, gridx=50, gridy=50)
            res_top = assy.calc_results(c=c0, group='top', vec=vec, gridx=50, gridy=50)
            
            temp_top = np.max(res_top['w'][2])
            temp_bot = np.max(res_top['w'][0])
            print(f'w_top {temp_top:.3f} -- w_bot {temp_bot:.6f}')
            
            if generate_plots:
                # if vec == 'w':
                if True:
                    assy.plot(c=c0, group='bot', vec=vec, filename='test_dcb_before_opening_bot_tsl.png', show_boundaries=True,
                                                colorbar=True, res = res_bot, display_zero=True, 
                                                flip_plot=False)
            
            # Open images
            if generate_plots:
                img_popup('test_dcb_before_opening_bot_tsl.png',1, f"{vec} bot")
                plt.show()

    # Testing Mx at the tip
    if False:
        stress = assy.stress(c0, group=None, gridx=100, gridy=100, NLterms=True, no_x_gauss=None, no_y_gauss=None,
                   eval_panel=top1, x_cte_force=None, y_cte_force=None)
        Mxx_end = stress["Mxx"][0][:,-1]
        dy_Mxx = np.linspace(0,disp_panel.b, 100)
        # print(np.shape(final_res))
    
    # TESTING Qx, Qy 
    if False:
        force = 0
        for panels_i in [top3]:
            force = assy.force_out_plane(c0, group=None, eval_panel=panels_i, x_cte_force=panels_i.a, y_cte_force=None,
                      gridx=grid_x, gridy=None, NLterms=True, no_x_gauss=300, no_y_gauss=no_y_gauss)
            # final_res += force_intgn
    
        # return force
    
    # Plotting results
    if False:
        for vec in ['w']:#, 'w', 'Myy', 'Mxy']:#, 'Nxx', 'Nyy']:
            res_bot = assy.calc_results(c=c0, group='bot', vec=vec, gridx=50, gridy=50)
            res_top = assy.calc_results(c=c0, group='top', vec=vec, gridx=50, gridy=50)
            vecmin = min(np.min(np.array(res_top[vec])), np.min(np.array(res_bot[vec])))
            vecmax = max(np.max(np.array(res_top[vec])), np.max(np.array(res_bot[vec])))

            # return res_top
            
            if vec != 'w':
                print(f'{vec} :: {vecmin:.3f}  {vecmax:.6f}')
            # if vec == 'w':
            if True:
                # Printing max min per panel
                if True:
                    for pan in range(0,np.shape(res_bot[vec])[0]):
                        print(f'{vec} top{pan+1} :: {np.min(np.array(res_top[vec][pan])):.3f}  {np.max(np.array(res_top[vec][pan])):.3f}')
                    print('------------------------------')
                    for pan in range(0,np.shape(res_bot[vec])[0]): 
                        print(f'{vec} bot{pan+1} :: {np.min(np.array(res_bot[vec][pan])):.3f}  {np.max(np.array(res_bot[vec][pan])):.3f}')
                    print('------------------------------')
                print(f'Global TOP {vec} :: {np.min(np.array(res_top[vec])):.3f}  {np.max(np.array(res_top[vec])):.3f}')
                print(f'Global BOT {vec} :: {np.min(np.array(res_bot[vec])):.3f}  {np.max(np.array(res_bot[vec])):.3f}')
                # print(res_bot[vec][1][:,-1]) # disp at the tip
                final_res = res_top
            
            if generate_plots:
                # if vec == 'w':
                if True:
                    assy.plot(c=c0, group='bot', vec=vec, filename='test_dcb_before_opening_bot_tsl.png', show_boundaries=True,
                                                colorbar=True, res = res_bot, vecmax=vecmax, vecmin=vecmin, display_zero=True, 
                                                flip_plot=False)
                    
                    assy.plot(c=c0, group='top', vec=vec, filename='test_dcb_before_opening_top_tsl.png', show_boundaries=True,
                                              colorbar=True, res = res_top, vecmax=vecmax, vecmin=vecmin, display_zero=True,
                                              flip_plot=False)
            
            
            # Open images
            if generate_plots:
                img_popup('test_dcb_before_opening_top_tsl.png',1, f"{vec} top")
                img_popup('test_dcb_before_opening_bot_tsl.png',2, f"{vec} bot")
                plt.show()
                
    # Calcuate separation
    if False:
        res_pan_top = assy.calc_results(c=c0, eval_panel=top1, vec='w', 
                                no_x_gauss=200, no_y_gauss=50)
        res_pan_bot = assy.calc_results(c=c0, eval_panel=bot1, vec='w', 
                                no_x_gauss=200, no_y_gauss=50)
        print(np.shape(res_pan_bot['w'][0]))
        del_d = assy.calc_separation(res_pan_top, res_pan_bot)
        # Plotting separation and displacements
        vmin=min(np.min(res_pan_top['w'][0]), np.min(res_pan_bot['w'][0]))
        vmax=max(np.max(res_pan_top['w'][0]), np.max(res_pan_bot['w'][0]))
        print(np.min(res_pan_top['w'][0]), np.max(res_pan_top['w'][0]))
        print(np.min(res_pan_bot['w'][0]), np.max(res_pan_bot['w'][0]))
        levels = np.linspace(vmin, vmax, 20)
        plt.figure(figsize=(8,10))
        plt.subplot(3,1,1)
        plt.contourf(del_d, cmap='jet')
        plt.gca().set_title('Separation between plates in contact [mm]')
        plt.colorbar()
        plt.subplot(3,1,2)
        plt.contourf(res_pan_top['w'][0], cmap='jet', levels=levels)
        plt.gca().set_title('Top displ [mm]')
        plt.colorbar()
        plt.subplot(3,1,3)
        plt.contourf(res_pan_bot['w'][0], cmap='jet', levels=levels)
        plt.gca().set_title('Bot displ [mm]')
        plt.colorbar()
        plt.show()
        
        monoton_top = monotonicity_check_displ(res_pan_top['w'][0])
        monoton_bot = monotonicity_check_displ(res_pan_bot['w'][0])
        
        if np.all(monoton_top):
            print('Top displ is monotonically increasing or constant')
        if np.all(monoton_bot):
            print('Bot displ is monotonically increasing or constant')
    else:
        monoton_top = None
        monoton_bot = None
        
    # return force
    return monoton_top, monoton_bot



def test_leg_sigm(res_x_prev=None, del_d_fit=None):
    import scipy
    from scipy.optimize import leastsq
    from panels import Shell
    
    c2 = 15
    
    def reference(xi, c2):
        b = 15
        a = -15
        c1 = 1
        x = (xi+1)*(b-a)/2 + a
        ftn = np.divide(1, 1 + np.exp(-c1*(x-c2)))
        return ftn
    
    # Properties
    E1 = (138300. + 128000.)/2. # MPa
    E2 = (10400. + 11500.)/2. # MPa
    G12 = 5190. # MPa
    nu12 = 0.316
    ply_thickness = 0.14 # mm

    # Plate dimensions (overall)
    a = 100 # mm
    b = 25  # mm

    simple_layup = [0]*15
    laminaprop = (E1, E2, nu12, G12, G12, G12)
    
    m = 30
    n = 2
    xi = np.linspace(-1,1,120)
    y = reference(xi, c2)
    # y1 = del_d[15,:,-1]
     
    def func(c_w):
        c = np.repeat(c_w, 3)
        s = Shell(x0=0, y0=0, a=a, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        s.x1u = 1 ; s.x1ur = 0 ; s.x2u = 1 ; s.x2ur = 1
        s.x1v = 1 ; s.x1vr = 0 ; s.x2v = 1 ; s.x2vr = 1
        s.x1w = 1 ; s.x1wr = 0 ; s.x2w = 1 ; s.x2wr = 1
        s.y1u = 1 ; s.y1ur = 1 ; s.y2u = 1 ; s.y2ur = 1
        s.y1v = 1 ; s.y1vr = 1 ; s.y2v = 1 ; s.y2vr = 1
        s.y1w = 1 ; s.y1wr = 1 ; s.y2w = 1 ; s.y2wr = 1
        _, fields = s.uvw(c, gridx=120)
        return fields['w'][100,:]  
    
    def residuals(c_w, del_d_fit):
        error = del_d_fit - func(c_w)
        print(f'Error {np.linalg.norm(error)}')
        return error.flatten()
    
    if res_x_prev is None:
        guess = np.random.rand(m*n)
    else:
        guess = res_x_prev
     
    res_x, ier = leastsq(func=residuals, x0=guess, args=(del_d_fit), ftol=1.49012e-4, xtol=1.49012e-4)
    # best_c = popt2.x
    
    if False:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
        plt.title(f'm={m} n={n}')
        plt.plot(xi, func(res_x)[0,:], label='Legendre Polynomials', color='red')
        plt.plot(xi, func(res_x)[1:,:].T, label='_nolegend_', color='red')
        plt.plot(xi, y, label='Sigmoid')
        plt.xlabel('xi')
        plt.ylabel('Ftn value')
        plt.legend()
        plt.grid()
        if False:
            ax_zoomed = zoomed_inset_axes(ax, zoom=40, loc='center right')
            ax_zoomed.plot(xi, func(res_x)[0,:], label='Legendre Polynomials', color='red')
            ax_zoomed.plot(xi, func(res_x)[1:,:].T, label='_nolegend_', color='red')
            ax_zoomed.plot(xi, y, label='Sigmoid')
            ax_zoomed.set(xlim=[-1,-0.95], ylim=[-0.0025,0.0025])
            mark_inset(ax, ax_zoomed, loc1=2, loc2=4, fc="none", ec="0.5")
        # plt.legend()
        plt.show()
    if True:
        filename = f'legen_ftn_val_mn_{m}_{n}_c2_{c2}'
        np.save(filename, func(res_x))
    
    return res_x, ier
    # plt.plot(x, func(x, best_c))

def plot_test_leg_sigm(c_w=None, ftnval_sig_leg=None, ftnval_leg=None, m=None, n=None, c2=None, del_d_fit=None):
    import scipy
    from scipy.optimize import leastsq
    from panels import Shell

     
    # Properties
    E1 = (138300. + 128000.)/2. # MPa
    E2 = (10400. + 11500.)/2. # MPa
    G12 = 5190. # MPa
    nu12 = 0.316
    ply_thickness = 0.14 # mm

    # Plate dimensions (overall)
    a = 100 # mm
    b = 25  # mm

    simple_layup = [0]*15
    laminaprop = (E1, E2, nu12, G12, G12, G12)
    
    xi = np.linspace(-1,1,500)
                
    def reference(xi, c2):
        b = 15
        a = -15
        c1 = 1
        x = (xi+1)*(b-a)/2 + a
        ftn = np.divide(1, 1 + np.exp(-c1*(x-c2)))
        return ftn
        
    # y = reference(xi, c2)

    def func(c_w):
        c = np.repeat(c_w, 3)
        s = Shell(x0=0, y0=0, a=a, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
        s.x1u = 1 ; s.x1ur = 0 ; s.x2u = 1 ; s.x2ur = 1
        s.x1v = 1 ; s.x1vr = 0 ; s.x2v = 1 ; s.x2vr = 1 
        s.x1w = 1 ; s.x1wr = 0 ; s.x2w = 1 ; s.x2wr = 1 
        s.y1u = 1 ; s.y1ur = 1 ; s.y2u = 1 ; s.y2ur = 1
        s.y1v = 1 ; s.y1vr = 1 ; s.y2v = 1 ; s.y2vr = 1
        s.y1w = 1 ; s.y1wr = 1 ; s.y2w = 1 ; s.y2wr = 1
        _, fields = s.uvw(c, gridx=120)
        return fields['w'][100,:]
    
    if False:
        if False:
            filename = f'ftn_val_mn_{m}_{n}_c2_{c2}'
            np.save(filename, func(c_w))
        if True:
            mpl.rcParams.update(mpl.rcParamsDefault)
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
            plt.title(f'm={m} n={n}', fontsize=14)
            plt.plot(xi, ftnval_sig_leg, label='Family of Sigmoid Functions', color='red')
            # plt.plot(xi, func(c_w), label='Family of Sigmoid Functions', color='red')
            # plt.plot(xi, func(c_w)[1:,:].T, label='_nolegend_', color='red')
            plt.plot(xi, del_d_fit, label='Representative Displacement Function', color='dodgerblue')
            # plt.plot(xi, reference(xi, c2), label='Sigmoid')
            plt.xlabel(r'$\xi$', fontsize=14)
            plt.ylabel('Function value', fontsize=14)
            plt.legend(fontsize=14, loc='upper left')
            plt.grid()
            # Zoomed in view of the existing curve
            if False:
                ax_zoomed = zoomed_inset_axes(ax, zoom=6, loc='center right')
                ax_zoomed.plot(xi[:75], func(c_w)[:75], label='Sigmoid enriched Legendre Polynomials', color='red')
                # ax_zoomed.plot(xi, func(c_w)[0,:], label='Sigmoid enriched Legendre Polynomials', color='red')
                # ax_zoomed.plot(xi, func(c_w)[1:,:].T, label='_nolegend_', color='red')
                ax_zoomed.plot(xi[:75], del_d_fit[:75], label='Sigmoid', color='blue')
                ax_zoomed.set(xlim=[0.072,0.073], ylim=[-0.00002,0.00002])
                mark_inset(ax, ax_zoomed, loc1=3, loc2=4, fc="none", ec="0.5")
                if False:
                    ax_zoomed = zoomed_inset_axes(ax, zoom=15000, loc='center left')
                    ax_zoomed.plot(xi, func(c_w), label='Sigmoid enriched Legendre Polynomials', color='red')
                    # ax_zoomed.plot(xi, func(c_w)[0,:], label='Sigmoid enriched Legendre Polynomials', color='red')
                    # ax_zoomed.plot(xi, func(c_w)[1:,:].T, label='_nolegend_', color='red')
                    ax_zoomed.plot(xi, del_d_fit, label='Sigmoid')
                    ax_zoomed.set(xlim=[-1,-0.99995], ylim=[-0.000002,0.000002])
                    mark_inset(ax, ax_zoomed, loc1=3, loc2=4, fc="none", ec="0.5")
            # Insert any curve in the zoomed in view
            if True:
                ax_inset = inset_axes(ax, width="60%", height="50%",
                                       bbox_to_anchor=(.1, .2, 1, 1),
                                       bbox_transform=ax.transAxes, loc=3)
                ax_inset.plot(xi[:75], ftnval_sig_leg[:75], label='Family of Sigmoid Functions', color='red')
                # ax_inset.plot(xi[:75], func(c_w)[:75], label='Family of Sigmoid Functions', color='red')
                ax_inset.plot(xi[:75], del_d_fit[:75], label='Family of Sigmoid Functions', color='dodgerblue')
                ax_inset.set(xlim=[-1,0.25])
                ax.add_patch(plt.Rectangle((0.04, 0.04), .575, .015, ls="--", ec="k", fc="none",
                           transform=ax.transAxes))
                mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
            # plt.legend()
            plt.show()
        if True:
            print(np.min(func(c_w)))
            # print(f'{func(c_w)[0:5]}')
            # print(np.shape(func(c_w)))
            # return func(c_w)
        if False:
            plt.figure()
            plt.plot(func(c_w)[:70])
            plt.show()
        if False:
            plt.figure()
            print(np.shape(func(c_w)))
            # plt.contourf((func(c_w) - y.reshape((500,1)))/y.reshape((500,1)))
            plt.contourf(func(c_w))
            plt.xlabel('eta')
            plt.ylabel('xi')
            plt.colorbar()
            plt.figure()

    # Directly using the function values
    if True:
        # Using zoomed in views of certain sections of the graphs
        if False:
            # m = 15
            # n = 8
            # c2 = 0
            
            mpl.rcParams.update(mpl.rcParamsDefault)
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
            plt.title(f'm={m} n={n}', fontsize=14)
            plt.plot(xi, ftnval_sig_leg[0,:], label='Sigmoid enriched Legendre Polynomials', color='red')
            plt.plot(xi, ftnval_sig_leg[1:,:].T, label='_nolegend_', color='red')
            plt.plot(xi, ftnval_leg[0,:], label='Legendre Polynomials', color='orange')
            plt.plot(xi, ftnval_leg[1:,:].T, label='_nolegend_', color='orange')
            plt.plot(xi, reference(xi, c2), label='Sigmoid', color='blue')
            plt.xlabel(r'$\xi$', fontsize=14)
            plt.ylabel('Function value', fontsize=14)
            plt.legend(fontsize=13, loc='upper left')
            plt.grid()
            if True:
                ax_zoomed = zoomed_inset_axes(ax, zoom=5, loc='center right')
                ax_zoomed.plot(xi, ftnval_sig_leg[0,:], label='Sigmoid enriched Legendre Polynomials', color='red')
                ax_zoomed.plot(xi, ftnval_sig_leg[1:,:].T, label='_nolegend_', color='red')
                ax_zoomed.plot(xi, ftnval_leg[0,:], label='Legendre Polynomials', color='orange')
                ax_zoomed.plot(xi, ftnval_leg[1:,:].T, label='_nolegend_', color='orange')
                ax_zoomed.plot(xi, reference(xi, c2), label='Sigmoid', color='blue')
                ax_zoomed.set(xlim=[-1,-0.85], ylim=[-0.02,0.01])
                mark_inset(ax, ax_zoomed, loc1=2, loc2=4, fc="none", ec="0.5")
                if False:
                    ax_zoomed = zoomed_inset_axes(ax, zoom=100, loc='center right')
                    ax_zoomed.plot(xi, ftnval_sig_leg[0,:], label='Sigmoid enriched Legendre Polynomials', color='red')
                    ax_zoomed.plot(xi, ftnval_sig_leg[1:,:].T, label='_nolegend_', color='red')
                    ax_zoomed.plot(xi, ftnval_leg[0,:], label='Legendre Polynomials', color='orange')
                    ax_zoomed.plot(xi, ftnval_leg[1:,:].T, label='_nolegend_', color='orange')
                    ax_zoomed.plot(xi, reference(xi, c2), label='Sigmoid', color='blue')
                    ax_zoomed.set(xlim=[-0.90,-0.895], ylim=[-0.001,0.001])
                    mark_inset(ax, ax_zoomed, loc1=2, loc2=4, fc="none", ec="0.5")
            plt.show()
    
        # Plotting another curve in the zoomed in view
        if True:
            mpl.rcParams.update(mpl.rcParamsDefault)
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
            plt.title(f'm={m} n={n}', fontsize=14)
            plt.title(f'm={m} n={n}', fontsize=14)
            plt.plot(xi, ftnval_sig_leg[0,:], label='Sigmoid enriched Legendre Polynomials', color='red')
            plt.plot(xi, ftnval_sig_leg[1:,:].T, label='_nolegend_', color='red')
            plt.plot(xi, ftnval_leg[0,:], label='Legendre Polynomials', color='orange')
            plt.plot(xi, ftnval_leg[1:,:].T, label='_nolegend_', color='orange')
            plt.plot(xi, reference(xi, c2), label='Sigmoid', color='dodgerblue')
            plt.xlabel(r'$\xi$', fontsize=14)
            plt.ylabel('Function value', fontsize=14)
            plt.legend(fontsize=13, loc='upper left')
            plt.grid()

            if True:
                ax_inset = inset_axes(ax, width="60%", height="50%",
                                       bbox_to_anchor=(.2 , .2, 1, 1),
                                       bbox_transform=ax.transAxes, loc=3)
                
                ax_inset.plot(xi, ftnval_sig_leg[0,:], label='Sigmoid enriched Legendre Polynomials', color='red', linestyle='dotted', linewidth=0.5, dashes=(2,10))
                ax_inset.plot(xi, ftnval_sig_leg[1:,:].T, label='_nolegend_', color='red', linestyle='dotted', linewidth=0.5, dashes=(2,10))
                ax_inset.plot(xi, ftnval_leg[0,:], label='Legendre Polynomials', color='orange')
                ax_inset.plot(xi, ftnval_leg[1:,:].T, label='_nolegend_', color='orange')
                ax_inset.plot(xi, reference(xi, c2), label='Sigmoid', color='blue')
                ax_inset.set(xlim=[-1,0.25], ylim=[-0.001,0.001])
                ax_inset.grid()
                # ax.add_patch(plt.Rectangle((0.04, 0.04), .575, .015, ls="--", ec="k", fc="none",
                           # transform=ax.transAxes))
                mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
            # plt.legend()
            plt.show()

            
    def residuals(c_w, y):
        error = y - func(c_w)
        print(f'Error {np.linalg.norm(error)}')
        return error


if __name__ == "__main__":
        
    # COMPLETE DCB
    if False:
        test_dcb_vs_fem(no_pan=4, no_terms=10, plies=1, disp_mag=1, a2=0.015, no_y_gauss=200, grid_x=200)

    # DCB TEST
    # Qxx_end_15_9, dy_Qxx_15_9, Mxx_end_15_9, dy_Mxx_15_9 = test_dcb_vs_fem(no_pan=3, no_terms=30, plies=1, disp_mag=1, 
    #                                 a2 = 1, no_y_gauss=100, grid_x=1000, kw=1e5)
    
    # SINGLE PLATE TEST 
    if False:
        sp_kr7_t8 = np.zeros((11,2))
        count = 0
        for a1 in [5]:#,10,17.5,25,32.5,50,62.5,75,82.5,90,95]:
            print(f'a1 {a1}')
            sp_kr7_t8[count, 1] = single_panel_bending(no_pan=2, no_terms=8, 
                                            plies=1, disp_mag=15, a1=a1, no_y_gauss=100, grid_x=500)
            sp_kr7_t8[count, 0] = a1
            count += 1
    if False: 
        f_calc = single_panel_bending(no_pan=2, no_terms=8, 
                                        plies=1, disp_mag=118.8, a1=a1, no_y_gauss=100, grid_x=500)

    
    # ONE AND HALF DCB TEST (HALF PANEL)
    if False:
        hp_kr9_ksb9_t15 = np.zeros((6,2))
        count = 0
        for a2 in [0.5]:#,1,5]:#,10,20,30]:
            print(f'a2 {a2}')
            hp_kr9_ksb9_t15[count, 1] = dcb_one_and_half(no_pan=3, no_terms=25, plies=1, disp_mag=15, 
                                    a2 = a2, no_y_gauss=300, grid_x=1000, kw=1e5)
            hp_kr9_ksb9_t15[count, 0] = a2
            count += 1
    if False:
        dcb_one_and_half(no_pan=3, no_terms=25, plies=1, disp_mag=1075.01, 
                                a2 = 24, no_y_gauss=300, grid_x=1000, kw=1e5)
        
    if False:
        hp_kr9_ksb9_t15 = np.zeros((7,2))
        count = 0
        for terms in [4,8,12,15,20,25,30]:#,10,20,30]:
            print(f'terms {terms}')
            hp_kr9_ksb9_t15[count, 1] = dcb_one_and_half(no_pan=3, no_terms=terms, plies=1, disp_mag=15, 
                                    a2 = 1, no_y_gauss=300, grid_x=1000, kw=1e5)
            hp_kr9_ksb9_t15[count, 0] = a2
            count += 1

    # monoton_top, monoton_bot = dcb_one_and_half(no_pan=3, no_terms=25, plies=1, disp_mag=15, 
                            # a2 = 0.5, no_y_gauss=300, grid_x=1000, kw=1e5)
                
    # Fitting arbitary curves using different sets of shape functions
    if True:
        # res_x, ier = test_leg_sigm(res_x_prev=None, del_d_fit=del_d_fit_interp)
        
        # Plotting stuff 
        if True:
            # m = 10
            n = 8
            c2 = 15
            for m in [10,15]:
                ftnval_sig_leg = f'ftn_val_mn_{m}_{n}_c2_{c2}'
                ftnval_leg = f'legen_ftn_val_mn_{m}_{n}_c2_{c2}'
                # plot_test_leg_sigm(c_w=None, ftnval_sig_leg=legen_ftn_val_mn_19_2_c2_15, ftnval_leg=None, m=m, n=n, c2=c2, del_d_fit=del_d_fit_interp)
                plot_test_leg_sigm(c_w=None, ftnval_sig_leg=locals()[ftnval_sig_leg], ftnval_leg=locals()[ftnval_leg], m=m, n=n, c2=c2)
                # plot_test_leg_sigm(c_w=res_x, m=m, n=n, c2=c2, del_d_fit=del_d_fit_interp)