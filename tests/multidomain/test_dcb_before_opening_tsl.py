import sys
sys.path.append('..\\..')

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

# Printing with reduced no of points (ease of viewing) - Suppress this to print in scientific notations and restart the kernel
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


# import os
# os.chdir('C:/Users/natha/Documents/GitHub/panels/tests/multidomain')

def img_popup(filename, plot_no = None):
    
    # plot_no = current plot no 
    
    # To open pop up images - Ignore the syntax warning :)
    # %matplotlib qt 
    # For inline images
    # %matplotlib inline
    
    # 
    image = img.imread(filename)
    if plot_no is None:
        plt.title(filename)
        plt.imshow(image)
        plt.show()
    else:
        plt.subplot(1,2,plot_no).set_title(filename)
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


def test_dcb_vs_fem(no_pan, no_terms, plies):

    '''
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
    a = 225 # mm
    b = 25  # mm
    # Dimensions of panel 1 and 2
    a1 = 0.5*a
    a2 = 0.3*a
    
    # no_pan = 3 # no of panels per structure
    # print('No of panels = ', no_pan)

    #others
    m = no_terms
    n = no_terms
    # print(f'no terms : {m}')

    simple_layup = [+45, -45]*plies + [0, 90]*plies
    # simple_layup = [0, 0]*10 + [0, 0]*10
    simple_layup += simple_layup[::-1]
    # simple_layup += simple_layup[::-1]
    print('plies ',np.shape(simple_layup)[0])

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
    
    clamped = True
    ss = False
    
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

    k0 = assy.calc_kC(conn)
    
    size = k0.shape[0]

    # Panel at which the disp is applied
    if no_pan == 2:
        disp_panel = top2
    if no_pan == 3:
        disp_panel = top3

    if True:
        ######## THIS SHOULD BE CHANGED LATER PER DISP TYPE ###########################################
        ku, kv, kw = calc_ku_kv_kw_point_pd(disp_panel)
        # kw = 1000*kw
        # print(f'         kw disp :             {kw:.1e}')
        # kw = 1e4*top1.a*top1.b
    # Prescribed Displacements
    if True:
        # print('called first')
        disp_type = 'line_xcte' # change based on what's being applied
        # print('disp_type : ', disp_type)
        
        if disp_type == 'point':
            # Penalty Stiffness
            # Disp in z, so only kw is non zero. ku and kv are zero
            kCp = fkCpd(0., 0., kw, disp_panel, disp_panel.a, disp_panel.b/2, size, disp_panel.row_start, disp_panel.col_start)
            # Point load (added to shell obj)
            wp = 5 # mm
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
    
    # print(f'max kCp {np.max(kCp):.1e}')
    # print(f'max k0 {np.max(k0):.1e}')
        
    # Tangent (complete) stiffness matrix
    k0 = k0 + kCp

    # fext = disp_panel.calc_fext(size=size, col0=disp_panel.col_start)
    c0 = solve(k0, fext, silent=True, **dict())
    
    generate_plots = True
    # Plotting results
    if True:
        for vec in ['w']:#, 'Nxx', 'Nyy']:
            res_bot = assy.calc_results(c=c0, group='bot', vec=vec, no_x_gauss=None, no_y_gauss=None)
            res_top = assy.calc_results(c=c0, group='top', vec=vec, no_x_gauss=None, no_y_gauss=None)
            vecmin = min(np.min(np.array(res_top[vec])), np.min(np.array(res_bot[vec])))
            vecmax = max(np.max(np.array(res_top[vec])), np.max(np.array(res_bot[vec])))
            if vec != 'w':
                print(f'{vec} :: {vecmin:.3f}  {vecmax:.3f}')
            if vec == 'w':
                # Printing max min per panel
                if True:
                    for pan in range(0,np.shape(res_bot[vec])[0]):
                        print(f'{vec} top{pan+1} :: {np.min(np.array(res_top[vec][pan])):.3f}  {np.max(np.array(res_top[vec][pan])):.3f}')
                        print(f'{vec} bot{pan+1} :: {np.min(np.array(res_bot[vec][pan])):.3f}  {np.max(np.array(res_bot[vec][pan])):.3f}')
                print(f'Global TOP {vec} :: {np.min(np.array(res_top[vec])):.3f}  {np.max(np.array(res_top[vec])):.3f}')
                print(f'Global BOT {vec} :: {np.min(np.array(res_bot[vec])):.3f}  {np.max(np.array(res_bot[vec])):.3f}')
                # print(res_bot[vec][1][:,-1]) # disp at the tip
                final_res = np.min(np.array(res_top[vec]))
            
            if generate_plots:
                if vec == 'w':
                    assy.plot(c=c0, group='bot', vec=vec, filename='test_dcb_before_opening_bot_tsl.png', show_boundaries=True,
                                                colorbar=True, res = res_bot, vecmax=vecmax, vecmin=vecmin, display_zero=True)
                    
                    assy.plot(c=c0, group='top', vec=vec, filename='test_dcb_before_opening_top_tsl.png', show_boundaries=True,
                                              colorbar=True, res = res_top, vecmax=vecmax, vecmin=vecmin, display_zero=True)
    
    # Test for force
    # Panel at which the disp is applied
    if no_pan == 2:
        force_panel = top2
    if no_pan == 3:
        force_panel = top3
    if False:
        vec = 'Fxx'
        res_bot = assy.calc_results(c=c0, group='bot', vec=vec, no_x_gauss=50, no_y_gauss=60,
                                cte_panel_force=force_panel, x_cte_force=force_panel.a)
        print(res_bot)
        
    # Open images
    if generate_plots:
        img_popup('test_dcb_before_opening_top_tsl.png',1)
        img_popup('test_dcb_before_opening_bot_tsl.png',2)
        plt.show()
        
    return final_res



def test_dcb_vs_fem_force(no_pan, no_terms, plies):

    '''
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
    a = 225 # mm
    b = 25  # mm
    # Dimensions of panel 1 and 2
    a1 = 0.5*a
    a2 = 0.3*a
    
    # no_pan = 3 # no of panels per structure
    # print('No of panels = ', no_pan)

    #others
    m = no_terms
    n = no_terms
    # print(f'no terms : {m}')

    simple_layup = [+45, -45]*plies + [0, 90]*plies
    # simple_layup = [0, 0]*10 + [0, 0]*10
    simple_layup += simple_layup[::-1]
    # simple_layup += simple_layup[::-1]
    print('plies ',np.shape(simple_layup)[0])

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
    
    clamped = True
    ss = False
    
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

    k0 = assy.calc_kC(conn)
    
    size = k0.shape[0]

    # Panel at which the disp is applied
    if no_pan == 2:
        disp_panel = top2
    if no_pan == 3:
        disp_panel = top3

# FORCE ADDED
    disp_panel.add_point_load(disp_panel.a, disp_panel.b/2, 0, 0, 100.)

    fext = disp_panel.calc_fext(size=size, col0=disp_panel.col_start)

    # fext = disp_panel.calc_fext(size=size, col0=disp_panel.col_start)
    c0 = solve(k0, fext, silent=True, **dict())
    
    generate_plots = True
    # Plotting results
    if True:
        for vec in ['w']:#, 'Nxx', 'Nyy']:
            res_bot = assy.calc_results(c=c0, group='bot', vec=vec, no_x_gauss=None, no_y_gauss=None)
            res_top = assy.calc_results(c=c0, group='top', vec=vec, no_x_gauss=None, no_y_gauss=None)
            vecmin = min(np.min(np.array(res_top[vec])), np.min(np.array(res_bot[vec])))
            vecmax = max(np.max(np.array(res_top[vec])), np.max(np.array(res_bot[vec])))
            if vec != 'w':
                print(f'{vec} :: {vecmin:.3f}  {vecmax:.3f}')
            if vec == 'w':
                # Printing max min per panel
                if True:
                    for pan in range(0,np.shape(res_bot[vec])[0]):
                        print(f'{vec} top{pan+1} :: {np.min(np.array(res_top[vec][pan])):.3f}  {np.max(np.array(res_top[vec][pan])):.3f}')
                        print(f'{vec} bot{pan+1} :: {np.min(np.array(res_bot[vec][pan])):.3f}  {np.max(np.array(res_bot[vec][pan])):.3f}')
                print(f'Global TOP {vec} :: {np.min(np.array(res_top[vec])):.3f}  {np.max(np.array(res_top[vec])):.3f}')
                print(f'Global BOT {vec} :: {np.min(np.array(res_bot[vec])):.3f}  {np.max(np.array(res_bot[vec])):.3f}')
                # print(res_bot[vec][1][:,-1]) # disp at the tip
                final_res = np.min(np.array(res_top[vec]))
            
            if generate_plots:
                if vec == 'w':
                    assy.plot(c=c0, group='bot', vec=vec, filename='test_dcb_before_opening_bot_tsl.png', show_boundaries=True,
                                                colorbar=True, res = res_bot, vecmax=vecmax, vecmin=vecmin, display_zero=True)
                    
                    assy.plot(c=c0, group='top', vec=vec, filename='test_dcb_before_opening_top_tsl.png', show_boundaries=True,
                                              colorbar=True, res = res_top, vecmax=vecmax, vecmin=vecmin, display_zero=True)
    
    # Test for force
    # Panel at which the disp is applied
    if no_pan == 2:
        force_panel = top2
    if no_pan == 3:
        force_panel = top3
    if True:
        vec = 'Fxx'
        res_bot = assy.calc_results(c=c0, group='bot', vec=vec, no_x_gauss=50, no_y_gauss=60,
                                cte_panel_force=force_panel, x_cte_force=force_panel.a)
        print(res_bot)
        
    # Open images
    if generate_plots:
        img_popup('test_dcb_before_opening_top_tsl.png',1)
        img_popup('test_dcb_before_opening_bot_tsl.png',2)
        plt.show()
        
    return final_res

if __name__ == "__main__":
    # test_dcb_bending_pd_tsl()
    # print('10 terms -- 3 panels')
    test_dcb_vs_fem(3, 10, 5)
    # test_dcb_vs_fem(3, 10, 1)
    # print('------------------------------------')
    # test_dcb_vs_fem(2, 10, 5)
    # test_dcb_vs_fem(3, 10, 5)
    # print('------------------------------------')
    # test_dcb_vs_fem(2, 10, 10)
    # test_dcb_vs_fem(3, 10, 10)
    # print('------------------------------------')
    # test_dcb_vs_fem(2, 10, 50)
    # test_dcb_vs_fem(3, 10, 50)
    # print('------------------------------------')
    # test_dcb_vs_fem(2, 10, 100)
    # test_dcb_vs_fem(3, 10, 100)
    # print('------------------------------------')