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

# import os
# os.chdir('C:/Users/natha/Documents/GitHub/panels/tests/multidomain')

def img_popup(filename):
    
    # To open pop up images - Ignore the syntax warning :)
    # %matplotlib qt 
    # For inline images
    %matplotlib inline
    
    plt.title(filename)
    image = img.imread(filename)
    plt.imshow(image)
    plt.show()


def test_dcb_bending_pd_tsl():

    '''
        Test function for a DBC with different BCs
    '''    

    # Properties
    E1 = 127560 # MPa
    E2 = 13030. # MPa
    G12 = 6410. # MPa
    nu12 = 0.3
    ply_thickness = 0.127#e-3 # mm

    # Plate dimensions
    a = 1181.1#e-3 # mm
    b = 746.74#e-3 # mm
    
    
    
    a1 = 0.5*a

    #others
    m = 8
    n = 8

    simple_layup = [+45, -45]*20 + [0, 90]*20
    simple_layup += simple_layup[::-1]

    laminaprop = (E1, E2, nu12, G12, G12, G12)
     
    # Top DCB panels
    top1 = Shell(group='top', x0=0, y0=0, a=a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    top2 = Shell(group='top', x0=a1, y0=0, a=a-a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    # Bottom DCB panels
    bot1 = Shell(group='bot', x0=0, y0=0, a=a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    bot2 = Shell(group='bot', x0=a1, y0=0, a=a-a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    
    
    # boundary conditions
    
    BC = 'bot_fully_fixed'
    # Possible strs: 'bot_fully_fixed', 'bot_end_fixed'
    # DCB with bottom fixed
    if BC == 'bot_fully_fixed':
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
    
        bot1.x1u = 1 ; bot1.x1ur = 1 ; bot1.x2u = 1 ; bot1.x2ur = 1
        bot1.x1v = 1 ; bot1.x1vr = 1 ; bot1.x2v = 1 ; bot1.x2vr = 1 
        bot1.x1w = 1 ; bot1.x1wr = 1 ; bot1.x2w = 1 ; bot1.x2wr = 1 
        bot1.y1u = 1 ; bot1.y1ur = 1 ; bot1.y2u = 1 ; bot1.y2ur = 1
        bot1.y1v = 1 ; bot1.y1vr = 1 ; bot1.y2v = 1 ; bot1.y2vr = 1
        bot1.y1w = 1 ; bot1.y1wr = 1 ; bot1.y2w = 1 ; bot1.y2wr = 1
        
        bot2.x1u = 1 ; bot2.x1ur = 1 ; bot2.x2u = 0 ; bot2.x2ur = 0 # only right extreme of plate 2 with line ll to x is fixed
        bot2.x1v = 1 ; bot2.x1vr = 1 ; bot2.x2v = 0 ; bot2.x2vr = 0 
        bot2.x1w = 1 ; bot2.x1wr = 1 ; bot2.x2w = 0 ; bot2.x2wr = 0 
        bot2.y1u = 1 ; bot2.y1ur = 1 ; bot2.y2u = 1 ; bot2.y2ur = 1
        bot2.y1v = 1 ; bot2.y1vr = 1 ; bot2.y2v = 1 ; bot2.y2vr = 1
        bot2.y1w = 1 ; bot2.y1wr = 1 ; bot2.y2w = 1 ; bot2.y2wr = 1

    # All connections - list of dict
    conn = [
     # skin-skin
     dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
     dict(p1=bot1, p2=bot2, func='SSxcte', xcte1=bot1.a, xcte2=0),
     dict(p1=bot1, p2=top1, func='SB'), 
     dict(p1=bot2, p2=top2, func='SB_TSL', tsl_type = 'linear')
    ]
    
    # This determines the positions of each panel's (sub)matrix in the global matrix when made a MD obj below
    # So changing this changes the placements i.e. starting row and col of each
    panels = [bot1, bot2, top1, top2]
    # panels = [top2, top1, bot2, bot1]

    assy = MultiDomain(panels) # assy is now an object of the MultiDomain class
    # Here the panels (shell objs) are modified -- their starting positions in the global matrix is assigned etc

    print('bot', bot1.row_start, bot1.col_start, bot2.row_start, bot2.col_start)
    print('top', top1.row_start, top1.col_start, top2.row_start, top2.col_start)
    k0 = assy.calc_kC(conn)
    
    size = k0.shape[0]

    if True:
        ku, kv, kw = calc_ku_kv_kw_point_pd(top2)
    
    # Prescribed Displacements
    if True:
        disp_type = 'point' # change based on what's being applied
        
        if disp_type == 'point':
            # Penalty Stiffness
            # Disp in z, so only kw is non zero. ku and kv are zero
            kCp = fkCpd(0., 0., kw, top2, top2.a, top2.b/2, size, top2.row_start, top2.col_start)
            # Point load (added to shell obj)
            wp = 5#e-3 # mm
            top2.add_point_pd(top2.a, top2.b/2, 0., 0., 0., 0., kw, wp)
        if disp_type == 'line_xcte':
            kCp = fkCld_xcte(0., 0., kw, top2, top2.a, size, top2.row_start, top2.col_start)
            top2.add_distr_pd_fixed_x(top2.a, None, None, kw,
                                      funcu=None, funcv=None, funcw = lambda y: 5) #*y/top2.b, cte=True)
        if disp_type == 'line_ycte':
            kCp = fkCld_ycte(0., 0., kw, top2, top2.b, size, top2.row_start, top2.col_start)
            top2.add_distr_pd_fixed_y(top2.b, None, None, kw,
                                      funcu=None, funcv=None, funcw = lambda x: 1) #*x/top2.a, cte=True)
        
        kCp = finalize_symmetric_matrix(kCp)
        
    # Tangent (complete) stiffness matrix
    k0 = k0 + kCp
        
    # top2.add_point_load(top2.a, top2.b/2, 0, 0, 1.)
        
    fext = top2.calc_fext(size=size, col0=top2.col_start)
    c0 = solve(k0, fext)
    # print(np.shape(c0))
    
    # Plotting results
    if True:
        # ax, data = assy.plot(c=c0, group='bot', vec='w', filename='test_dcb_before_opening_bot_tsl.png', colorbar=True)
        
        #vecmin = data['vecmin']
        #vecmax = data['vecmax']
        vecmin = vecmax = None
        assy.plot(c=c0, group='top', vec='w', filename='test_dcb_before_opening_top_tsl.png', colorbar=True, vecmin=vecmin, vecmax=vecmax)
    
    # Open images
    if True:
        # plt.figure
        # img_popup('test_dcb_before_opening_bot.png')
        # plt.show()
        # plt.figure()
        img_popup('test_dcb_before_opening_top_tsl.png')
        # plt.show()
        

if __name__ == "__main__":
    test_dcb_bending_pd_tsl()