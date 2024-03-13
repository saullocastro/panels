import sys
sys.path.append('C:/Users/natha/Documents/GitHub/panels')
# sys.path.append('..\\..')
import os
os.chdir('C:/Users/natha/Documents/GitHub/panels/tests/multidomain')

import numpy as np
from structsolve import solve
from structsolve.sparseutils import finalize_symmetric_matrix
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

# Printing with reduced no of points (ease of viewing) - Suppress this to print in scientific notations and restart the kernel
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

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


def test_dcb_damage(no_pan, no_terms, plies):

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

    assy = MultiDomain(panels=panels, conn=conn) # assy is now an object of the MultiDomain class
    # Here the panels (shell objs) are modified -- their starting positions in the global matrix is assigned etc

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
        
        
        
        
        
    kC = assy.calc_kT()
    size = kC.shape[0]
        
    wp = 0.01   
    
    # --------- IMPROVE THE STARTING GUESS --------------
    if True:
        # Prescribed Displacements
        if True:
            disp_type = 'line_xcte' # change based on what's being applied
            
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
        # print(f'fext before {np.linalg.norm(fext)}')
        
        k0 = kC + kCp
        ci = solve(k0, fext)
        # print(f'ci {np.linalg.norm(ci)}')
    else:
        ci = np.zeros(size) # Only used to calc initial fint
        
    wp_max = 5 # [mm]
    no_iter = 10  
    
    # Displacement Incrementation
    for wp in np.linspace(0.01, wp_max, no_iter): 
        print('wp', wp)
    
        kC = assy.calc_kT(c=ci)
        print('kC',np.linalg.det(kC.toarray()))
        size = kC.shape[0]
        
        # Prescribed Displacements
        if True:
            disp_type = 'line_xcte' # change based on what's being applied
            
            # Clears all previously added displs
            disp_panel.clear_disps()
            
            ####### ENSURE THAT PER ITERATION, ADDITIONAL PDS ARENT ADDED - OLD ONES NEED TO BE DELETED ################
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

        # Tangent (complete) stiffness matrix
        k0 = kC + kCp
        print('k0',np.linalg.det(k0.toarray()))
        
        # Inital guess ci and increment dc (same size as fext)
        dc = np.zeros_like(fext)
        
        # Inital fint (0 bec c=0)
        fint = assy.calc_fint(c=ci)
        Ri = fint - fext
        
        epsilon = 1.e-4 # Convergence criteria
        D = k0.diagonal() # For convergence - calc at beginning of load increment
        
        count = 0 # Tracks number of NR iterations 
        ignore = False
        
        all_fint = []
        all_dc = []
        all_Ri = []
        all_crisfield = []
        # Modified Newton Raphson Iteration
        while True:
            print()
            print('count',count)
            # print('k0',np.linalg.det(k0.toarray()))
            print(f'Ri {np.linalg.norm(Ri)}')
            all_Ri.append(np.linalg.norm(Ri))
            dc = solve(k0, -Ri, silent=True)
            all_dc.append(np.linalg.norm(dc))
            
            if np.isclose(np.linalg.norm(fext)/np.linalg.norm(fint), 0.8):
                ignore = True
            if not ignore:
                if np.linalg.norm(fext)/np.linalg.norm(fint) > 1000: #and np.linalg.norm(c)/np.linalg.norm(c_org) < 100:
                    dc = 15*dc
                elif np.linalg.norm(fext)/np.linalg.norm(fint) > 100: #and np.linalg.norm(c)/np.linalg.norm(c_org) < 100:
                    dc = 1.5*dc
            
            print(f'dc {np.linalg.norm(dc)}')
            
            # print(f'ci {max(ci)}')
            c = ci + dc
            print(f'c {np.linalg.norm(c)}')
            fint = np.asarray(assy.calc_fint(c=c))
            all_fint.append(np.linalg.norm(fint))
            print(f'fext {np.linalg.norm(fext)}')
            print(f'fint {np.linalg.norm(fint)}')
            Ri = fint - fext
            crisfield_test = scaling(Ri, D)/max(scaling(fext, D), scaling(fint, D))
            all_crisfield.append(crisfield_test)
            # print('     ',crisfield_test)
            if crisfield_test < epsilon:
                break
            count += 1
            kC = assy.calc_kT(c=c) 
            # print(f'kC {np.max(kC)}')
            k0 = kC + kCp
            ci = c.copy()
            if count > 500:
                break
                raise RuntimeError('NR not converged :(')
            # if count == 1:
            #     c_org = c.copy()
            # if np.isclose(np.linalg.norm(fext)/np.linalg.norm(fint), 0.8):
            #     ignore = True
            # if not ignore:
            #     if np.linalg.norm(fext)/np.linalg.norm(fint) > 1000: #and np.linalg.norm(c)/np.linalg.norm(c_org) < 100:
            #         ci = 1.05*ci
            #     elif np.linalg.norm(fext)/np.linalg.norm(fint) > 100: #and np.linalg.norm(c)/np.linalg.norm(c_org) < 100:
            #         ci = 1.005*ci
            if np.linalg.norm(fint)/np.linalg.norm(fext) > 1.2: 
                break
        
        plt.figure()
        plt.plot(all_dc)
        plt.ylabel('dc')
        plt.xlabel('NR iteration number')
        # plt.plot(np.linalg.norm(fext))
        plt.show()
        
        plt.figure()
        plt.plot(all_Ri)        
        plt.ylabel('Ri')
        plt.xlabel('NR iteration number')
        plt.show()
    
        plt.figure()
        plt.plot(all_fint)        
        plt.ylabel('F_int')
        plt.xlabel('NR iteration number')
        plt.show()
        
        plt.figure()
        plt.plot(all_crisfield)        
        plt.ylabel('crisfield value')
        plt.xlabel('NR iteration number')
        plt.show()
        
        
    c0 = c.copy()
            
            
    
    
    
    
    
    
    
    
    
    
    # ------------------ RESULTS AND POST PROCESSING --------------------
    
    generate_plots = False
    
    # Plotting results
    if True:
        for vec in ['w', 'Mxx']:#, 'Myy', 'Mxy']:#, 'Nxx', 'Nyy']:
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
    
        
    return final_res

def test_saullo(no_pan, no_terms, plies):

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

    assy = MultiDomain(panels=panels, conn=conn) # assy is now an object of the MultiDomain class
    # Here the panels (shell objs) are modified -- their starting positions in the global matrix is assigned etc

    # Panel at which the disp is applied
    if no_pan == 2:
        disp_panel = top2
    if no_pan == 3:
        disp_panel = top3
        

    if True:
        ######## THIS SHOULD BE CHANGED LATER PER DISP TYPE ###########################################
        ku, kv, kw = calc_ku_kv_kw_point_pd(disp_panel)

    #### While it says Kc, ALL KC's also include Kg i.e. Kc is now Kt
    kC = assy.calc_kT()
    size = kC.shape[0]
        
    # To match the inital increment 
    wp = 0.01   
    
    # --------- IMPROVE THE STARTING GUESS --------------
    if True:
        # Prescribed Displacements
        if True:
            disp_type = 'line_xcte' # change based on what's being applied
            
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
        
        k0 = kC + kCp
        ci = solve(k0, fext)
    else:
        ci = np.zeros(size) # Only used to calc initial fint
        
    # -------------------- INCREMENTATION --------------------------
    wp_max = 5 # [mm]
    no_iter = 10  
    
    # Displacement Incrementation
    for wp in np.linspace(0.01, wp_max, no_iter): 
        print('wp', wp)
    
        kC = assy.calc_kT(c=ci)
        size = kC.shape[0]
        
        # Prescribed Displacements
        if True:
            disp_type = 'line_xcte' # change based on what's being applied
            
            # Clears all previously added displs - in NL case, theyre readded so you have 2 disps at the tip
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

        # Tangent (complete) stiffness matrix
        k0 = kC + kCp
        
        # Inital guess ci and increment dc (same size as fext)
        dc = np.zeros_like(fext)
        
        # Inital fint (0 bec c=0)
        fint = assy.calc_fint(c=ci)
        Ri = fint - fext
        
        epsilon = 1.e-4 # Convergence criteria
        D = k0.diagonal() # For convergence - calc at beginning of load increment
        
        count = 0 # Tracks number of NR iterations 
        ignore = False
        
        # Modified Newton Raphson Iteration
        while True:
            dc = solve(k0, -Ri, silent=True)
            
            # Helps initial guess be closer to what it should be
            if np.isclose(np.linalg.norm(fext)/np.linalg.norm(fint), 0.8):
                ignore = True
            if not ignore:
                if np.linalg.norm(fext)/np.linalg.norm(fint) > 1000: #and np.linalg.norm(c)/np.linalg.norm(c_org) < 100:
                    dc = 10*dc
                elif np.linalg.norm(fext)/np.linalg.norm(fint) > 100: #and np.linalg.norm(c)/np.linalg.norm(c_org) < 100:
                    dc = 1.5*dc
            
            c = ci + dc
            fint = np.asarray(assy.calc_fint(c=c))
            Ri = fint - fext
            crisfield_test = scaling(Ri, D)/max(scaling(fext, D), scaling(fint, D))
            if crisfield_test < epsilon:
                break
            count += 1
            kC = assy.calc_kT(c=c) 
            # print(f'kC {np.max(kC)}')
            k0 = kC + kCp
            ci = c.copy()
            if count > 500:
                break
                raise RuntimeError('NR not converged :(')
            # Break loop for diverging results
            if np.linalg.norm(fint)/np.linalg.norm(fext) > 1.5: 
                break
        
    c0 = c.copy()

    # ------------------ RESULTS AND POST PROCESSING --------------------
    
    generate_plots = False
    
    # Plotting results
    if True:
        for vec in ['w', 'Mxx']:#, 'Myy', 'Mxy']:#, 'Nxx', 'Nyy']:
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
    
        
    return final_res

if __name__ == "__main__":
    # test_dcb_damage(2, 4, 1)
    test_saullo(2, 4, 1)