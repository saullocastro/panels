import sys
sys.path.append('C:/Users/natha/Documents/GitHub/panels')
sys.path.append('..\\..')
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
        
        
        
def test_dcb_damage_prop_fext_conn(no_pan, no_terms, plies, filename='', k_i=None):

    '''
        Damage propagation from a DCB with a precrack - Uses a connecting force at the interface
            instead of the stiffness penalty connection. Check multidomain.calc_fext_dmg() for more info
            
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
    a1 = 52
    a2 = 0.015
    a3 = 10

    #others
    m_tsl = no_terms
    n_tsl = no_terms
    m = 8
    n = 8
    # print(f'no terms : {m}')

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

    # boundary conditions
    
    BC = 'bot_end_fixed'
    # Possible strs: 'bot_fully_fixed', 'bot_end_fixed'
    
    clamped = True
    ss = False
    
    if clamped:
        top_r = 0
        top_t = 0
        # top1_x1_wr = 1
    if ss:
        bot_r = 1
        bot_t = 0
        top1_x1_wr = 0
        
    # DCB with only the lower extreme end fixed at the tip. Rest free
    if BC == 'bot_end_fixed':
        top1.x1u = top_t ; top1.x1ur = 1 ; top1.x2u = 1 ; top1.x2ur = 1
        top1.x1v = top_t ; top1.x1vr = 1 ; top1.x2v = 1 ; top1.x2vr = 1 
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
    
            
    # All connections - list of dict
    if False: # incomplete
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
           dict(p1=top1, func='SB_force', tsl_type = 'bilinear', no_x_gauss=150, no_y_gauss=80, k_i=k_i)
        ]
    if no_pan == 4:
        conn = [
         # skin-skin
           dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
           dict(p1=top2, p2=top3, func='SSxcte', xcte1=top2.a, xcte2=0),
           dict(p1=top3, p2=top4, func='SSxcte', xcte1=top3.a, xcte2=0),
           dict(p1=top1, func='SB_force', tsl_type = 'bilinear', no_x_gauss=60, no_y_gauss=30, k_i=k_i)
        ]
    
    # This determines the positions of each panel's (sub)matrix in the global matrix when made a MD obj below
    # So changing this changes the placements i.e. starting row and col of each
    if no_pan == 2:
        panels = [top1, top2]
    if no_pan == 3:
        panels = [top1, top2, top3]
    if no_pan == 4:
        panels = [top1, top2, top3, top4]

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
        kw = 1e10*kw
        
    size = assy.get_size()
    
    # --------- IMPROVE THE STARTING GUESS --------------
    ci = np.zeros(size) # Only used to calc initial fint
        
        
    # -------------------- INCREMENTATION --------------------------
    # wp_max = 10 # [mm]
    # no_iter_disp = 100
    
    disp_iter_no = 0
    
    # Finding info of the connection
    for conn_list in conn:
        if conn_list['func'] == 'SB_force':
            no_x_gauss = conn_list['no_x_gauss']
            no_y_gauss = conn_list['no_y_gauss']
            tsl_type = conn_list['tsl_type']
            p_top = conn_list['p1']
            break # Assuming there is only 1 connection
        
    # Displacements 
    # w_iter = np.unique(np.concatenate((np.linspace(0.01,3,2), np.linspace(3,8,300))))
    # w_iter = np.linspace(0.01,8,100)
    w_iter = [0.01, 0.010001, 0.1, 2, 4, 6, 8]
    
    # Initilaize mat to store results
    dmg_index = np.zeros((no_y_gauss,no_x_gauss,np.shape(w_iter)[0]))
    del_d = np.zeros((no_y_gauss,no_x_gauss,np.shape(w_iter)[0]))
    kw_tsl = np.zeros((no_y_gauss,no_x_gauss,np.shape(w_iter)[0]))
    force_intgn = np.zeros((np.shape(w_iter)[0], 2)) # First col is displacements
    displ_top_root = np.zeros((50,200,np.shape(w_iter)[0]))
    displ_bot_root = np.zeros((50,200,np.shape(w_iter)[0]))
    c_all = np.zeros((size, np.shape(w_iter)[0]))
    
    
    # Displacement Incrementation
    for wp in w_iter: 
        print(f'------------ wp = {wp:.3f} ---------------')
        
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
        print(f'fext max {np.max(fext):.3e}')
        
        if hasattr(assy, "del_d"):
            print('has attr')
            fext_conn_force = assy.calc_fext_dmg(conn, c=c, prev_max_del_d=assy.del_d)
        else:
            fext_conn_force = assy.calc_fext_dmg(conn, c=None, prev_max_del_d=None)
        # Adding the contribution of fext due to connection to the original fext
        fext += fext_conn_force
        print(f'fext max {np.max(fext):.3e}')
            
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
            # Doing this to avoid recalc kc_conn and help pass the correct k_i
            kC_conn = assy.get_kC_conn(c=ci)
            # Inital fint (0 bec c=0)
            fint = np.asarray(assy.calc_fint(c=ci, kC_conn=kC_conn))
            # Residual with disp terms
            Ri = fint - fext + kCp*ci
            kT = assy.calc_kT(c=ci, kC_conn=kC_conn)
        k0 = kT + kCp
        
        print(f'kT {np.max(kT):.3e} kcp {np.max(kCp):.3e} k0 {np.max(k0):.3e}')
        
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
            
            # No need to check if del_d is an attribute or not as by now you already have c
            if disp_iter_no == 0:
                kw_tsl_temp, dmg_index_temp, del_d_temp, T_tsl_temp = assy.calc_traction(c=c, pA=p_top, no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, 
                                   tsl_type=tsl_type, prev_max_del_d=None, k_i=k_i)
            else:
                kw_tsl_temp, dmg_index_temp, del_d_temp, T_tsl_temp = assy.calc_traction(c=c, pA=p_top, no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, 
                                   tsl_type=tsl_type, prev_max_del_d=assy.del_d, k_i=k_i)
            
            print(f'max del_d {np.max(del_d_temp)}')
            
            crisfield_test = scaling(Ri, D)/max(scaling(fext, D), scaling(fint + kCp*c, D)) # + kcp*c bec its part of the internal force vector
            # print(f'    crisfield {crisfield_test:.4f}')
            if crisfield_test < epsilon:
                break
            
            count += 1
            kT = assy.calc_kT(c=c, kC_conn=kC_conn) 
            k0 = kT + kCp
            ci = c.copy()
            
            if count > 1000:
                print('Unconverged Results !!!!!!!!!!!!!!!!!!!')
                # return None, dmg_index, del_d, kw_tsl
                if True:
                    np.save(f'dmg_index_{filename}', dmg_index)
                    np.save(f'del_d_{filename}', del_d)
                    np.save(f'kw_tsl_{filename}', kw_tsl)
                    np.save(f'force_intgn_{filename}', force_intgn)
                    # np.save(f'displ_top_root_{filename}', displ_top_root)
                    # np.save(f'displ_bot_root_{filename}', displ_bot_root)
                    np.save(f'c_all_{filename}', c_all)
                raise RuntimeError('NR didnt converged :(') 
        
        
        # No need to check if del_d is an attribute or not as by now you already have c
        if disp_iter_no == 0:
            kw_tsl[:,:,disp_iter_no], dmg_index[:,:,disp_iter_no], del_d[:,:,disp_iter_no],
            T_tsl = assy.calc_traction(c=c, pA=p_top, no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, 
                               tsl_type=tsl_type, prev_max_del_d=None, k_i=k_i)
        else:
            kw_tsl[:,:,disp_iter_no], dmg_index[:,:,disp_iter_no], del_d[:,:,disp_iter_no],
            T_tsl = assy.calc_traction(c=c, pA=p_top, no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, 
                               tsl_type=tsl_type, prev_max_del_d=assy.del_d, k_i=k_i)
            
        c_all[:,disp_iter_no] = c
            
        # Update max del_d AFTER a converged NR iteration
        assy.update_max_del_d(curr_max_del_d=del_d[:,:,disp_iter_no])
        
        # kw_tsl[:,:,disp_iter_no], dmg_index[:,:,disp_iter_no], del_d[:,:,disp_iter_no] = assy.calc_k_dmg(c=c, pA=p_top, pB=p_bot, no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, tsl_type=tsl_type)
        print(f' ki = {k_i:.3e} -- wp = {wp:.3f} -- max DI {np.max(dmg_index[:,:,disp_iter_no]):.4f}')
        sys.stdout.flush()
        
        
        # Force - TEMP - CHECK LATER AND EDIT/REMOVE
        if True:
            force_intgn[disp_iter_no, 0] = wp
            force_intgn[disp_iter_no, 1] = assy.force_out_plane(c, group=None, eval_panel=disp_panel, x_cte_force=None, y_cte_force=None,
                      gridx=100, gridy=50, NLterms=True, no_x_gauss=128, no_y_gauss=128)
        else: 
            force_intgn = None
        
        # Calc displ of top and bottom panels at each increment
        if True:
            res_pan_top = assy.calc_results(c=c, eval_panel=top3, vec='w', 
                                    no_x_gauss=200, no_y_gauss=50)
            # displ_top_root[:,:,disp_iter_no] = res_pan_top['w'][0]
            res_pan_top_w = res_pan_top['w']
            print(f'Displ top {np.min(res_pan_top_w)} {np.max(res_pan_top_w)}')
        else:
            res_pan_top = None
            res_pan_bot = None
        
        disp_iter_no += 1
        print()
        
        if np.all(dmg_index == 1):
            print('Cohesive Zone has failed')
            break
        
        
    # ------------------ SAVING VARIABLES --------------------
    if True:
        np.save(f'dmg_index_{filename}', dmg_index)
        np.save(f'del_d_{filename}', del_d)
        np.save(f'kw_tsl_{filename}', kw_tsl)
        np.save(f'force_intgn_{filename}', force_intgn)
        # np.save(f'displ_top_root_{filename}', displ_top_root)
        # np.save(f'displ_bot_root_{filename}', displ_bot_root)
        np.save(f'c_all_{filename}', c_all)
        
        
    
    # ------------------ RESULTS AND POST PROCESSING --------------------
    
    c0 = c.copy()

    generate_plots = True
    
    final_res = None
    # Plotting results
    if True:
        for vec in ['w']:#, 'Mxx']:#, 'Myy', 'Mxy']:#, 'Nxx', 'Nyy']:
            res_bot = assy.calc_results(c=c0, group='bot', vec=vec, no_x_gauss=None, no_y_gauss=None)
            res_top = assy.calc_results(c=c0, group='top', vec=vec, no_x_gauss=None, no_y_gauss=None)
            # vecmin = min(np.min(np.array(res_top[vec])))
            # vecmax = max(np.max(np.array(res_top[vec])))
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
                # print(f'Global BOT {vec} :: {np.min(np.array(res_bot[vec])):.3f}  {np.max(np.array(res_bot[vec])):.3f}')
                # print(res_bot[vec][1][:,-1]) # disp at the tip
                final_res = np.min(np.array(res_top[vec]))
            
            if generate_plots:
                # if vec == 'w':
                if True:
                    # assy.plot(c=c0, group='bot', vec=vec, filename='test_dcb_before_opening_bot_tsl.png', show_boundaries=True,
                    #                             colorbar=True, res = res_bot, vecmax=vecmax, vecmin=vecmin, display_zero=True)
                    
                    assy.plot(c=c0, group='top', vec=vec, filename='test_dcb_before_opening_top_tsl.png', show_boundaries=True,
                                              colorbar=True, res = res_top,display_zero=True)
            
            # Open images
            if generate_plots:
                img_popup('test_dcb_before_opening_top_tsl.png',1, f"{vec} top")
                # img_popup('test_dcb_before_opening_bot_tsl.png',2, f"{vec} bot")
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
        
    return dmg_index, del_d, kw_tsl, force_intgn, c_all



def postprocess_results_damage(c_all, no_pan, no_terms, filename='', k_i=None, tau_o=None, no_x_gauss=None, no_y_gauss=None, w_iter_no_pts=None):

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
           dict(p1=top1, p2=bot1, func='SB_TSL', tsl_type = 'bilinear', no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, k_i=k_i, tau_o=tau_o)
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
           dict(p1=top1, p2=bot1, func='SB_TSL', tsl_type = 'bilinear', no_x_gauss=60, no_y_gauss=30, k_i=k_i, tau_o=tau_o)
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
    w_iter = np.unique(np.concatenate((np.linspace(0.01,3,int(0.2*w_iter_no_pts)), np.linspace(3,6,int(0.4*w_iter_no_pts)), 
                                       np.linspace(6,8,int(0.4*w_iter_no_pts)))))
    # w_iter = np.linspace(0.01,8,100)
    # w_iter = [0.01, 2]
    
    dmg_index = np.zeros((no_y_gauss,no_x_gauss,np.shape(w_iter)[0]))
    del_d = np.zeros((no_y_gauss,no_x_gauss,np.shape(w_iter)[0]))
    kw_tsl = np.zeros((no_y_gauss,no_x_gauss,np.shape(w_iter)[0]))
    force_intgn = np.zeros((np.shape(w_iter)[0], 2))
    force_intgn_dmg = np.zeros((np.shape(w_iter)[0], 2))
    energy_dissp = np.zeros((np.shape(w_iter)[0], 2))
    
    
    # Displacement Incrementation
    for wp in w_iter: 
        
        c = c_all[:,disp_iter_no]

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
            
            
            
        # Update max del_d AFTER a converged NR iteration
        assy.update_max_del_d(curr_max_del_d=del_d[:,:,disp_iter_no])
        
        print(f'            {filename} -- wp={wp:.3f} -- max DI {np.max(dmg_index[:,:,disp_iter_no]):.4f}')
        sys.stdout.flush()
        
        
        # Force - TEMP - CHECK LATER AND EDIT/REMOVE
        if False:
            force_intgn[disp_iter_no, 0] = wp
            force_intgn[disp_iter_no, 1] = assy.force_out_plane(c, group=None, eval_panel=disp_panel, x_cte_force=None, y_cte_force=None,
                      gridx=100, gridy=50, NLterms=True, no_x_gauss=128, no_y_gauss=128)
        else: 
            force_intgn = None
        
        # Force by area integral of traction in the damaged region
        if False:
            force_intgn_dmg[disp_iter_no, 0] = wp
            force_intgn_dmg[disp_iter_no, 1] = assy.force_out_plane_damage(conn=conn, c=c)
        
        # Calc displ of top and bottom panels at each increment
        if False:
            res_pan_top = assy.calc_results(c=c, eval_panel=top1, vec='w', 
                                    no_x_gauss=200, no_y_gauss=50)
            res_pan_bot = assy.calc_results(c=c, eval_panel=bot1, vec='w', 
                                    no_x_gauss=200, no_y_gauss=50)
            displ_top_root[:,:,disp_iter_no] = res_pan_top['w'][0]
            displ_bot_root[:,:,disp_iter_no] = res_pan_bot['w'][0]
        else:
            displ_top_root = None
            displ_bot_root = None
        
        # Calculate energy dissipation
        if True:
            if disp_iter_no != 0: # Bec there is no previous step
                energy_dissp[disp_iter_no, 0] = wp
                energy_dissp[disp_iter_no, 1] = assy.calc_energy_dissipation(kw_tsl_j=kw_tsl[:,:,disp_iter_no], 
                                    kw_tsl_j_1=kw_tsl[:,:,disp_iter_no-1], 
                                    del_d_j=del_d[:,:,disp_iter_no], del_d_j_1=del_d[:,:,disp_iter_no-1], 
                                    tau_o=tau_o, k_i=k_i, no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss)
        
        disp_iter_no += 1
        print()
        
        
        
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
    
    
        
    return dmg_index, del_d, kw_tsl, force_intgn, displ_top_root, displ_bot_root, energy_dissp


def calc_area_tsl_curve(kw_tsl, del_d):
    '''
    Calculates the area under the traction separation curve history for each integration point across all load steps
    Returns
    -------
    None.

    '''
    
    tau = np.multiply(kw_tsl, del_d)
    
    




if __name__ == "__main__":
    
    if False:
        test_dcb_damage_prop_fext_conn(no_pan = 3, no_terms=8, plies=15, k_i=1e4)
    
    # Post process results
    if False:
        dmg_index, del_d, kw_tsl, force_intgn, displ_top_root, displ_bot_root, energy_dissp = postprocess_results_damage(c_all=c_all_p3_m15_10_ki1e4_tauo87_nx60_ny30_wpts45, no_pan=3, 
                   no_terms=15, filename='p3_m15_10_ki1e4_tauo87_nx60_ny30_wpts45', k_i=1e4, 
                   tau_o=67, no_x_gauss=60, no_y_gauss=30, w_iter_no_pts=45)
