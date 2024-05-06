# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:57:33 2024

@author: Nathan
"""

def test_dcb_damage_prop_modified_k(no_terms, plies):

    '''
        Damage propagation from a DCB with a precrack
        
        Code for 2 panels might not be right
            
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
    a1 = 0.4*a
    a2 = 0.5*a

    #others
    m = no_terms
    n = no_terms
    # print(f'no terms : {m}')

    # simple_layup = [+45, -45]*plies + [0, 90]*plies
    simple_layup = [0]*plies
    # simple_layup = [0, 0]*10 + [0, 0]*10
    simple_layup += simple_layup[::-1]
    # simple_layup += simple_layup[::-1]
    print('plies ',np.shape(simple_layup)[0])

    laminaprop = (E1, E2, nu12, G12, G12, G12)
    
    no_pan = 3
    
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
           # dict(p1=top1, p2=bot1, func='SB'), #'_TSL', tsl_type = 'linear'), 
           dict(p1=top1, p2=bot1, func='SB_TSL', tsl_type = 'bilinear', no_x_gauss=50, no_y_gauss=50)
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

    kT = assy.calc_kT()
    size = kT.shape[0]
        
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
        
        k0 = kT + kCp
        ci = solve(k0, fext)
        size = k0.shape[0]
    else:
        ci = np.zeros(size) # Only used to calc initial fint
        raise RuntimeError ('Matrix SIZE needs to be provided')
        
    # -------------------- INCREMENTATION --------------------------
    wp_max = 20 # [mm]
    no_iter_disp = 20
    
    disp_iter_no = 0
    
    # Finding info of the connection
    for conn_list in conn:
        if conn_list['func'] == 'SB_TSL':
            no_x_gauss = conn_list['no_x_gauss']
            no_y_gauss = conn_list['no_y_gauss']
            tsl_type = conn_list['tsl_type']
            pA = conn_list['p1']
            pB = conn_list['p2']
            break
        
    # Initilaize mat to store results
    dmg_index = np.zeros((no_y_gauss,no_x_gauss,no_iter_disp))
    del_d = np.zeros((no_y_gauss,no_x_gauss,no_iter_disp))
    kw_tsl = np.zeros((no_y_gauss,no_x_gauss,no_iter_disp))
    
    # Displacement Incrementation
    for wp in np.linspace(0.01, wp_max, no_iter_disp): 
        print('wp', wp)
        
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

        # Inital guess ci and increment dc (same size as fext)
        dc = np.zeros_like(fext)
        
        # Inital fint (0 bec c=0)
        fint = assy.calc_fint(c=ci)
        # Residual with disp terms
        Ri = fint - fext + kCp*ci
        
        print(disp_iter_no)
        if disp_iter_no != 0:
            kT = assy.calc_kT(c=c) 
        
        epsilon = 1.e-8 # Convergence criteria
        D = k0.diagonal() # For convergence - calc at beginning of load increment
        
        count = 0 # Tracks number of NR iterations 
        rerun = True
        fact_dc = 1
        
        while rerun:
            # Modified Newton Raphson Iteration
            while True:
                dc = solve(k0, -Ri, silent=True)
                c = ci + fact_dc*dc
                fint = np.asarray(assy.calc_fint(c=c))
                Ri = fint - fext + kCp*c
                # print(f'Ri {np.linalg.norm(Ri)}')
                crisfield_test = scaling(Ri, D)/max(scaling(fext, D), scaling(fint, D))
                if crisfield_test < epsilon:
                    # print(f'         Ri {np.linalg.norm(Ri)}')
                    # print()
                    break
                # print(crisfield_test)
                # print()
                count += 1
                kT = assy.calc_kT(c=c) 
                # print(f'kC {np.max(kC)}')
                k0 = kT + kCp
                ci = c.copy()
                if count > 500:
                    raise RuntimeError('NR not converged :(')
            
            k_dmg_iter, dmg_index_iter, del_d_iter = assy.calc_k_dmg(c=c, pA=pA, pB=pB, no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, tsl_type=tsl_type)
            # print(f'           max dmg_index {np.max(dmg_index_iter)}')
            # print(f'           max del_d {np.max(del_d_iter)}')
            print(f'          min del_d {np.min(del_d_iter)}')
            # print(f'           max kw_tsl {np.max(k_dmg_iter):.2e}')
            if np.max(k_dmg_iter) < 1e5:
                rerun = False
                # kT = assy.calc_kT(c=c, kw_tsl=k_dmg_iter)
                # k0 = kT + kCp
                fact_dc = 0.1
            
        kw_tsl[:,:,disp_iter_no], dmg_index[:,:,disp_iter_no], del_d[:,:,disp_iter_no] = assy.calc_k_dmg(c=c, pA=pA, pB=pB, no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, tsl_type=tsl_type)
        print(f'        max dmg_index {np.max(dmg_index[:,:,disp_iter_no])}')
        print(f'        max del_d {np.max(del_d[:,:,disp_iter_no])}')
        print(f'       min del_d {np.min(del_d[:,:,disp_iter_no])}')
        print(f'        max kw_tsl {np.max(kw_tsl[:,:,disp_iter_no]):.2e}')
        disp_iter_no += 1
        
        if np.all(dmg_index == 1):
            print('Cohesive Zone has failed')
            break
        
        
        
    c0 = c.copy()

    # ------------------ RESULTS AND POST PROCESSING --------------------
    
    generate_plots = True
    
    # Plotting results
    if True:
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
    
    # Test for force
    # Panel at which the disp is applied
    if no_pan == 2:
        force_panel = top2
    if no_pan == 3:
        force_panel = top3
    if False:
        vec = 'Fxx'
        res_bot = assy.calc_results(c=c0, group='bot', vec=vec, no_x_gauss=50, no_y_gauss=60,
                                eval_panel=force_panel, x_cte_force=force_panel.a)
        print(res_bot)
    
    animate = True
    if animate:    
        for animate_var in ["dmg_index", "del_d", 'kw_tsl']:
            frames = [] # for storing the generated images
            fig = plt.figure()
            for i in range(np.shape(locals()[animate_var])[2]):
                frames.append([plt.imshow(locals()[animate_var][:,:,i],animated=True)])
            
            ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                            repeat_delay=1000)
            plt.colorbar()
            ani.save(f'{animate_var}.gif', writer='imagemagick')
            
            # ani.save('movie.mp4')
            # plt.show()
        
    return final_res, dmg_index, del_d, kw_tsl




# %% Amination - other methods 
# Artist animation
if False:
    for animate_var in ["dmg_index"]:#, "del_d", 'kw_tsl']:
        frames = [] # for storing the generated images
        fig = plt.figure()
        plt.title(animate_var)
        for i in range(np.shape(globals()[animate_var])[2]):
            # plt.title(f'{animate_var}')
            frames.append([plt.imshow(globals()[animate_var][:,:,i],animated=True), 
                           plt.annotate(f"Step {i}", xy=(5,0)), 
                           plt.colorbar()])
        
        ani = animation.ArtistAnimation(fig, frames, interval=300, blit=True,
                                        repeat_delay=5000)
        # plt.colorbar()
        FFwriter = animation.FFMpegWriter(fps=10)
        ani.save(f'{animate_var}.mp4', writer=FFwriter)

if False:
    for animate_var in ["dmg_index"]:#, "del_d", 'kw_tsl']:
        plt.figure()
        ax = plt.gca()
        ticks_low_dmg = np.linspace(np.min(globals()[animate_var]), np.max(globals()[animate_var]), 10)
        ticklabels_low_dmg = [('%1.4f' % label) for label in ticks_low_dmg]
        ticks_high_dmg = np.linspace(0.85, 1, 10)
        ticklabels_high_dmg = [('%1.4f' % label) for label in ticks_high_dmg]
        for disp_step in range(np.shape(globals()[animate_var])[2]):
            plt.title(animate_var)
            if np.min(globals()[animate_var]) > 0.85:
                ticks = ticks_high_dmg
                ticklabels = ticklabels_high_dmg
                vmin = 0.85
                vmax = 1.00
            else:
                ticks = ticks_low_dmg
                ticklabels = ticklabels_low_dmg
                vmin = 0.00
                vmax = 1.00
            plt.contourf(globals()[animate_var][:,:,disp_step], vmin=vmin, vmax=vmax)
            cbar = plt.colorbar()
            # cbar.set_ticks(ticks)
            # cbar.set_ticklabels(ticklabels)
            plt.pause(1e-9)
        # fig.savefig(f'{animate_var}.mp4', transparent=True,
        #             bbox_inches='tight', pad_inches=0.05)
