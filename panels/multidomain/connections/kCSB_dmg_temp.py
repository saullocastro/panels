# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:01:33 2024

@author: Nathan
"""


def kcrack_term2_11(object p_top, object p_bot, int size, int row0, int col0, int no_x_gauss, int no_y_gauss,
              double k_o, double del_o, double del_f, 
              double [:,::1] del_d_i_1, double [:,::1] del_d_i, double [:,::1] dmg_index):

    cdef int m_top, n_top, m_bot, n_bot
    cdef int i1, k2, j1, l2
    cdef double xi, eta, weight
    cdef int ptx, pty, c, row, col
    cdef double a_top, b_top
    
    cdef double x1u_top, x1ur_top, x2u_top, x2ur_top, x1u_bot, x1ur_bot, x2u_bot, x2ur_bot
    cdef double x1v_top, x1vr_top, x2v_top, x2vr_top, x1v_bot, x1vr_bot, x2v_bot, x2vr_bot
    cdef double x1w_top, x1wr_top, x2w_top, x2wr_top, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot
    cdef double y1u_top, y1ur_top, y2u_top, y2ur_top, y1u_bot, y1ur_bot, y2u_bot, y2ur_bot
    cdef double y1v_top, y1vr_top, y2v_top, y2vr_top, y1v_bot, y1vr_bot, y2v_bot, y2vr_bot
    cdef double y1w_top, y1wr_top, y2w_top, y2wr_top, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot
    
    cdef double fAw_top, gAw_top
    cdef double fBw_bot, gBw_bot
    
    cdef long [:] k_crack12r, k_crack12c
    cdef double [:] k_crack12v
    
    cdef double del_d_i_1_iter, del_d_i_iter
    
    cdef double [:] weights_xi, weights_eta, xis, etas
    
    m_top = p_top.m
    n_top = p_top.n
    m_bot = p_bot.m
    n_bot = p_bot.n
    
    a_top = p_top.a
    b_top = p_top.b
    
    # Panel top (ends in top)
    x1u_top = p_top.x1u ; x1ur_top = p_top.x1ur ; x2u_top = p_top.x2u ; x2ur_top = p_top.x2ur
    x1v_top = p_top.x1v ; x1vr_top = p_top.x1vr ; x2v_top = p_top.x2v ; x2vr_top = p_top.x2vr
    x1w_top = p_top.x1w ; x1wr_top = p_top.x1wr ; x2w_top = p_top.x2w ; x2wr_top = p_top.x2wr
    y1u_top = p_top.y1u ; y1ur_top = p_top.y1ur ; y2u_top = p_top.y2u ; y2ur_top = p_top.y2ur
    y1v_top = p_top.y1v ; y1vr_top = p_top.y1vr ; y2v_top = p_top.y2v ; y2vr_top = p_top.y2vr
    y1w_top = p_top.y1w ; y1wr_top = p_top.y1wr ; y2w_top = p_top.y2w ; y2wr_top = p_top.y2wr

    # Panel bot (ends in bot)
    x1u_bot = p_bot.x1u ; x1ur_bot = p_bot.x1ur ; x2u_bot = p_bot.x2u ; x2ur_bot = p_bot.x2ur
    x1v_bot = p_bot.x1v ; x1vr_bot = p_bot.x1vr ; x2v_bot = p_bot.x2v ; x2vr_bot = p_bot.x2vr
    x1w_bot = p_bot.x1w ; x1wr_bot = p_bot.x1wr ; x2w_bot = p_bot.x2w ; x2wr_bot = p_bot.x2wr
    y1u_bot = p_bot.y1u ; y1ur_bot = p_bot.y1ur ; y2u_bot = p_bot.y2u ; y2ur_bot = p_bot.y2ur
    y1v_bot = p_bot.y1v ; y1vr_bot = p_bot.y1vr ; y2v_bot = p_bot.y2v ; y2vr_bot = p_bot.y2vr
    y1w_bot = p_bot.y1w ; y1wr_bot = p_bot.y1wr ; y2w_bot = p_bot.y2w ; y2wr_bot = p_bot.y2wr
    
    # Initializing gauss points, weights
    xis = np.zeros(no_x_gauss, dtype=DOUBLE)
    weights_xi = np.zeros(no_x_gauss, dtype=DOUBLE)
    etas = np.zeros(no_y_gauss, dtype=DOUBLE)
    weights_eta = np.zeros(no_y_gauss, dtype=DOUBLE)

    # Calc gauss points and weights
    leggauss_quad(no_x_gauss, &xis[0], &weights_xi[0])
    leggauss_quad(no_y_gauss, &etas[0], &weights_eta[0])
    
    # Dimension of the number of values that need to be stored
    fdim = 1*m_bot*n_bot*m_top*n_top    # 1 bec only 1 term is being added in the for loops
    
    k_crack12r = np.zeros((fdim,), dtype=INT)
    k_crack12c = np.zeros((fdim,), dtype=INT)
    k_crack12v = np.zeros((fdim,), dtype=DOUBLE)
    
    with nogil:
        
        for ptx in range(no_x_gauss):
            for pty in range(no_y_gauss):
                # Takes the correct index instead of the location
                xi = xis[ptx]
                eta = etas[pty]

                weight = weights_xi[ptx] * weights_eta[pty]
                
                del_d_i_1_iter = del_d_i_1[pty, ptx]
                del_d_i_iter = del_d_i[pty, ptx]
                
                if del_d_i_iter == 0:
                    continue
                
                if dmg_index[pty, ptx] == 0 or dmg_index[pty, ptx] == 1:
                    continue
                
                c = -1
                
                for i1 in range(m_top):
                    fAw_top = f(i1, xi, x1w_top, x1wr_top, x2w_top, x2wr_top)
                    for k1 in range(m_top):
                        fBw_top = f(k1, xi, x1w_top, x1wr_top, x2w_top, x2wr_top)
                        for o1 in range(m_top):
                            fCw_top = f(o1, xi, x1w_top, x1wr_top, x2w_top, x2wr_top)
                            for j1 in range(n_top):
                                gAw_top = f(j1, eta, y1w_top, y1wr_top, y2w_top, y2wr_top)
                                for l1 in range(n_top):
                                    gBw_top = f(l1, eta, y1w_top, y1wr_top, y2w_top, y2wr_top)
                                    for p1 in range(n_top):
                                        gCw_top = f(p1, eta, y1w_top, y1wr_top, y2w_top, y2wr_top)
                
                                        # WHAT INDICES???
                                        row = row0 + DOF*(j1*m_top + i1)
                                        col = col0 + DOF*(l1*m_top + k1)
                            
                
                                        #NOTE No symmetry - 12
                                        if row > col:
                                            continue
                
                                        c += 1
                                        k_crack_term2_11r[c] = row+2
                                        k_crack_term2_11c[c] = col+2
                                        k_crack_term2_11v[c] += a_1*b_1*c1w*del_f*del_i_1*del_o*fAw_top*fBw_top*fCw_top*gAw_top*gBw_top*gCw_top*k_o*weight/(4*(del_i*del_i*del_i)*(del_f - del_o))

    kcrack_term2_11 = coo_matrix((k_crack12v, (k_crack12r, k_crack12c)), shape=(size, size))
    # Builds a matrix of size = size x size (so complete size of global MD) and populates it with the data in ..v 
        # where the rows and cols where that data should go are specified by ..r, ...c
    # This way its at the correct positions in the global MD matrix as row and col are the starting indices of the 
        # submatrices in the global MD matrix

    return kcrack_term2_11



def k_crack_term2_partA_11(object p_top, int size, int row0, int col0, int no_x_gauss, int no_y_gauss,
              double k_o, double del_o, double del_f, 
              double [:,::1] del_d_i_1, double [:,::1] del_d_i, double [:,::1] dmg_index):

    cdef int m_top, n_top
    cdef int i1, k1, j1, l1
    cdef double xi, eta, weight
    cdef int ptx, pty, c, row, col
    cdef double a_top, b_top
    
    cdef double x1u_top, x1ur_top, x2u_top, x2ur_top
    cdef double x1v_top, x1vr_top, x2v_top, x2vr_top
    cdef double x1w_top, x1wr_top, x2w_top, x2wr_top
    cdef double y1u_top, y1ur_top, y2u_top, y2ur_top
    cdef double y1v_top, y1vr_top, y2v_top, y2vr_top
    cdef double y1w_top, y1wr_top, y2w_top, y2wr_top
    
    cdef double fAw_top, fBw_top 
    cdef double gAw_top, gBw_top
    
    cdef long [:] k_crack_term2_partA_11r, k_crack_term2_partA_11c
    cdef double [:] k_crack_term2_partA_11v
    
    cdef double del_d_i_1_iter, del_d_i_iter
    
    cdef double [:] weights_xi, weights_eta, xis, etas
    
    m_top = p_top.m
    n_top = p_top.n
    
    a_top = p_top.a
    b_top = p_top.b
    
    # Panel top (ends in top)
    x1u_top = p_top.x1u ; x1ur_top = p_top.x1ur ; x2u_top = p_top.x2u ; x2ur_top = p_top.x2ur
    x1v_top = p_top.x1v ; x1vr_top = p_top.x1vr ; x2v_top = p_top.x2v ; x2vr_top = p_top.x2vr
    x1w_top = p_top.x1w ; x1wr_top = p_top.x1wr ; x2w_top = p_top.x2w ; x2wr_top = p_top.x2wr
    y1u_top = p_top.y1u ; y1ur_top = p_top.y1ur ; y2u_top = p_top.y2u ; y2ur_top = p_top.y2ur
    y1v_top = p_top.y1v ; y1vr_top = p_top.y1vr ; y2v_top = p_top.y2v ; y2vr_top = p_top.y2vr
    y1w_top = p_top.y1w ; y1wr_top = p_top.y1wr ; y2w_top = p_top.y2w ; y2wr_top = p_top.y2wr
    
    # Initializing gauss points, weights
    xis = np.zeros(no_x_gauss, dtype=DOUBLE)
    weights_xi = np.zeros(no_x_gauss, dtype=DOUBLE)
    etas = np.zeros(no_y_gauss, dtype=DOUBLE)
    weights_eta = np.zeros(no_y_gauss, dtype=DOUBLE)

    # Calc gauss points and weights
    leggauss_quad(no_x_gauss, &xis[0], &weights_xi[0])
    leggauss_quad(no_y_gauss, &etas[0], &weights_eta[0])
    
    # Dimension of the number of values that need to be stored
    fdim = 1*m_top*n_top*m_top*n_top    # 1 bec only 1 term is being added in the for loops
    
    k_crack_term2_partA_11r = np.zeros((fdim,), dtype=INT)
    k_crack_term2_partA_11c = np.zeros((fdim,), dtype=INT)
    k_crack_term2_partA_11v = np.zeros((fdim,), dtype=DOUBLE)
    
    with nogil:
        
        for ptx in range(no_x_gauss):
            for pty in range(no_y_gauss):
                # Takes the correct index instead of the location
                xi = xis[ptx]
                eta = etas[pty]

                weight = weights_xi[ptx] * weights_eta[pty]
                
                del_d_i_1_iter = del_d_i_1[pty, ptx]
                del_d_i_iter = del_d_i[pty, ptx]
                
                # CHANGE TO ABS < CRITERIA - For now its fine bec its being manually set to 0
                if del_d_i_iter == 0:
                    continue
                
                if dmg_index[pty, ptx] == 0 or dmg_index[pty, ptx] == 1:
                    continue
                
            
                
                c = -1
                for i1 in range(m_top):
                    fAw_top = f(i1, xi, x1w_top, x1wr_top, x2w_top, x2wr_top)
                    
                    for k1 in range(m_top):
                        fBw_top = f(k1, xi, x1w_top, x1wr_top, x2w_top, x2wr_top)
                        
                        for j1 in range(n_top): 
                            gAw_top = f(j1, eta, y1w_top, y1wr_top, y2w_top, y2wr_top)
                                    
                            for l1 in range(n_top):
                                gBw_top = f(l1, eta, y1w_top, y1wr_top, y2w_top, y2wr_top)
        
                                row = row0 + DOF*(j1*m_top + i1)
                                col = col0 + DOF*(l1*m_top + k1)
        
                                #NOTE symmetry - 11
                                if row > col:
                                    continue
        
                                c += 1
                                k_crack_term2_partA_11r[c] = row+2
                                k_crack_term2_partA_11c[c] = col+2
                                # k_crack_term2_partA_11v[c] += -a_top*b_top*del_o*fAw_top*fBw_top*gAw_top*gBw_top*k_o*weight/(4*(del_f - del_o)) - # OLD WRONG
                                k_crack_term2_partA_11v[c] += a_top*b_top*del_f*del_d_i_1_iter*del_o*fAw_top*fBw_top*gAw_top*gBw_top*k_o*weight/(4*(del_d_i_iter*del_d_i_iter*del_d_i_iter)*(del_f - del_o))

                                # with gil:
                                #     print(k_crack_term2_partA_11v[c])


    k_crack_term2_partA_11 = coo_matrix((k_crack_term2_partA_11v, (k_crack_term2_partA_11r, k_crack_term2_partA_11c)), shape=(size, size))
    # Builds a matrix of size = size x size (so complete size of global MD) and populates it with the data in ..v 
        # where the rows and cols where that data should go are specified by ..r, ...c
    # This way its at the correct positions in the global MD matrix as row and col are the starting indices of the 
        # submatrices in the global MD matrix

    return k_crack_term2_partA_11


def k_crack_term2_partA_12(object p_top, object p_bot, int size, int row0, int col0, int no_x_gauss, int no_y_gauss,
              double k_o, double del_o, double del_f, 
              double [:,::1] del_d_i_1, double [:,::1] del_d_i, double [:,::1] dmg_index):

    cdef int m_top, n_top, m_bot, n_bot
    cdef int i1, k2, j1, l2
    cdef double xi, eta, weight
    cdef int ptx, pty, c, row, col
    cdef double a_top, b_top
    
    cdef double x1u_top, x1ur_top, x2u_top, x2ur_top, x1u_bot, x1ur_bot, x2u_bot, x2ur_bot
    cdef double x1v_top, x1vr_top, x2v_top, x2vr_top, x1v_bot, x1vr_bot, x2v_bot, x2vr_bot
    cdef double x1w_top, x1wr_top, x2w_top, x2wr_top, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot
    cdef double y1u_top, y1ur_top, y2u_top, y2ur_top, y1u_bot, y1ur_bot, y2u_bot, y2ur_bot
    cdef double y1v_top, y1vr_top, y2v_top, y2vr_top, y1v_bot, y1vr_bot, y2v_bot, y2vr_bot
    cdef double y1w_top, y1wr_top, y2w_top, y2wr_top, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot
    
    cdef double fAw_top, gAw_top
    cdef double fBw_bot, gBw_bot
    
    cdef long [:] k_crack_term2_partA_12r, k_crack_term2_partA_12c
    cdef double [:] k_crack_term2_partA_12v
    
    cdef double del_d_i_1_iter, del_d_i_iter
    
    cdef double [:] weights_xi, weights_eta, xis, etas
    
    m_top = p_top.m
    n_top = p_top.n
    m_bot = p_bot.m
    n_bot = p_bot.n
    
    a_top = p_top.a
    b_top = p_top.b
    
    # Panel top (ends in top)
    x1u_top = p_top.x1u ; x1ur_top = p_top.x1ur ; x2u_top = p_top.x2u ; x2ur_top = p_top.x2ur
    x1v_top = p_top.x1v ; x1vr_top = p_top.x1vr ; x2v_top = p_top.x2v ; x2vr_top = p_top.x2vr
    x1w_top = p_top.x1w ; x1wr_top = p_top.x1wr ; x2w_top = p_top.x2w ; x2wr_top = p_top.x2wr
    y1u_top = p_top.y1u ; y1ur_top = p_top.y1ur ; y2u_top = p_top.y2u ; y2ur_top = p_top.y2ur
    y1v_top = p_top.y1v ; y1vr_top = p_top.y1vr ; y2v_top = p_top.y2v ; y2vr_top = p_top.y2vr
    y1w_top = p_top.y1w ; y1wr_top = p_top.y1wr ; y2w_top = p_top.y2w ; y2wr_top = p_top.y2wr

    # Panel bot (ends in bot)
    x1u_bot = p_bot.x1u ; x1ur_bot = p_bot.x1ur ; x2u_bot = p_bot.x2u ; x2ur_bot = p_bot.x2ur
    x1v_bot = p_bot.x1v ; x1vr_bot = p_bot.x1vr ; x2v_bot = p_bot.x2v ; x2vr_bot = p_bot.x2vr
    x1w_bot = p_bot.x1w ; x1wr_bot = p_bot.x1wr ; x2w_bot = p_bot.x2w ; x2wr_bot = p_bot.x2wr
    y1u_bot = p_bot.y1u ; y1ur_bot = p_bot.y1ur ; y2u_bot = p_bot.y2u ; y2ur_bot = p_bot.y2ur
    y1v_bot = p_bot.y1v ; y1vr_bot = p_bot.y1vr ; y2v_bot = p_bot.y2v ; y2vr_bot = p_bot.y2vr
    y1w_bot = p_bot.y1w ; y1wr_bot = p_bot.y1wr ; y2w_bot = p_bot.y2w ; y2wr_bot = p_bot.y2wr
    
    # Initializing gauss points, weights
    xis = np.zeros(no_x_gauss, dtype=DOUBLE)
    weights_xi = np.zeros(no_x_gauss, dtype=DOUBLE)
    etas = np.zeros(no_y_gauss, dtype=DOUBLE)
    weights_eta = np.zeros(no_y_gauss, dtype=DOUBLE)

    # Calc gauss points and weights
    leggauss_quad(no_x_gauss, &xis[0], &weights_xi[0])
    leggauss_quad(no_y_gauss, &etas[0], &weights_eta[0])
    
    # Dimension of the number of values that need to be stored
    fdim = 1*m_bot*n_bot*m_top*n_top    # 1 bec only 1 term is being added in the for loops
    
    k_crack_term2_partA_12r = np.zeros((fdim,), dtype=INT)
    k_crack_term2_partA_12c = np.zeros((fdim,), dtype=INT)
    k_crack_term2_partA_12v = np.zeros((fdim,), dtype=DOUBLE)
    
    with nogil:
        
        for ptx in range(no_x_gauss):
            for pty in range(no_y_gauss):
                # Takes the correct index instead of the location
                xi = xis[ptx]
                eta = etas[pty]

                weight = weights_xi[ptx] * weights_eta[pty]
                
                del_d_i_1_iter = del_d_i_1[pty, ptx]
                del_d_i_iter = del_d_i[pty, ptx]
                
                if del_d_i_iter == 0:
                    continue
                
                if dmg_index[pty, ptx] == 0 or dmg_index[pty, ptx] == 1:
                    continue
                
                c = -1
                for i1 in range(m_top):
                    fAw_top = f(i1, xi, x1w_top, x1wr_top, x2w_top, x2wr_top)
                    
                    for k2 in range(m_bot):
                        fBw_bot = f(k2, xi, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot)
                        
                        for j1 in range(n_top):
                            gAw_top = f(j1, eta, y1w_top, y1wr_top, y1w_top, y1wr_top)
                                    
                            for l2 in range(n_bot):
                                gBw_bot = f(l2, eta, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot)
        
                                row = row0 + DOF*(j1*m_top + i1)
                                col = col0 + DOF*(l2*m_bot + k2)
        
                                #NOTE No symmetry - 12
                                # if row > col:
                                #     continue
        
                                c += 1
                                k_crack_term2_partA_12r[c] = row+2
                                k_crack_term2_partA_12c[c] = col+2
                                # k_crack_term2_partA_12v[c] += a_top*b_top*del_o*fAw_top*fBw_bot*gAw_top*gBw_bot*k_o*weight/(4*(del_f - del_o)) # OLD WRONG
                                k_crack_term2_partA_12v[c] += -a_top*b_top*del_f*del_d_i_1_iter*del_o*fAw_top*fBw_bot*gAw_top*gBw_bot*k_o*weight/(4*(del_d_i_iter*del_d_i_iter*del_d_i_iter)*(del_f - del_o))

    k_crack_term2_partA_12 = coo_matrix((k_crack_term2_partA_12v, (k_crack_term2_partA_12r, k_crack_term2_partA_12c)), shape=(size, size))
    # Builds a matrix of size = size x size (so complete size of global MD) and populates it with the data in ..v 
        # where the rows and cols where that data should go are specified by ..r, ...c
    # This way its at the correct positions in the global MD matrix as row and col are the starting indices of the 
        # submatrices in the global MD matrix

    return k_crack_term2_partA_12


def k_crack_term2_partA_22(object p_bot, int size, int row0, int col0, int no_x_gauss, int no_y_gauss,
              double k_o, double del_o, double del_f,
              double [:,::1] del_d_i_1, double [:,::1] del_d_i, double [:,::1] dmg_index):

    cdef int m_bot, n_bot
    cdef int i1, k1, j1, l1
    cdef double xi, eta, weight
    cdef int ptx, pty, c, row, col
    cdef double a_bot, b_bot
    
    cdef double x1u_bot, x1ur_bot, x2u_bot, x2ur_bot
    cdef double x1v_bot, x1vr_bot, x2v_bot, x2vr_bot
    cdef double x1w_bot, x1wr_bot, x2w_bot, x2wr_bot
    cdef double y1u_bot, y1ur_bot, y2u_bot, y2ur_bot
    cdef double y1v_bot, y1vr_bot, y2v_bot, y2vr_bot
    cdef double y1w_bot, y1wr_bot, y2w_bot, y2wr_bot
    
    cdef double fAw_bot, fBw_bot 
    cdef double gAw_bot, gBw_bot
    
    cdef long [:] k_crack_term2_partA_22r, k_crack_term2_partA_22c
    cdef double [:] k_crack_term2_partA_22v
    
    cdef double del_d_i_1_iter, del_d_i_iter
    
    cdef double [:] weights_xi, weights_eta, xis, etas
    
    m_bot = p_bot.m
    n_bot = p_bot.n
    
    a_bot = p_bot.a
    b_bot = p_bot.b
    
    # Panel top (ends in top)
    x1u_bot = p_bot.x1u ; x1ur_bot = p_bot.x1ur ; x2u_bot = p_bot.x2u ; x2ur_bot = p_bot.x2ur
    x1v_bot = p_bot.x1v ; x1vr_bot = p_bot.x1vr ; x2v_bot = p_bot.x2v ; x2vr_bot = p_bot.x2vr
    x1w_bot = p_bot.x1w ; x1wr_bot = p_bot.x1wr ; x2w_bot = p_bot.x2w ; x2wr_bot = p_bot.x2wr
    y1u_bot = p_bot.y1u ; y1ur_bot = p_bot.y1ur ; y2u_bot = p_bot.y2u ; y2ur_bot = p_bot.y2ur
    y1v_bot = p_bot.y1v ; y1vr_bot = p_bot.y1vr ; y2v_bot = p_bot.y2v ; y2vr_bot = p_bot.y2vr
    y1w_bot = p_bot.y1w ; y1wr_bot = p_bot.y1wr ; y2w_bot = p_bot.y2w ; y2wr_bot = p_bot.y2wr
    
    # Initializing gauss points, weights
    xis = np.zeros(no_x_gauss, dtype=DOUBLE)
    weights_xi = np.zeros(no_x_gauss, dtype=DOUBLE)
    etas = np.zeros(no_y_gauss, dtype=DOUBLE)
    weights_eta = np.zeros(no_y_gauss, dtype=DOUBLE)

    # Calc gauss points and weights
    leggauss_quad(no_x_gauss, &xis[0], &weights_xi[0])
    leggauss_quad(no_y_gauss, &etas[0], &weights_eta[0])
    
    # Dimension of the number of values that need to be stored
    fdim = 1*m_bot*n_bot*m_bot*n_bot    # 1 bec only 1 term is being added in the for loops
    
    k_crack_term2_partA_22r = np.zeros((fdim,), dtype=INT)
    k_crack_term2_partA_22c = np.zeros((fdim,), dtype=INT)
    k_crack_term2_partA_22v = np.zeros((fdim,), dtype=DOUBLE)
    
    with nogil:
        
        for ptx in range(no_x_gauss):
            for pty in range(no_y_gauss):
                # Takes the correct index instead of the location
                xi = xis[ptx]
                eta = etas[pty]

                weight = weights_xi[ptx] * weights_eta[pty]
                
                del_d_i_1_iter = del_d_i_1[pty, ptx]
                del_d_i_iter = del_d_i[pty, ptx]
                
                if del_d_i_iter == 0:
                    continue
                
                if dmg_index[pty, ptx] == 0 or dmg_index[pty, ptx] == 1:
                    continue
                
                c = -1
                for i1 in range(m_bot):
                    fAw_bot = f(i1, xi, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot)
                    
                    for k1 in range(m_bot):
                        fBw_bot = f(k1, xi, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot)
                        
                        for j1 in range(n_bot): 
                            gAw_bot = f(j1, eta, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot)
                                    
                            for l1 in range(n_bot):
                                gBw_bot = f(l1, eta, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot)
        
                                row = row0 + DOF*(j1*m_bot + i1)
                                col = col0 + DOF*(l1*m_bot + k1)
        
                                #NOTE symmetry - 22
                                if row > col:
                                    continue
        
                                c += 1
                                k_crack_term2_partA_22r[c] = row+2
                                k_crack_term2_partA_22c[c] = col+2
                                # k_crack22v[c] += -a_bot*b_bot*del_o*fAw_bot*fBw_bot*gAw_bot*gBw_bot*k_o*weight/(4*(del_f - del_o)) # OLD REMOVE
                                k_crack_term2_partA_22v[c] += a_bot*b_bot*del_f*del_d_i_1_iter*del_o*fAw_bot*fBw_bot*gAw_bot*gBw_bot*k_o*weight/(4*(del_d_i_iter*del_d_i_iter*del_d_i_iter)*(del_f - del_o))

    k_crack_term2_partA_22 = coo_matrix((k_crack_term2_partA_22v, (k_crack_term2_partA_22r, k_crack_term2_partA_22c)), shape=(size, size))
    # Builds a matrix of size = size x size (so complete size of global MD) and populates it with the data in ..v 
        # where the rows and cols where that data should go are specified by ..r, ...c
    # This way its at the correct positions in the global MD matrix as row and col are the starting indices of the 
        # submatrices in the global MD matrix

    return k_crack_term2_partA_22


def k_crack_term2_partB(object p_top, object p_bot, int size, int no_x_gauss, int no_y_gauss, double [:,::1] dmg_index, 
            double [:] c_i):
    # double[:,::1] defines the variable as a 2D, C contiguous memoryview of doubles
    
    cdef int m_top, n_top, m_bot, n_bot
    cdef int i_top, j_top, i_bot, j_bot
    cdef int i_outer, j_outer
    cdef int row_start_top, row_start_bot
    cdef double xi, eta, weight
    cdef int ptx, pty, row
    cdef double a_top, b_top, a_bot, b_bot
    
    cdef double x1u_top, x1ur_top, x2u_top, x2ur_top, x1u_bot, x1ur_bot, x2u_bot, x2ur_bot
    cdef double x1v_top, x1vr_top, x2v_top, x2vr_top, x1v_bot, x1vr_bot, x2v_bot, x2vr_bot
    cdef double x1w_top, x1wr_top, x2w_top, x2wr_top, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot
    cdef double y1u_top, y1ur_top, y2u_top, y2ur_top, y1u_bot, y1ur_bot, y2u_bot, y2ur_bot
    cdef double y1v_top, y1vr_top, y2v_top, y2vr_top, y1v_bot, y1vr_bot, y2v_bot, y2vr_bot
    cdef double y1w_top, y1wr_top, y2w_top, y2wr_top, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot
    
    # Only single power of SF so only A. No B needed like for u*u
    cdef double fAw_top, fAw_bot 
    cdef double gAw_top, gAw_bot

    cdef double [::1] s_delta
    cdef double [:,::1] k_crack_term2_partB
    
    cdef double [:] weights_xi, weights_eta, xis, etas
    
    m_top = p_top.m
    n_top = p_top.n
    m_bot = p_bot.m
    n_bot = p_bot.n
    
    a_top = p_top.a
    b_top = p_top.b
    a_bot = p_bot.a
    b_bot = p_bot.b
    
    row_start_top = p_top.row_start
    row_start_bot = p_bot.row_start
    
    # Panel top (ends in top)
    x1u_top = p_top.x1u ; x1ur_top = p_top.x1ur ; x2u_top = p_top.x2u ; x2ur_top = p_top.x2ur
    x1v_top = p_top.x1v ; x1vr_top = p_top.x1vr ; x2v_top = p_top.x2v ; x2vr_top = p_top.x2vr
    x1w_top = p_top.x1w ; x1wr_top = p_top.x1wr ; x2w_top = p_top.x2w ; x2wr_top = p_top.x2wr
    y1u_top = p_top.y1u ; y1ur_top = p_top.y1ur ; y2u_top = p_top.y2u ; y2ur_top = p_top.y2ur
    y1v_top = p_top.y1v ; y1vr_top = p_top.y1vr ; y2v_top = p_top.y2v ; y2vr_top = p_top.y2vr
    y1w_top = p_top.y1w ; y1wr_top = p_top.y1wr ; y2w_top = p_top.y2w ; y2wr_top = p_top.y2wr

    # Panel bot (ends in bot)
    x1u_bot = p_bot.x1u ; x1ur_bot = p_bot.x1ur ; x2u_bot = p_bot.x2u ; x2ur_bot = p_bot.x2ur
    x1v_bot = p_bot.x1v ; x1vr_bot = p_bot.x1vr ; x2v_bot = p_bot.x2v ; x2vr_bot = p_bot.x2vr
    x1w_bot = p_bot.x1w ; x1wr_bot = p_bot.x1wr ; x2w_bot = p_bot.x2w ; x2wr_bot = p_bot.x2wr
    y1u_bot = p_bot.y1u ; y1ur_bot = p_bot.y1ur ; y2u_bot = p_bot.y2u ; y2ur_bot = p_bot.y2ur
    y1v_bot = p_bot.y1v ; y1vr_bot = p_bot.y1vr ; y2v_bot = p_bot.y2v ; y2vr_bot = p_bot.y2vr
    y1w_bot = p_bot.y1w ; y1wr_bot = p_bot.y1wr ; y2w_bot = p_bot.y2w ; y2wr_bot = p_bot.y2wr
    
    # Initializing gauss points, weights
    xis = np.zeros(no_x_gauss, dtype=DOUBLE)
    weights_xi = np.zeros(no_x_gauss, dtype=DOUBLE)
    etas = np.zeros(no_y_gauss, dtype=DOUBLE)
    weights_eta = np.zeros(no_y_gauss, dtype=DOUBLE)

    # Calc gauss points and weights
    leggauss_quad(no_x_gauss, &xis[0], &weights_xi[0])
    leggauss_quad(no_y_gauss, &etas[0], &weights_eta[0])
    
    s_delta = np.zeros(size, dtype=DOUBLE)
    k_crack_term2_partB = np.zeros((size, size), dtype=DOUBLE)
    
    with nogil:
        
        # TOP Panel
        for ptx in range(no_x_gauss):
            for pty in range(no_y_gauss):
                # Takes the correct index instead of the location
                xi = xis[ptx]
                eta = etas[pty]
    
                if dmg_index[pty, ptx] == 0 or dmg_index[pty, ptx] == 1:
                    continue
    
                weight = weights_xi[ptx] * weights_eta[pty]
                
                # Extracting the correct values (later in fcrack's eqn) [pty, ptx]
                    # Currently, the outer loop of x and inner of y, causes it to go through all y for a single x
                    # That is going through all rows for a single col then onto the next col
                    # (as per x, y and results by calc_results)

                for j_top in range(n_top):
                    gAw_top = f(j_top, eta, y1w_top, y1wr_top, y2w_top, y2wr_top)
                    
                    for i_top in range(m_top):
                        fAw_top = f(i_top, xi, x1w_top, x1wr_top, x2w_top, x2wr_top)
                        
                        row = row_start_top + DOF*(j_top*m_top + i_top)
                        
                        s_delta[row+2] += (fAw_top * gAw_top)
        
        # BOT Panel
        for ptx in range(no_x_gauss):
            for pty in range(no_y_gauss):
                # Takes the correct index instead of the location
                xi = xis[ptx]
                eta = etas[pty]
    
                if dmg_index[pty, ptx] == 0 or dmg_index[pty, ptx] == 1:
                    continue
    
                weight = weights_xi[ptx] * weights_eta[pty]
                
                # Extracting the correct values (later in fcrack's eqn) [pty, ptx]
                    # Currently, the outer loop of x and inner of y, causes it to go through all y for a single x
                    # That is going through all rows for a single col then onto the next col
                    # (as per x, y and results by calc_results)
                
                for j_bot in range(n_bot):
                    gAw_bot = f(j_bot, eta, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot)
                    
                    for i_bot in range(m_bot):
                        fAw_bot = f(i_bot, xi, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot)
                        
                        row = row_start_bot + DOF*(j_bot*m_bot + i_bot)
                        
                        s_delta[row+2] += - (fAw_bot * gAw_bot)
                        
        for i_outer in range(size):
            for j_outer in range(size):
                k_crack_term2_partB[i_outer, j_outer] = c_i[i_outer]*s_delta[j_outer]
    
    return k_crack_term2_partB
    
    
    