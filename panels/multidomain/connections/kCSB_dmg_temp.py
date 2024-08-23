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

