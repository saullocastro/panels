
import numpy as np
from structsolve import solve
from structsolve.sparseutils import finalize_symmetric_matrix

from panels import Shell
from panels.multidomain.connections import calc_ku_kv_kw_point_pd
from panels.multidomain.connections import fkCpd
from panels.plot_shell import plot_shell
from panels.multidomain import MultiDomain

def test_dcb_bending_pd():

    # Properties
    E1 = 127560e6 # Pa
    E2 = 13030.e6 # Pa
    G12 = 6410.e6 # Pa
    nu12 = 0.3
    ply_thickness = 0.127e-3 # m

    # Plate dimensions
    a = 1.1811
    b = 0.74674
    
    a1 = 0.5*a

    #others
    m = 8
    n = 8

    simple_layup = [+45, -45]*20 + [0, 90]*20
    simple_layup += simple_layup[::-1]

    laminaprop = (E1, E2, nu12, G12, G12, G12)
     
    # skin panels
    top1 = Shell(group='top', x0=0, y0=0, a=a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    top2 = Shell(group='top', x0=a1, y0=0, a=a-a1, b=b, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    # skin panels
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
        
        bot2.x1u = 1 ; bot2.x1ur = 1 ; bot2.x2u = 0 ; bot2.x2ur = 0
        bot2.x1v = 1 ; bot2.x1vr = 1 ; bot2.x2v = 0 ; bot2.x2vr = 0 
        bot2.x1w = 1 ; bot2.x1wr = 1 ; bot2.x2w = 0 ; bot2.x2wr = 0 
        bot2.y1u = 1 ; bot2.y1ur = 1 ; bot2.y2u = 1 ; bot2.y2ur = 1
        bot2.y1v = 1 ; bot2.y1vr = 1 ; bot2.y2v = 1 ; bot2.y2vr = 1
        bot2.y1w = 1 ; bot2.y1wr = 1 ; bot2.y2w = 1 ; bot2.y2wr = 1

    conn = [
     # skin-skin
     dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
     dict(p1=bot1, p2=bot2, func='SSxcte', xcte1=bot1.a, xcte2=0),
     dict(p1=bot1, p2=top1, func='SB'), 
    ]
    
    panels = [bot1, bot2, top1, top2]

    assy = MultiDomain(panels)

    k0 = assy.calc_kC(conn)
    
    size = k0.shape[0]

    if True:
        ku, kv, kw = calc_ku_kv_kw_point_pd(top2)
        print(ku, kv, kw)
        kCp = fkCpd(0., 0., kw, top2, top2.a, top2.b/2, size, top2.row_start, top2.col_start)
        print(top2.row_start, top2.col_start)
        kCp = finalize_symmetric_matrix(kCp)
        k0 = k0 + kCp
        wp = 0.001
        top2.add_point_pd(top2.a, top2.b/2, 0., 0., 0., 0., kw, wp)
        
    else:
        top2.add_point_load(top2.a, top2.b/2, 0, 0, 1.)
        
    fext = top2.calc_fext(size=size, col0=top2.col_start)
    c0 = solve(k0, fext)
    ax, data = assy.plot(c=c0, group='bot', vec='w', filename='test_dcb_before_opening_bot.png', colorbar=True)
    
    #vecmin = data['vecmin']
    #vecmax = data['vecmax']
    vecmin = vecmax = None
    assy.plot(c=c0, group='top', vec='w', filename='test_dcb_before_opening_top.png', colorbar=True, vecmin=vecmin, vecmax=vecmax)


if __name__ == "__main__":
    test_dcb_bending_pd()