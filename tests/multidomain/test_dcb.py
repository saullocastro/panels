
import pytest
import numpy as np
from structsolve import lb, static

from panels import Shell
from panels.multidomain import MultiDomain


@pytest.fixture
def dcb_bending():

    # Properties
    E1 = 127560 # MPa
    E2 = 13030. # MPa
    G12 = 6410. # MPa
    nu12 = 0.3
    ply_thickness = 0.127 # mm

    # Plate dimensions
    a = 1181.1
    b = 746.74

    #others
    m = 8
    n = 8

    simple_layup = [+45, -45]*20 + [0, 90]*20
    simple_layup += simple_layup[::-1]

    laminaprop = (E1, E2, nu12, G12, G12, G12)
    
    # DCB panels 
    top = Shell(group='top', a=a, b=b,m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    bot = Shell(group='bot', a=a, b=b,m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)

    # boundary conditions
    top.x1u =  ; top.x1ur =  ; top.x2u =  ; top.x2ur = 
    top.x1v =  ; top.x1vr =  ; top.x2v =  ; top.x2vr = 
    top.x1w =  ; top.x1wr =  ; top.x2w =  ; top.x2wr = 
    top.y1u =  ; top.y1ur =  ; top.y2u =  ; top.y2ur = 
    top.y1v =  ; top.y1vr =  ; top.y2v =  ; top.y2vr = 
    top.y1w =  ; top.y1wr =  ; top.y2w =  ; top.y2wr = 

    bot.x1u =  ; bot.x1ur =  ; bot.x2u =  ; bot.x2ur = 
    bot.x1v =  ; bot.x1vr =  ; bot.x2v =  ; bot.x2vr = 
    bot.x1w =  ; bot.x1wr =  ; bot.x2w =  ; bot.x2wr = 
    bot.y1u =  ; bot.y1ur =  ; bot.y2u =  ; bot.y2ur = 
    bot.y1v =  ; bot.y1vr =  ; bot.y2v =  ; bot.y2vr = 
    bot.y1w =  ; bot.y1wr =  ; bot.y2w =  ; bot.y2wr = 
