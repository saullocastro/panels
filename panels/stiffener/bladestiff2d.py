from __future__ import division, absolute_import
import gc

import numpy as np
from numpy import deg2rad
from composites import laminate
from structsolve.sparseutils import finalize_symmetric_matrix

from .. modelDB import db
from .. logger import msg, warn
from .. shell import Shell
from .. panel.connections import fkCBFycte11, fkCBFycte12, fkCBFycte22
from .. panel.connections import calc_kt_kr

class BladeStiff2D(object):
    r"""Blade Stiffener using 2D Formulation for Flange

    Blade-type of stiffener model using a 2D formulation for the flange and a
    2D formulation for the base (padup)::


                 || --> flange       |
                 ||                  |-> stiffener
               ======  --> padup     |
      =========================  --> shells
         Shell1      Shell2

    Both the flange and the base are optional. The stiffener's base is modeled
    using the same approximation functions as the skin, with the proper
    offset.

    Each stiffener has a constant `y_s` coordinate.

    """
    def __init__(self, bay, rho, shell1, shell2, ys, bb, bf, bstack, bplyts,
            blaminaprops, fstack, fplyts, flaminaprops, mf=14, nf=11):
        self.bay = bay
        self.shell1 = shell1
        self.shell2 = shell2
        self.rho = rho
        self.ys = ys
        self.bb = bb
        self.forces_flange = []

        self.bstack = bstack
        self.bplyts = bplyts
        self.blaminaprops = blaminaprops

        self.matrices = dict(kC=None, kG=None, kM=None)

        self.base = None
        if bstack is not None:
            y1 = self.ys - bb/2.
            y2 = self.ys + bb/2.
            h = 0.5*sum(self.shell1.plyts) + 0.5*sum(self.shell2.plyts)
            hb = sum(self.bplyts)
            self.base = Shell(a=bay.a, b=bay.b, r=bay.r, alphadeg=bay.alphadeg,
                    stack=bstack, plyts=bplyts, laminaprops=blaminaprops,
                    rho=rho, m=bay.m, n=bay.n, offset=(-h/2.-hb/2.),
                    x1u=bay.x1u, x2u=bay.x2u,
                    x1v=bay.x1v, x2v=bay.x2v,
                    x1w=bay.x1w, x1wr=bay.x1wr, x2w=bay.x2w, x2wr=bay.x2wr,
                    y1u=bay.y1u, y2u=bay.y2u,
                    y1v=bay.y1v, y2v=bay.y2v,
                    y1w=bay.y1w, y1wr=bay.y1wr, y2w=bay.y2w, y2wr=bay.y2wr,
                    y1=y1, y2=y2)

        self.flange = None
        if fstack is not None:
            self.flange = Shell(m=mf, n=nf, a=bay.a, b=bf, rho=rho,
                    stack=fstack, plyts=fplyts, laminaprops=flaminaprops,
                    model='plate_clpt_donnell_bardell',
                    u1tx=0., u2tx=0.,
                    v1tx=0., v2tx=0.,
                    w1tx=0., w1rx=1., w2tx=0., w2rx=1.,
                    u1ty=1., u2ty=1.,
                    v1ty=1., v2ty=1.,
                    w1ty=1., w1ry=1., w2ty=1., w2ry=1.)

        self._rebuild()


    def _rebuild(self):
        assert self.shell1.model == self.shell2.model
        assert self.shell1.m == self.shell2.m
        assert self.shell1.n == self.shell2.n
        assert self.shell1.r == self.shell2.r
        assert self.shell1.alphadeg == self.shell2.alphadeg
        if self.flange is not None:
            self.flange.lam = laminate.read_stack(self.flange.stack, plyts=self.flange.plyts,
                                            laminaprops=self.flange.laminaprops)
            self.flange.lam.calc_equivalent_modulus()

        if self.base is not None:
            h = 0.5*sum(self.shell1.plyts) + 0.5*sum(self.shell2.plyts)
            hb = sum(self.bplyts)
            self.dpb = h/2. + hb/2.
            self.base.lam = laminate.read_stack(self.bstack, plyts=self.bplyts,
                                            laminaprops=self.blaminaprops,
                                            offset=(-h/2.-hb/2.))


    def calc_kC(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        """Calculate the linear constitutive stiffness matrix
        """
        self._rebuild()
        msg('Calculating kC... ', level=2, silent=silent)

        flangemod = db[self.flange.model]['matrices_num']

        bay = self.bay
        a = bay.a
        b = bay.b
        m = bay.m
        n = bay.n

        kC = 0.
        if self.base is not None:
            kC += self.base.calc_kC(size=size, row0=0, col0=0, silent=True,
                    finalize=False)
        if self.flange is not None:
            kC += self.flange.calc_kC(size=size, row0=row0, col0=col0,
                    silent=True, finalize=False)

            # connectivity between stiffener'base and stiffener's flange
            if self.base is None:
                ktbf, krbf = calc_kt_kr(self.shell1, self.flange, 'ycte')
            else:
                ktbf, krbf = calc_kt_kr(self.base, self.flange, 'ycte')

            mod = db['bladestiff2d_clpt_donnell_bardell']['connections']
            kC += mod.fkCss(bay, self,
                    #ktbf, krbf, self.ys, a, b, m, n,
                            size, 0, 0)
            bf = self.flange.b
            kC += mod.fkCsf(bay, self,
                    #ktbf, krbf, self.ys, a, b, bf, m, n, self.flange.m, self.flange.n,
                            size, 0, col0)
            kC += mod.fkCff(bay, self,
                    #ktbf, krbf, a, bf, self.flange.m, self.flange.n,
                            size, row0, col0)

        if finalize:
            kC = finalize_symmetric_matrix(kC)
        self.matrices['kC'] = kC

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kC


    def calc_kG(self, size=None, row0=0, col0=0, silent=False, finalize=True,
            c=None, NLgeom=False):
        """Calculate the linear geometric stiffness matrix
        """
        self._rebuild()
        msg('Calculating kG... ', level=2, silent=silent)

        kG = 0.
        if self.base is not None:
            #TODO include kG for pad-up and Nxx load that arrives there
            pass
        if self.flange is not None:
            kG += self.flange.calc_kG(size=size, row0=row0, col0=col0,
                    silent=True, finalize=False, NLgeom=NLgeom)

        if finalize:
            kG = finalize_symmetric_matrix(kG)
        self.matrices['kG'] = kG

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kG


    def calc_kM(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        """Calculate the mass matrix
        """
        self._rebuild()
        msg('Calculating kM... ', level=2, silent=silent)

        kM = 0.
        if self.base is not None:
            kM += self.base.calc_kM(size=size, row0=0, col0=0, silent=True, finalize=False)
        if self.flange is not None:
            kM += self.flange.calc_kM(size=size, row0=row0, col0=col0, silent=True, finalize=False)

        if finalize:
            kM = finalize_symmetric_matrix(kM)
        self.matrices['kM'] = kM

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kM

