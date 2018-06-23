from __future__ import division, absolute_import
import gc
import copy

import numpy as np
from numpy import deg2rad
from composites import laminate
from structsolve.sparseutils import finalize_symmetric_matrix

from .. import modelDB
from .. shell import Shell
from .. logger import msg, warn


class BladeStiff1D(object):
    r"""Blade Stiffener using 1D Formulation for Flange

    Blade-type of stiffener model using a 1D formulation for the flange and a
    2D formulation for the padup (base)::


                 || --> flange       |
                 ||                  |-> stiffener
               ======  --> padup     |
      =========================  --> shells
         Shell1      Shell2

    Both the flange and the padup are optional, but one must exist.

    Each stiffener has a constant `y` coordinate.

    """
    def __init__(self, bay, rho, shell1, shell2, ys, bb, bf, bstack, bplyts,
            blaminaprops, fstack, fplyts, flaminaprops):
        self.bay = bay
        self.shell1 = shell1
        self.shell2 = shell2
        self.model = 'bladestiff1d_clpt_donnell_bardell'
        self.rho = rho
        self.ys = ys
        self.bb = bb
        self.hb = 0.
        self.bf = bf
        self.hf = 0.

        self.bstack = bstack
        self.bplyts = bplyts
        self.blaminaprops = blaminaprops
        self.base = None
        self.fstack = fstack
        self.fplyts = fplyts
        self.flaminaprops = flaminaprops
        self.flam = None

        self.As = None
        self.Asb = None
        self.Asf = None
        self.Jxx = None
        self.Iyy = None

        self.Fx = None

        self.matrices = dict(kC=None, kG=None, kM=None)

        self._rebuild()


    def _rebuild(self):
        assert self.shell1.model == self.shell2.model
        assert self.shell1.m == self.shell2.m
        assert self.shell1.n == self.shell2.n
        assert self.shell1.r == self.shell2.r
        assert self.shell1.alphadeg == self.shell2.alphadeg

        if self.fstack is not None:
            self.hf = sum(self.fplyts)
            self.Asf = self.bf*self.hf
            self.flam = laminate.read_stack(self.fstack, plyts=self.fplyts,
                                             laminaprops=self.flaminaprops)
            self.flam.calc_equivalent_modulus()

        h = 0.5*sum(self.shell1.plyts) + 0.5*sum(self.shell2.plyts)
        hb = 0.
        if self.bstack is not None:
            hb = sum(self.bplyts)
            y1 = self.ys - self.bb/2.
            y2 = self.ys + self.bb/2.
            self.base = Shell(a=bay.a, b=bay.b, r=bay.r, alphadeg=bay.alphadeg,
                    stack=self.bstack, plyts=self.bplyts,
                    rho=self.rho, m=bay.m, n=bay.n,
                    laminaprops=self.blaminaprops, offset=(-h/2.-hb/2.),
                    u1tx=bay.u1tx, u1rx=bay.u1rx, u2tx=bay.u2tx, u2rx=bay.u2rx,
                    v1tx=bay.v1tx, v1rx=bay.v1rx, v2tx=bay.v2tx, v2rx=bay.v2rx,
                    w1tx=bay.w1tx, w1rx=bay.w1rx, w2tx=bay.w2tx, w2rx=bay.w2rx,
                    u1ty=bay.u1ty, u1ry=bay.u1ry, u2ty=bay.u2ty, u2ry=bay.u2ry,
                    v1ty=bay.v1ty, v1ry=bay.v1ry, v2ty=bay.v2ty, v2ry=bay.v2ry,
                    w1ty=bay.w1ty, w1ry=bay.w1ry, w2ty=bay.w2ty, w2ry=bay.w2ry,
                    y1=y1, y2=y2)
            self.Asb = self.bb*hb

        #TODO check offset effect on curved shells
        self.dbf = self.bf/2. + hb + h/2.
        self.Iyy = self.hf*self.bf**3/12.
        self.Jxx = self.hf*self.bf**3/12. + self.bf*self.hf**3/12.

        Asb = self.Asb if self.Asb is not None else 0.
        Asf = self.Asf if self.Asf is not None else 0.
        self.As = Asb + Asf

        if self.fstack is not None:
            self.E1 = 0
            #E3 = 0
            self.S1 = 0
            yply = self.flam.plies[0].h/2.
            for i, ply in enumerate(self.flam.plies):
                if i > 0:
                    yply += self.flam.plies[i-1].h/2. + self.flam.plies[i].h/2.
                q = ply.QL
                self.E1 += ply.h*(q[0,0] - q[0,1]**2/q[1,1])
                #E3 += ply.h*(q[2,2] - q[1,2]**2/q[1,1])
                self.S1 += -yply*ply.h*(q[0,2] - q[0,1]*q[1,2]/q[1,1])

            self.F1 = self.bf**2/12.*self.E1


    def calc_kC(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        """Calculate the linear constitutive stiffness matrix
        """
        self._rebuild()
        msg('Calculating kC... ', level=2, silent=silent)

        kC = 0.
        if self.base is not None:
            kC += self.base.calc_kC(size=size, row0=row0, col0=col0,
                    silent=True, finalize=False)
        if self.flam is not None:
            mod = modelDB.db[self.model]['matrices']
            bay = self.bay
            kC += mod.fkCf(self.ys, bay.a, bay.b, self.bf, self.dbf, self.E1, self.F1,
                           self.S1, self.Jxx, bay.m, bay.n,
                           bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                           bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                           bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                           bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                           size=size, row0=row0, col0=col0)

        if finalize:
            kC = finalize_symmetric_matrix(kC)
        self.matrices['kC'] = kC

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kC


    def calc_kG(self, size=None, row0=0, col0=0, silent=False, finalize=True, c=None):
        """Calculate the geometric stiffness matrix
        """
        #TODO
        if c is not None:
            raise NotImplementedError('numerical kG not implemented')

        self._rebuild()
        msg('Calculating kG... ', level=2, silent=silent)

        kG = 0.
        if self.base is not None:
            # TODO include kG for padup
            #      now it is assumed that all the load goes to the flange
            pass
        if self.flam is not None:
            Fx = self.Fx if self.Fx is not None else 0.
            mod = modelDB.db[self.model]['matrices']
            bay = self.bay
            kG += mod.fkGf(self.ys, Fx, bay.a, bay.b, self.bf, bay.m, bay.n,
                            bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                            bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                            size, row0, col0)

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

        mod = modelDB.db[self.model]['matrices']

        kM = 0.
        if self.base is not None:
            kM += self.base.calc_kM(size=size, row0=row0, col0=col0, silent=silent, finalize=False)
        if self.flam is not None:
            bay = self.bay
            h = 0.5*sum(self.shell1.plyts) + 0.5*sum(self.shell2.plyts)
            kM += mod.fkMf(self.ys, self.rho, h, self.hb, self.hf, bay.a, bay.b,
                           self.bf, self.dbf, bay.m, bay.n,
                           bay.u1tx, bay.u1rx, bay.u2tx, bay.u2rx,
                           bay.v1tx, bay.v1rx, bay.v2tx, bay.v2rx,
                           bay.w1tx, bay.w1rx, bay.w2tx, bay.w2rx,
                           bay.u1ty, bay.u1ry, bay.u2ty, bay.u2ry,
                           bay.v1ty, bay.v1ry, bay.v2ty, bay.v2ry,
                           bay.w1ty, bay.w1ry, bay.w2ty, bay.w2ry,
                           size=size, row0=row0, col0=col0)

        if finalize:
            kM = finalize_symmetric_matrix(kM)
        self.matrices['kM'] = kM

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kM

