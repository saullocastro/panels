import gc

from composites import laminated_plate
from structsolve.sparseutils import finalize_symmetric_matrix

from panels import Shell
from panels.logger import msg
from . import modelDB


class BladeStiff1D(object):
    r"""Blade Stiffener using 1D Formulation for Flange

    Blade-type of stiffener model using a 1D formulation for the flange and a
    2D formulation for the padup (base)::


                 || --> flange       |
                 ||                  |-> stiffener
               ======  --> padup     |
      =========================  --> panels
         Panel1      Panel2

    Both the flange and the padup are optional, but one must exist.

    Each stiffener has a constant `y` coordinate.

    """
    def __init__(self, bay, rho, panel1, panel2, ys, bb, bf, bstack, bplyts,
            blaminaprops, fstack, fplyts, flaminaprops):
        self.bay = bay
        self.panel1 = panel1
        self.panel2 = panel2
        self.model = 'bladestiff1d_clt_donnell_bardell'
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

        self._rebuild()


    def _rebuild(self):
        assert self.panel1.model == self.panel2.model
        assert self.panel1.m == self.panel2.m
        assert self.panel1.n == self.panel2.n
        assert self.panel1.r == self.panel2.r

        if self.fstack is not None:
            self.hf = sum(self.fplyts)
            self.Asf = self.bf*self.hf
            self.flam = laminated_plate(self.fstack, plyts=self.fplyts, laminaprops=self.flaminaprops)
            self.flam.calc_equivalent_properties()

        h = 0.5*sum(self.panel1.plyts) + 0.5*sum(self.panel2.plyts)
        hb = 0.
        bay = self.bay
        if self.bstack is not None:
            hb = sum(self.bplyts)
            y1 = self.ys - self.bb/2.
            y2 = self.ys + self.bb/2.
            self.base = Shell(a=bay.a, b=bay.b, r=bay.r,
                    stack=self.bstack, plyts=self.bplyts,
                    rho=self.rho, m=bay.m, n=bay.n,
                    laminaprops=self.blaminaprops, offset=(-h/2.-hb/2.),
                    x1u=bay.x1u, x1ur=bay.x1ur, x2u=bay.x2u, x2ur=bay.x2ur,
                    x1v=bay.x1v, x1vr=bay.x1vr, x2v=bay.x2v, x2vr=bay.x2vr,
                    x1w=bay.x1w, x1wr=bay.x1wr, x2w=bay.x2w, x2wr=bay.x2wr,
                    y1u=bay.y1u, y1ur=bay.y1ur, y2u=bay.y2u, y2ur=bay.y2ur,
                    y1v=bay.y1v, y1vr=bay.y1vr, y2v=bay.y2v, y2vr=bay.y2vr,
                    y1w=bay.y1w, y1wr=bay.y1wr, y2w=bay.y2w, y2wr=bay.y2wr,
                    y1=y1, y2=y2)
            self.Asb = self.bb*hb

        #TODO check offset effect on curved panels
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
                self.E1 += ply.h*(ply.q11L - ply.q12L**2/ply.q22L)
                self.S1 += -yply*ply.h*(ply.q16L - ply.q12L*ply.q16L/ply.q22L)

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
                           bay.x1u, bay.x1ur, bay.x2u, bay.x2ur,
                           bay.x1w, bay.x1wr, bay.x2w, bay.x2wr,
                           bay.y1u, bay.y1ur, bay.y2u, bay.y2ur,
                           bay.y1w, bay.y1wr, bay.y2w, bay.y2wr,
                           size=size, row0=row0, col0=col0)

        if finalize:
            kC = finalize_symmetric_matrix(kC)
        self.kC = kC

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)


    def calc_kG(self, size=None, row0=0, col0=0, silent=False, finalize=True,
            c=None):
        """Calculate the linear geometric stiffness matrix
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
                             bay.x1w, bay.x1wr, bay.x2w, bay.x2wr,
                             bay.y1w, bay.y1wr, bay.y2w, bay.y2wr,
                             size, row0, col0)

        if finalize:
            kG = finalize_symmetric_matrix(kG)
        self.kG = kG

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)


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
            h = 0.5*sum(self.panel1.plyts) + 0.5*sum(self.panel2.plyts)
            kM += mod.fkMf(self.ys, self.rho, h, self.hb, self.hf, bay.a, bay.b,
                           self.bf, self.dbf, bay.m, bay.n,
                           bay.x1u, bay.x1ur, bay.x2u, bay.x2ur,
                           bay.x1v, bay.x1vr, bay.x2v, bay.x2vr,
                           bay.x1w, bay.x1wr, bay.x2w, bay.x2wr,
                           bay.y1u, bay.y1ur, bay.y2u, bay.y2ur,
                           bay.y1v, bay.y1vr, bay.y2v, bay.y2vr,
                           bay.y1w, bay.y1wr, bay.y2w, bay.y2wr,
                           size=size, row0=row0, col0=col0)

        if finalize:
            kM = finalize_symmetric_matrix(kM)
        self.kM = kM

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

