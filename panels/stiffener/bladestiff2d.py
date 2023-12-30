import gc

from composites import laminated_plate
from structsolve.sparseutils import finalize_symmetric_matrix

from panels import Shell
from panels.logger import msg
from panels.multidomain.connections import calc_kt_kr
from .modelDB import db

class BladeStiff2D(object):
    r"""Blade Stiffener using 2D Formulation for Flange

    Blade-type of stiffener model using a 2D formulation for the flange and a
    2D formulation for the base (padup)::


                 || --> flange       |
                 ||                  |-> stiffener
               ======  --> padup     |
      =========================  --> panels
         Panel1      Panel2

    Both the flange and the base are optional. The stiffener's base is modeled
    using the same approximation functions as the skin, with the proper
    offset.

    Each stiffener has a constant `y_s` coordinate.

    """
    def __init__(self, bay, rho, panel1, panel2, ys, bb, bf, bstack, bplyts,
            blaminaprops, fstack, fplyts, flaminaprops, mf=14, nf=11):
        self.bay = bay
        self.panel1 = panel1
        self.panel2 = panel2
        self.rho = rho
        self.ys = ys
        self.bb = bb
        self.forces_flange = []

        self.bstack = bstack
        self.bplyts = bplyts
        self.blaminaprops = blaminaprops

        self.kC = None
        self.kM = None
        self.kG = None

        self.base = None
        if bstack is not None:
            y1 = self.ys - bb/2.
            y2 = self.ys + bb/2.
            h = 0.5*sum(self.panel1.plyts) + 0.5*sum(self.panel2.plyts)
            hb = sum(self.bplyts)
            self.base = Shell(a=bay.a, b=bay.b, r=bay.r,
                    stack=bstack, plyts=bplyts, laminaprops=blaminaprops,
                    rho=rho, m=bay.m, n=bay.n, offset=(-h/2.-hb/2.),
                    x1u=bay.x1u, x1ur=bay.x1ur, x2u=bay.x2u, x2ur=bay.x2ur,
                    x1v=bay.x1v, x1vr=bay.x1vr, x2v=bay.x2v, x2vr=bay.x2vr,
                    x1w=bay.x1w, x1wr=bay.x1wr, x2w=bay.x2w, x2wr=bay.x2wr,
                    y1u=bay.y1u, y1ur=bay.y1ur, y2u=bay.y2u, y2ur=bay.y2ur,
                    y1v=bay.y1v, y1vr=bay.y1vr, y2v=bay.y2v, y2vr=bay.y2vr,
                    y1w=bay.y1w, y1wr=bay.y1wr, y2w=bay.y2w, y2wr=bay.y2wr,
                    y1=y1, y2=y2)

        self.flange = None
        if fstack is not None:
            self.flange = Shell(m=mf, n=nf, a=bay.a, b=bf, rho=rho,
                    stack=fstack, plyts=fplyts, laminaprops=flaminaprops,
                    model='plate_clpt_donnell_bardell',
                    x1u=0., x1ur=0., x2u=0., x2ur=0.,
                    x1v=0., x1vr=0., x2v=0., x2vr=0.,
                    x1w=0., x1wr=1., x2w=0., x2wr=1.,
                    y1u=1., y1ur=0., y2u=1., y2ur=0.,
                    y1v=1., y1vr=0., y2v=1., y2vr=0.,
                    y1w=1., y1wr=1., y2w=1., y2wr=1.)

        self._rebuild()


    def _rebuild(self):
        assert self.panel1.model == self.panel2.model
        assert self.panel1.m == self.panel2.m
        assert self.panel1.n == self.panel2.n
        assert self.panel1.r == self.panel2.r
        if self.flange is not None:
            self.flange.lam = laminated_plate(self.flange.stack, plyts=self.flange.plyts,
                                            laminaprops=self.flange.laminaprops)
            self.flange.lam.calc_equivalent_properties()

        if self.base is not None:
            h = 0.5*sum(self.panel1.plyts) + 0.5*sum(self.panel2.plyts)
            hb = sum(self.bplyts)
            self.dpb = h/2. + hb/2.
            self.base.lam = laminated_plate(self.bstack, plyts=self.bplyts,
                                            laminaprops=self.blaminaprops,
                                            offset=(-h/2.-hb/2.))
            self.base.lam.calc_equivalent_properties()


    def calc_kC(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        """Calculate the linear constitutive stiffness matrix
        """
        self._rebuild()
        msg('Calculating kC... ', level=2, silent=silent)

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
                ktbf, krbf = calc_kt_kr(self.panel1, self.flange, 'ycte')
            else:
                ktbf, krbf = calc_kt_kr(self.base, self.flange, 'ycte')

            mod = db['bladestiff2d_clt_donnell_bardell']['connections']
            kC += mod.fkCss(ktbf, krbf, self.ys, a, b, m, n,
                            bay.x1u, bay.x1ur, bay.x2u, bay.x2ur,
                            bay.x1v, bay.x1vr, bay.x2v, bay.x2vr,
                            bay.x1w, bay.x1wr, bay.x2w, bay.x2wr,
                            bay.y1u, bay.y1ur, bay.y2u, bay.y2ur,
                            bay.y1v, bay.y1vr, bay.y2v, bay.y2vr,
                            bay.y1w, bay.y1wr, bay.y2w, bay.y2wr,
                            size, 0, 0)
            bf = self.flange.b
            kC += mod.fkCsf(ktbf, krbf, self.ys, a, b, bf, m, n, self.flange.m, self.flange.n,
                            bay.x1u, bay.x1ur, bay.x2u, bay.x2ur,
                            bay.x1v, bay.x1vr, bay.x2v, bay.x2vr,
                            bay.x1w, bay.x1wr, bay.x2w, bay.x2wr,
                            bay.y1u, bay.y1ur, bay.y2u, bay.y2ur,
                            bay.y1v, bay.y1vr, bay.y2v, bay.y2vr,
                            bay.y1w, bay.y1wr, bay.y2w, bay.y2wr,
                            self.flange.x1u, self.flange.x1ur, self.flange.x2u, self.flange.x2ur,
                            self.flange.x1v, self.flange.x1vr, self.flange.x2v, self.flange.x2vr,
                            self.flange.x1w, self.flange.x1wr, self.flange.x2w, self.flange.x2wr,
                            self.flange.y1u, self.flange.y1ur, self.flange.y2u, self.flange.y2ur,
                            self.flange.y1v, self.flange.y1vr, self.flange.y2v, self.flange.y2vr,
                            self.flange.y1w, self.flange.y1wr, self.flange.y2w, self.flange.y2wr,
                            size, 0, col0)
            kC += mod.fkCff(ktbf, krbf, a, bf, self.flange.m, self.flange.n,
                            self.flange.x1u, self.flange.x1ur, self.flange.x2u, self.flange.x2ur,
                            self.flange.x1v, self.flange.x1vr, self.flange.x2v, self.flange.x2vr,
                            self.flange.x1w, self.flange.x1wr, self.flange.x2w, self.flange.x2wr,
                            self.flange.y1u, self.flange.y1ur, self.flange.y2u, self.flange.y2ur,
                            self.flange.y1v, self.flange.y1vr, self.flange.y2v, self.flange.y2vr,
                            self.flange.y1w, self.flange.y1wr, self.flange.y2w, self.flange.y2wr,
                            size, row0, col0)

        if finalize:
            kC = finalize_symmetric_matrix(kC)
        self.kC = kC

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)


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

        kM = 0.
        if self.base is not None:
            kM += self.base.calc_kM(size=size, row0=0, col0=0, silent=True, finalize=False)
        if self.flange is not None:
            kM += self.flange.calc_kM(size=size, row0=row0, col0=col0, silent=True, finalize=False)

        if finalize:
            kM = finalize_symmetric_matrix(kM)
        self.kM = kM

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

