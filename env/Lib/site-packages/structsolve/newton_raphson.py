import numpy as np

from .logger import msg, warn
from .static import solve


def _solver_NR(an, silent=False, initialInc=None):
    """Newton-Raphson solver

    """
    msg('Initializing...', level=1, silent=silent)

    modified_NR = an.modified_NR
    if initialInc is None:
        initialInc = an.initialInc
    inc = initialInc
    total = inc
    once_at_total = False
    max_total = 0.

    fext = an.calc_fext(inc=1., silent=True)
    kC = an.calc_kC(silent=silent)
    c = solve(kC, total*fext, silent=True)
    kC = an.calc_kC(c=c, NLgeom=True, silent=True)
    kG = an.calc_kG(c=c, NLgeom=True, silent=True)
    kT = kC + kG
    dc = 0 * c
    c_last = 0 * c
    compute_NL_matrices = False

    step_num = 1

    while True:
        msg('Step %d, attempting time %1.5f' % (step_num, total), level=1, silent=silent)
        min_Rmax = 1.e6
        prev_Rmax = 1.e6
        converged = False
        iteration = 0

        iter_NR = 0
        while True:
            iteration += 1
            msg('Iteration: {}'.format(iteration), level=2, silent=silent)
            if iteration > an.maxNumIter:
                warn('Maximum number of iterations achieved!', level=2, silent=silent)
                break

            if compute_NL_matrices or (an.kT_initial_state and step_num == 1 and
                    iteration == 1) or iter_NR == (an.compute_every_n - 1):
                iter_NR = 0
                kC = an.calc_kC(c=(c + dc), NLgeom=True, silent=True)
                kG = an.calc_kG(c=(c + dc), NLgeom=True, silent=True)
                kT = kC + kG
                compute_NL_matrices = False
            else:
                iter_NR += 1
                if not modified_NR:
                    compute_NL_matrices = True

            fint = an.calc_fint(c=(c + dc), silent=True)
            R = total*fext - fint

            # convergence criteria:
            # - maximum residual force Rmax
            Rmax = np.abs(R).max()

            if iteration >= 2 and Rmax < an.absTOL:
                converged = True
                break
            if (Rmax > prev_Rmax and Rmax > min_Rmax and iteration > 3):
                warn('Diverged - Rmax value significantly increased', level=2, silent=silent)
                break
            else:
                min_Rmax = min(min_Rmax, Rmax)
            change_rate_Rmax = abs(1 - Rmax/prev_Rmax)
            if (iteration > 2 and change_rate_Rmax < an.too_slow_TOL):
                warn('Diverged - convergence too slow', level=2, silent=silent)
                break
            prev_Rmax = Rmax

            varc = solve(kT, R, silent=True)

            eta1 = 0.
            eta2 = 1.
            if an.line_search:
                msg('Performing line-search... ', level=2, silent=silent)
                iter_line_search = 0
                while True:
                    c1 = c + dc + eta1*varc
                    c2 = c + dc + eta2*varc
                    fint1 = an.calc_fint(c=c1, silent=True)
                    fint2 = an.calc_fint(c=c2, silent=True)
                    R1 = total*fext - fint1
                    R2 = total*fext - fint2
                    s1 = varc.dot(R1)
                    s2 = varc.dot(R2)
                    eta_new = (eta2-eta1)*(-s1/(s2-s1)) + eta1
                    eta1 = eta2
                    eta2 = eta_new
                    eta2 = min(max(eta2, 0.2), 10.)
                    if abs(eta2-eta1) < 0.01:
                        break
                    iter_line_search += 1
                    if iter_line_search == an.max_iter_line_search:
                        eta2 = 1.
                        warn('maxinum number of iterations', level=3, silent=silent)
                        break
                msg('finished line-search', level=2, silent=silent)
            dc = dc + eta2*varc

        if converged:
            msg('Converged at time %1.5f' % total, level=2, silent=silent)
            c_last = c.copy()
            c = c + dc
            an.cs.append(c.copy()) #NOTE copy required
            an.increments.append(total)
            total += inc
            finished = False
            if abs(total - 1) < 1e-3:
                finished = True
            else:
                factor = 1.1111
                if once_at_total:
                    inc_new = min(factor*inc, an.maxInc, (1.-total)/2)
                else:
                    inc_new = min(factor*inc, an.maxInc, 1.-total)
                inc = inc_new
                total += inc
                total = min(1, total)
                step_num += 1
            if finished:
                break

        else:
            max_total = max(max_total, total)
            while True:
                factor = 0.9
                msg('Reseting step with reduced time increment', level=1, silent=silent)
                if abs(total - 1) < 1e-3:
                    once_at_total = True
                total -= inc
                inc *= factor
                if inc < an.minInc:
                    msg('Minimum step size achieved!', level=1, silent=silent)
                    break
                total += inc
                if total >= max_total:
                    continue
                else:
                    break

        if inc < an.minInc:
            msg('Minimum step size of %1.5f achieved!' % an.minInc, level=1, silent=silent)
            break

        dc = c - c_last

        kC = an.calc_kC(c=c, NLgeom=True, silent=True)
        kG = an.calc_kG(c=c, NLgeom=True, silent=True)
        kT = kC + kG
        compute_NL_matrices = False

    msg('Finished Non-Linear Static Analysis', silent=silent)
    msg('    total time %1.5f' % total, level=1, silent=silent)

