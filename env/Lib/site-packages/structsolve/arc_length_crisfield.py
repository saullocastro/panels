import numpy as np
from numpy import dot

from .static import solve
from .logger import msg, warn


TOO_SLOW = 0.01

def _solver_arc_length_crisfield(run, silent=False):
    r"""Arc-Length solver using Crisfield`s method

    """
    msg('Initialization...', level=1)
    lbd = run.initialInc
    last_lbd = 0.

    length_inc = 1.
    length = length_inc
    max_length = 10*run.maxInc
    modified_NR = run.modified_NR
    fext = run.calc_fext(inc=1., silent=silent)
    kC = run.calc_kC(silent=silent)
    c = solve(kC, lbd*fext, silent=silent)
    dc = c
    fint = kC*c
    last_fint = fint
    step_num = 1
    total_length = 0

    if modified_NR:
        if run.kT_initial_state:
            msg('Updating kT for initial imperfections...', level=1, silent=silent)
            kC = run.calc_kC(c=c*0, silent=silent, NLgeom=True)
            kG = run.calc_kG(c=c*0, silent=silent, NLgeom=True)
            kT_last = kC + kG
            msg('kT updated!', level=1)
        else:
            kT_last = kC
        compute_NL_matrices = False
    else:
        compute_NL_matrices = True
        kT_last = kC

    while step_num < 1000:
        msg('Attempting lambda %1.5f at step_num %d' % (lbd, step_num), level=1)
        min_Rmax = 1.e6
        prev_Rmax = 1.e6
        converged = False
        iteration = 0
        dlbd = 0

        kT = kT_last

        while True:
            iteration += 1
            msg('Iteration: %d' % iteration, level=2)
            if iteration > run.maxNumIter:
                warn('Maximum number of iterations achieved!', level=2)
                break

            # applying the arc-length constraint to find the new lbd and the
            # new c
            from scipy.sparse import csr_matrix
            varphi = 1 # spheric arc-length

            q = fext
            TMP = np.zeros((kT.shape[0]+1, kT.shape[1]+1), dtype=kT.dtype)
            TMP[:-1, :-1] = kT.toarray()
            TMP[:-1, -1] = -q
            TMP[-1, :-1] = 2*dc
            TMP[-1, -1] = 2*varphi**2*lbd*np.dot(q, q)
            TMP = csr_matrix(TMP)
            right_vec = np.zeros(q.shape[0]+1, dtype=q.dtype)

            fint = run.calc_fint(c, silent=silent)
            R = fint - (lbd + dlbd)*q
            A = -(np.dot(dc, dc) + varphi**2 * lbd**2 * np.dot(q, q) - length**2)
            right_vec[:-1] = R
            right_vec[-1] = A
            dc_dlbd = solve(TMP, -right_vec, silent=silent)
            dc = dc_dlbd[:-1]
            dlbd = dc_dlbd[-1]

            print('DEBUG lbd', lbd)
            print('DEBUG dlbd', dlbd)
            print('DEBUG compute_NL_matrices', compute_NL_matrices)
            lbd = lbd + dlbd
            c = c + dc

            # computing the Non-Linear matrices
            if compute_NL_matrices:
                print('HERE')
                kC = run.calc_kC(c=c, silent=silent, NLgeom=True)
                kG = run.calc_kG(c=c, silent=silent, NLgeom=True)
                kT = kC + kG
            else:
                if not modified_NR:
                    compute_NL_matrices = True
                #NOTE attempt to calculate fint more often than kT

            fint = run.calc_fint(c, silent=silent)

            # calculating the residual
            Rmax = np.abs(lbd*fext - fint).max()
            msg('Rmax = %1.5f' % Rmax, level=3)
            msg('lbd = %1.5f' % lbd, level=3)
            if Rmax <= run.absTOL:
                converged = True
                break
            if (Rmax > min_Rmax and Rmax > prev_Rmax and iteration > 2):
                warn('Diverged!', level=2)
                break
            else:
                min_Rmax = min(min_Rmax, Rmax)
            change_rate_Rmax = abs(1 - Rmax/prev_Rmax)
            if (iteration > 2 and change_rate_Rmax < TOO_SLOW):
                warn('Diverged! (convergence too slow)', level=2)
                break
            prev_Rmax = Rmax

        if converged:
            msg('Converged at increment with total length = %1.5f' % length, level=1)
            run.increments.append(lbd)
            run.cs.append(c.copy())

            last_fint = fint.copy()
            last_lbd = lbd
            step_num += 1
            new_length = length + length_inc
            length_inc *= 1.1
            #TODO
            while new_length >= max_length:
                new_length *= 0.95
            msg('Changing arc-length from %1.5f to %1.5f' % (length, new_length), level=1)
            length = new_length
            if modified_NR:
                msg('Updating kT...', level=1, silent=silent)
                kC = run.calc_kC(c=c, silent=silent, NLgeom=True)
                kG = run.calc_kG(c=c, silent=silent, NLgeom=True)
                kT = kC + kG
                msg('kT updated!', level=1)
            compute_NL_matrices = False
            kT_last = kT

        else:
            dlbd = 0
            if len(run.cs) > 0:
                c = run.cs[-1].copy()
            else:
                lbd = run.initialInc
                kT = kC
                c = solve(kT, lbd*fext, silent=silent)
            fint = last_fint
            old_length = length
            length -= length_inc
            factor = 0.9 # keep in the range 0 < factor < 1.
            length_inc *= factor
            msg('Diverged! Reducing arc-length from %1.5f to %1.5f' % (old_length, length), level=1)
            if length_inc < run.minInc:
                msg('Minimum arc-length achieved!', level=1)
                break

    print('DEBUG arc_length", step_num', step_num)
    msg('Finished Non-Linear Static Analysis')
    msg('with a total arc-length %1.5f' % length, level=1)
