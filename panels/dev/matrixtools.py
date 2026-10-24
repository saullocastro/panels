r"""Symbolic matrices printed as source code (:mod:`panels.dev.matrixtools`)

Functions used by the scripts that derive the integrands of the structural
matrices with :mod:`sympy`: the non-zero entries of a matrix are written as
lines of Python or Cython code, which are returned and optionally written
to a text file ``print_<name>.txt`` in the current directory.

"""
import re

import numpy as np
import sympy
from sympy import collect, var, sin, cos, pi


def pow2mult(instr):
    """Power to multiplications

    Substitutes x**5 or pow(x, 5) by x*x*x*x*x

    Parameters
    ----------
    instr : str
        The input string

    Returns
    -------
    outstr : str
        The output string

    """
    old_new = []
    for p in re.findall(r'\w+\*\*\d+', instr):
        var, exp = p.split('**')
        new = '(' + '*'.join([var]*int(exp)) + ')'
        old_new.append([p, new])
    # putting longer patterns first to avoid wrong
    # substitutions
    old_new = sorted(old_new, key=lambda x: len(x[0]))[::-1]
    outstr = instr
    for old, new in old_new:
        outstr = outstr.replace(old, new)

    old_new = []
    for p in re.findall(r'pow\(\w+,\s*\w+\)', instr):
        var, exp = p.split('pow')[1].split('(')[1].split(')')[0].split(',')
        new = '(' + '*'.join([var]*int(exp)) + ')'
        old_new.append([p, new])
    for old, new in old_new:
        outstr = outstr.replace(old, new)

    return outstr


def mprint_as_sparse(m, mname, sufix, subs=None,
        header=None, print_file=True, collect_for=None,
        pow_by_mul=True, full_symmetric=False):
    r"""Print the non-zero entries of a matrix as code of a sparse matrix

    The output starts with ``header``, if given, then the lines ``# NAME``
    and ``# NAME_num=NUM``, with ``NAME`` the name ``mname_sufix`` (or
    ``mname`` when ``sufix`` is ``''``) and ``NUM`` the number of entries
    printed. Each non-zero entry ``(i, j)`` gives the lines ``c += 1``,
    ``MNAMEr[c] = row+i``, ``MNAMEc[c] = col+j`` and ``MNAMEv[c] += VALUE``,
    with ``MNAME`` the value of ``mname``. With ``subs``, the substitutions
    whose symbols appear in the output are listed at its start, after the
    line ``subs``.

    Parameters
    ----------
    m : sympy.Matrix
        The 2-D matrix of symbolic expressions.
    mname : str
        Prefix of the names of the arrays of rows, columns and values.
    sufix : str
        ``''`` or two characters, for the rows and the columns; a ``'0'``
        prints the absolute index ``i`` (or ``j``) instead of ``row+i`` (or
        ``col+j``). Also used in the name of the matrix and of the file.
    subs : dict, optional
        Substitutions ``{expression: symbol}`` applied to each entry with
        ``subs()``. Default ``None``.
    header : str, optional
        First line of the output. Default ``None``.
    print_file : bool, optional
        Whether the output is also written to the file ``print_NAME.txt``
        in the current directory. Default ``True``.
    collect_for : optional
        Passed to :func:`sympy.collect` with ``evaluate=False``, each
        collected term being printed on its own line. Default ``None``.
    pow_by_mul : bool, optional
        Whether the powers are written as multiplications, see
        :func:`pow2mult`. Not applied with ``collect_for``. Default ``True``.
    full_symmetric : bool, optional
        Whether only the entries with ``i <= j`` are printed, always with
        absolute indices. Default ``False``.

    Returns
    -------
    string : str
        The output.

    See Also
    --------
    mprint_as_array

    """
    if sufix == '':
        left = right = '1'
        namesufix = '{0}'.format(mname)
    else:
        left, right = sufix
        namesufix = '{0}_{1}'.format(mname, sufix)
    filename = 'print_{0}.txt'.format(namesufix)
    ls = []
    if header:
        ls.append(header)
    ls.append('# {0}'.format(namesufix))

    if full_symmetric:
        num = len([None for (i, j), v in np.ndenumerate(m) if (v and i <= j)])
    else:
        num = len([i for i in list(m) if i])

    ls.append('# {0}_num={1}'.format(namesufix, num))

    for (i, j), v in np.ndenumerate(m):
        if v:
            if full_symmetric and i > j:
                continue

            if subs:
                v = v.subs(subs)
            ls.append('c += 1')

            if left == '0' or full_symmetric:
                ls.append('{mname}r[c] = {i}'.format(mname=mname, i=i))
            else:
                ls.append('{mname}r[c] = row+{i}'.format(mname=mname, i=i))

            if right == '0' or full_symmetric:
                ls.append('{mname}c[c] = {j}'.format(mname=mname, j=j))
            else:
                ls.append('{mname}c[c] = col+{j}'.format(mname=mname, j=j))

            if collect_for is not None:
                v = collect(v, collect_for, evaluate=False)
                ls.append('{mname}v[c] +='.format(mname=mname))
                for k, expr in v.items():
                    ls.append('#   collected for {k}'.format(k=k))
                    ls.append('    {expr}'.format(expr=k*expr))
            else:
                if pow_by_mul:
                    v = pow2mult(str(v))
                ls.append('{mname}v[c] += {v}'.format(mname=mname, v=v))

    string = '\n'.join(ls)

    if subs:
        items = sorted(subs.items(), key= lambda x: str(x[1]))
        items = [(k, v) for k, v in items if str(v) in string]
        if items:
            ls_header = []
            ls_header.append('subs\n')
            for k, v in items:
                ls_header.append('{0} = {1}'.format(v, k))
            ls_header.append('\n')

            string = '\n'.join(ls_header + ls)

    if print_file:
        with open(filename, 'w') as f:
            f.write(string)
    return string

def mprint_as_array(m, mname, sufix, use_cse=False,
                    header=None, print_file=True, collect_for=None,
                    pow_by_mul=True, order='C', op='+='):
    r"""Print the non-zero entries of a matrix as code of a flat array

    The output starts with ``header``, if given, then, with ``use_cse``, the
    declarations ``cdef double ...`` of the common subexpressions, nine per
    line, and their assignments after the line ``# subs``, then the lines
    ``# NAME`` and ``# NAME_num=NUM``, with ``NAME`` the name
    ``mname_sufix`` and ``NUM`` the number of non-zero entries. Each
    non-zero entry of flat index ``i`` gives the line ``MNAME[pos+i] OP
    VALUE``, with ``MNAME`` the value of ``mname`` and ``OP`` that of
    ``op``.

    Parameters
    ----------
    m : sympy.Matrix
        The matrix of symbolic expressions. With ``use_cse=True`` its
        entries are replaced in place by the reduced expressions.
    mname : str
        Name of the flat array.
    sufix : str
        Used in the name of the matrix and of the file.
    use_cse : bool, optional
        Whether the common subexpressions are eliminated with
        :func:`sympy.cse`. Default ``False``.
    header : str, optional
        First line of the output. Default ``None``.
    print_file : bool, optional
        Whether the output is also written to the file ``print_NAME.txt``
        in the current directory. Default ``True``.
    collect_for : optional
        Passed to :func:`sympy.collect` with ``evaluate=False``, each
        collected term being printed on its own line. Default ``None``.
    pow_by_mul : bool, optional
        Whether the powers ``x**n`` are written as multiplications. Not
        applied with ``collect_for``. Default ``True``.
    order : str, optional
        ``'C'`` for the flat index by rows or ``'F'`` by columns. Another
        value fails, the iterator over the entries not being defined.
        Default ``'C'``.
    op : str, optional
        The assignment operator of each line. Default ``'+='``.

    Returns
    -------
    string : str
        The output.

    See Also
    --------
    mprint_as_sparse

    """
    ls = []
    if use_cse:
        subs, m_list = sympy.cse(m)
        for i, v in enumerate(m_list):
            m[i] = v
    namesufix = '{0}_{1}'.format(mname, sufix)
    filename = 'print_{0}.txt'.format(namesufix)
    if header:
        ls.append(header)
    if use_cse:
        ls.append('# cdefs')
        num = 9
        for i, sub in enumerate(subs[::num]):
            ls.append('cdef double ' + ', '.join(
                        map(str, [j[0] for j in subs[num*i:num*(i+1)]])))
        ls.append('# subs')
        for sub in subs:
            ls.append('{0} = {1}'.format(*sub))
    ls.append('# {0}'.format(namesufix))
    num = len([i for i in list(m) if i])
    ls.append('# {0}_num={1}'.format(namesufix, num))
    if order == 'C':
        miter = enumerate(np.ravel(m))
    elif order == 'F':
        miter = enumerate(np.ravel(m.T))
    c = -1
    miter = list(miter)
    for i, v in miter:
        if v:
            if collect_for is not None:
                v = collect(v, collect_for, evaluate=False)
                ls.append('{0}[pos+{1}] +='.format(mname, i))
                for k, expr in v.items():
                    ls.append('#   collected for {k}'.format(k=k))
                    ls.append('    {expr}'.format(expr=k*expr))
            else:
                if pow_by_mul:
                    v = str(v)
                    for p in re.findall(r'\w+\*\*\d+', v):
                        var, exp = p.split('**')
                        v = v.replace(p, '(' + '*'.join([var]*int(exp)) + ')')
                ls.append('{0}[pos+{1}] {2} {3}'.format(mname, i, op, v))
    string = '\n'.join(ls)
    if print_file:
        with open(filename, 'w') as f:
            f.write(string)
    return string
