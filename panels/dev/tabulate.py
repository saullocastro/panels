r"""Tabulation of the approximation functions in the numerical kernels

The numerically integrated kernels, e.g. ``fkC_num`` of
``panels/models/<model>_num.pyx`` or ``fkCSB11_dmg`` of
``panels/multidomain/connections/kCSB_dmg.pyx``, evaluate the approximation
functions inside their loops over the terms, e.g. ``fBu = f(k, xi, x1u,
x1ur, x2u, x2ur)`` in the innermost loop over the pairs of terms of a
stiffness matrix. :func:`tabulate_basis` rewrites a kernel such that each
function is evaluated once per integration point and term, in tables such
as ``ftab[k, i]`` along
`x` and ``gtab[k, j]`` along `y`, filled at the start of the loop over the
integration points and read in the loops over the terms. The arithmetic of
the integrands is unchanged, so are the results, bit for bit.

A call ``NAME = f(IDX, COORD, x1F, x1Fr, x2F, x2Fr)``, with ``f``, ``fp`` or
``fpp``, the boundary flags ``x1F...`` along `x` or ``y1F...`` along `y` of
the field ``F`` and an optional suffix of the panel, e.g. ``x1u1, x1ur1,
x2u1, x2ur1``, becomes ``NAME = ftab1[k, IDX]``, with one table per
direction, suffix and coordinate ``COORD``, whose number of columns is that
of the loop ``for IDX in range(...)`` of the kernel, and which is filled
with ``IDX`` as loop variable. Without a loop over the integration points enclosing the calls,
e.g. along a line of constant `\xi` of the connections, the tables are filled
once, before the outermost loop over the terms. A kernel is left unchanged when it has
no such call, or when it is already tabulated. A call in another form raises a ``ValueError``.

The scripts that generate the kernel modules from the symbolic integrands
apply it to their output; ``python -m panels.dev.tabulate <file.pyx> ...``
applies it to the modules written by hand.

Notes
-----
An entry of a numerically integrated stiffness matrix between the terms
`(i, j)` and `(k, l)` of the Ritz series is a sum over the Gauss-Legendre
points of products of the approximation functions `f_i(\xi)`, `f_k(\xi)`,
`g_j(\eta)` and `g_l(\eta)`, and of their derivatives, see
:mod:`panels.bardell`. In the innermost loop over the pairs of terms each
function is evaluated `m^2 n^2` times per integration point, although it
depends on one index and one coordinate only. Tabulated, it is evaluated
`m` (or `n`) times per point, and the cost of the functions becomes
negligible against the `m^2 n^2` products of the assembly, see
:ref:`tabulation`. Since only where the values are computed changes, and
not the operations on them, the results are identical.

"""
import re
import sys

CALL = re.compile(r'(?P<lhs>\w+) = (?P<func>fpp|fp|f)\((?P<idx>\w+), '
                  r'(?P<coord>[\w.+-]+), (?P<x>[xy])1(?P<field>phix|phiy|u|v|w)'
                  r'(?P<suf>\w*), (?P=x)1(?P=field)r(?P=suf), '
                  r'(?P=x)2(?P=field)(?P=suf), (?P=x)2(?P=field)r(?P=suf)\)')
ANY_CALL = re.compile(r'(?<![\w.])(?:fpp|fp|f)\(')
FOR = re.compile(r'^(?P<sp> *)for (?P<var>\w+) in ')
MARK = '# approximation functions evaluated once, see panels/dev/tabulate.py'
TODO = re.compile(r'^ +#TODO (?:save in buffer|put these in a lookup vector)\s*$')


def _indent(line):
    return len(line) - len(line.lstrip(' '))


def _code(line):
    s = line.strip()
    return s and not s.startswith('#')


def _block_end(lines, start):
    """Index after the block of the statement at ``start``"""
    ind = _indent(lines[start])
    end = start + 1
    while end < len(lines) and (not _code(lines[end])
                                or _indent(lines[end]) > ind):
        end += 1
    # trailing blank and comment lines do not belong to the block
    while end - 1 > start and not _code(lines[end - 1]):
        end -= 1
    return end


def _tabulate_function(text):
    if MARK in text or re.search(r'\b[fg]tab\w*\[', text):
        return text
    lines = text.split('\n')
    calls = []
    for n, line in enumerate(lines):
        if not _code(line):
            continue
        found = list(CALL.finditer(line))
        if len(found) != len(ANY_CALL.findall(line)):
            raise ValueError('approximation functions not in the expected '
                             'form: %s' % line.strip())
        calls += [(n, mo) for mo in found]
    if not calls:
        return text

    # the innermost loop over the integration points enclosing the calls
    def enclosing(n):
        out = []
        ind = _indent(lines[n])
        for k in range(n - 1, -1, -1):
            if _code(lines[k]) and _indent(lines[k]) < ind:
                ind = _indent(lines[k])
                out.append(k)
        return out

    points = set()
    for n, _ in calls:
        loops = [k for k in enclosing(n) if FOR.match(lines[k])
                 and FOR.match(lines[k])['var'].startswith('pt')]
        points.add(loops[0] if loops else None)
    if len(points) != 1:
        raise ValueError('approximation functions in several loops over the '
                         'integration points')
    # without a loop over the integration points, e.g. along a line of
    # constant xi, the tables are filled once, before the loops over the
    # terms
    P = points.pop()
    first = min(n for n, _ in calls)
    if P is None:
        # the outermost loop over the terms enclosing the first call
        S = [k for k in enclosing(first) if FOR.match(lines[k])][-1]
        body = _indent(lines[S])
    else:
        # the statement of the loop body that contains the first call
        body = _indent(lines[P]) + 4
        S = [k for k in enclosing(first) + [first]
             if _indent(lines[k]) == body and k > P][0]
    for n, mo in calls:
        if n < S:
            raise ValueError('unexpected structure')
    # the coordinates must not be changed after the tables are filled
    for _, mo in calls:
        assign = re.compile(r'^ *%s = ' % mo['coord'])
        for k in range(S, len(lines) if P is None else _block_end(lines, P)):
            if assign.match(lines[k]):
                raise ValueError('%s assigned after the first call'
                                 % mo['coord'])

    def terms(n, idx):
        # the number of terms of the loop over idx enclosing the call n
        for k in enclosing(n):
            mo = re.match(r' *for %s in range\((\w+)\):' % idx, lines[k])
            if mo is not None:
                return mo.group(1)
        raise ValueError('loop over %s not found' % idx)

    tables = {}
    for n, mo in calls:
        key = (mo['x'], mo['suf'], mo['coord'])
        tab = tables.setdefault(key, dict(rows=[], idx=mo['idx'],
                                          size=terms(n, mo['idx'])))
        if terms(n, mo['idx']) != tab['size']:
            raise ValueError('several numbers of terms for %s' % (key, ))
        row = (mo['func'], mo['field'])
        if row not in tab['rows']:
            tab['rows'].append(row)
    coords = {}
    for (x, suf, coord) in tables:
        coords.setdefault((x, suf), []).append(coord)

    def name(key):
        x, suf, coord = key
        extra = ('' if len(coords[(x, suf)]) == 1 else '_' +
                 coord.replace('-', 'm').replace('+', 'p').replace('.', 'd'))
        return '{0}tab{1}{2}'.format('f' if x == 'x' else 'g', suf, extra)

    def size(key):
        return tables[key]['size']

    for n, mo in reversed(calls):
        key = (mo['x'], mo['suf'], mo['coord'])
        k = tables[key]['rows'].index((mo['func'], mo['field']))
        new = '{0} = {1}[{2}, {3}]'.format(mo['lhs'], name(key), k, mo['idx'])
        lines[n] = lines[n][:mo.start()] + new + lines[n][mo.end():]

    sp = ' '*body
    fill = [sp + MARK]
    for key, tab in tables.items():
        x, suf, coord = key
        idx = tab['idx']
        fill.append('{0}for {1} in range({2}):'.format(sp, idx, size(key)))
        for k, (func, field) in enumerate(tab['rows']):
            fill.append('{0}    {1}[{2}, {3}] = {4}({3}, {5}, {6}1{7}{8}, '
                        '{6}1{7}r{8}, {6}2{7}{8}, {6}2{7}r{8})'.format(sp,
                        name(key), k, idx, func, coord, x, field, suf))
    fill.append('')
    lines[S:S] = fill

    # allocation before the statement of the function body enclosing P
    top = [k for k in enclosing(S) + [S] if _indent(lines[k]) == 4][0]
    alloc = ['    {0} = np.zeros(({1}, {2}), dtype=DOUBLE)'.format(name(key),
             len(tab['rows']), size(key)) for key, tab in tables.items()]
    lines[top:top] = alloc + ['']
    # declaration before the first one of the function
    decl = [k for k, line in enumerate(lines) if line.startswith('    cdef ')][0]
    lines[decl:decl] = ['    cdef double [:, ::1] ' + ', '.join(
                        name(key) for key in tables)]
    lines = [line for line in lines if not TODO.match(line)]
    return '\n'.join(lines)


def tabulate_basis(source):
    r"""Tabulate the approximation functions of every kernel of the module
    ``source``, see the module docstring

    The source is split before each line starting with ``def``, ``cdef``
    or ``cpdef``; only the parts that are ``def`` functions are rewritten,
    the others are kept as they are. A ``ValueError`` is raised for a
    kernel whose calls are not in the expected form.

    Parameters
    ----------
    source : str
        The text of a ``.pyx`` module.

    Returns
    -------
    source : str
        The rewritten text, equal to ``source`` when no kernel is changed.

    """
    #NOTE the cdef functions are not rewritten, their arguments differ
    parts = re.split(r'(?=^(?:def|cdef|cpdef) )', source, flags=re.M)
    return ''.join(_tabulate_function(part) if part.startswith('def ')
                   else part for part in parts)


if __name__ == '__main__':
    for path in sys.argv[1:]:
        with open(path) as f:
            text = f.read()
        new = tabulate_basis(text)
        with open(path, 'w', newline='\n') as f:
            f.write(new)
        print(path, 'unchanged' if new == text else 'tabulated')
