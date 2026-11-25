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

The integration points, the numbers of terms and the boundary flags of a
shell do not change from one call of a kernel to the next, nor do the
tables. :func:`keep_tables`, applied last by :func:`tabulate_basis`,
rewrites the kernels with a ``shell`` argument such that they keep their
tables on the shell across calls, one row per integration point, with the
key of all the values the tables depend on, see :mod:`panels.basis_tables`:
the tables are filled at the first call, or after a change of the key, and
only read afterwards.

The scripts that generate the kernel modules from the symbolic integrands
apply :func:`tabulate_basis` or :func:`keep_tables` to their output;
``python -m panels.dev.tabulate <file.pyx> ...`` applies both to the
modules written by hand.

Notes
-----
An entry of a numerically integrated stiffness matrix between the terms
`(i, j)` and `(k, l)` of the Ritz series is a sum over the Gauss-Legendre
points of products of the approximation functions `P_i(\xi)`, `P_k(\xi)`,
`P_j(\eta)` and `P_l(\eta)`, and of their derivatives, see
:mod:`panels.bardell`. In the innermost loop over the pairs of terms each
function is evaluated `m^2 n^2` times per integration point, although it
depends on one index and one coordinate only. Tabulated, it is evaluated
`m` (or `n`) times per point, and the cost of the functions becomes
negligible against the `m^2 n^2` products of the assembly. Since only where the values are computed changes, and
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


KEEP_MARK = ('# tables of the approximation functions kept across calls, see '
             'panels/dev/tabulate.py')
IMPORT = 'from panels.basis_tables import lookup_table, store_table'
FILL_FOR = re.compile(r'^(?P<sp> *)for (?P<idx>\w+) in range\((?P<size>\w+)\):'
                      r'\s*$')
FILL_ROW = re.compile(r'^ *(?P<tab>\w+)\[(?P<row>\d+), (?P<idx>\w+)\] = '
                      r'(?P<rhs>.+?)\s*$')
NAME = re.compile(r'(?<![\w.])([A-Za-z_]\w*)(?=\s*(\(?))')
PURE = ('f', 'fp', 'fpp')
KEYWORDS = ('if', 'elif', 'else', 'while', 'return', 'with', 'raise', 'pass',
            'break', 'continue', 'print', 'cdef', 'def', 'assert')


def _statements(line):
    # the statements of a line of code, without the comment
    code = line.split('#')[0].strip()
    return [s.strip() for s in code.split(';') if s.strip()]


def _targets(stmt):
    """Names bound or modified by the statement ``stmt``"""
    mo = re.match(r'for (.+?) in ', stmt)
    if mo is not None:
        return [s.strip() for s in mo.group(1).split(',')]
    if stmt.split(' ')[0].rstrip(':') in KEYWORDS:
        return []
    mo = re.match(r'^([\w\s,.\[\]+\-*/]+?)\s*[+\-*/]?=(?!=)', stmt)
    if mo is None:
        return []
    lhs = re.sub(r'\[[^\]]*\]', '', mo.group(1))
    return [s.strip().split('.')[0] for s in lhs.split(',') if s.strip()]


def _names(expr):
    """Free names of the expression ``expr``; only the functions of the
    approximation are admitted as calls"""
    out = []
    for mo in NAME.finditer(expr):
        name, call = mo.group(1), mo.group(2)
        if call:
            if name not in PURE:
                raise ValueError('call of %s in a table' % name)
            continue
        if name not in out:
            out.append(name)
    return out


class _NotKept(Exception):
    """A table not filled in the loops over the integration points"""


def _keep_table(lines, func, tab, loop, shell):
    """Rewrite the table ``tab`` filled by the loop at ``loop`` such that it
    is kept on the shell across calls, see the module docstring"""
    def enclosing(n):
        out = []
        ind = _indent(lines[n])
        for k in range(n - 1, -1, -1):
            if _code(lines[k]) and _indent(lines[k]) < ind:
                ind = _indent(lines[k])
                out.append(k)
        return out

    def assigns(name, start, end):
        return [k for k in range(start, end) if _code(lines[k]) and
                any(name in _targets(s) for s in _statements(lines[k]))]

    head = FILL_FOR.match(lines[loop])
    idx, size = head['idx'], head['size']
    end = _block_end(lines, loop)
    rows = {}
    rhs = []
    for k in range(loop + 1, end):
        if not _code(lines[k]):
            continue
        mo = FILL_ROW.match(lines[k])
        if mo is None or mo['tab'] != tab or mo['idx'] != idx:
            raise ValueError('loop filling %s with other statements' % tab)
        rows[int(mo['row'])] = k
        rhs.append(mo['rhs'])
    # allocation and declaration
    alloc = [k for k, line in enumerate(lines) if re.match(
             r'^ +%s = np\.zeros\(\((\w+), (\w+)\), dtype=DOUBLE\)\s*$' % tab,
             line)]
    if len(alloc) != 1:
        raise ValueError('allocation of %s not found' % tab)
    nrows, size2 = re.match(r'^ +%s = np\.zeros\(\((\w+), (\w+)\)' % tab,
                            lines[alloc[0]]).groups()
    if size2 != size or sorted(rows) != list(range(int(nrows))):
        raise ValueError('shape of %s differs from its filling' % tab)
    decl = [k for k, line in enumerate(lines) if re.match(
            r'^ +cdef double \[:, ::1\] .*\b%s\b' % tab, line)]
    if len(decl) != 1:
        raise ValueError('declaration of %s not found' % tab)
    # the loops over the integration points and the nogil block
    outer = enclosing(loop)
    pts = [k for k in outer if FOR.match(lines[k])
           and FOR.match(lines[k])['var'].startswith('pt')]
    nogil = [k for k in outer if lines[k].strip() == 'with nogil:']
    if not pts or len(nogil) != 1:
        # e.g. the functions at a single point given as argument
        raise _NotKept(tab)
    if not shell:
        # e.g. the connections, whose tables depend on several panels
        raise _NotKept(tab)
    nogil = nogil[0]
    first = pts[-1]
    region_end = _block_end(lines, first)
    ptvars = {FOR.match(lines[k])['var']: k for k in pts}
    # the values the table depends on, through the names assigned in the
    # loops over the points before it is filled
    key, dep, todo, seen = [], [], _names(' '.join(rhs)), set()
    while todo:
        name = todo.pop(0)
        if name in seen or name == idx:
            continue
        seen.add(name)
        if name in ptvars:
            dep.append(name)
            continue
        before = assigns(name, first + 1, loop)
        if before:
            for k in before:
                parent = enclosing(k)[0]
                if parent not in pts:
                    raise ValueError('%s assigned conditionally' % name)
                for s in _statements(lines[k]):
                    if name in _targets(s):
                        if s.startswith('for '):
                            raise ValueError('%s is a loop variable' % name)
                        value = re.split(r'(?<![=!<>])=(?!=)', s, maxsplit=1)[1]
                        if k == before[0] and name in _names(value):
                            raise ValueError('%s depends on a previous '
                                             'point' % name)
                        todo += _names(value)
            continue
        # a value of the function, which must not change in the nogil block
        if assigns(name, nogil, _block_end(lines, nogil)):
            raise ValueError('%s modified in the nogil block' % name)
        key.append(name)
    if len(dep) > 1:
        raise ValueError('%s depends on several loops over the points' % tab)
    if dep:
        pt = dep[0]
        mo = re.match(r' *for \w+ in range\((\w+)\):', lines[ptvars[pt]])
        if mo is None:
            raise ValueError('number of points of %s not found' % pt)
        npts = mo.group(1)
    else:
        pt, npts = '0', '1'
    for name in (size, npts):
        if not name.isdigit() and name not in key:
            key.insert(0, name)
    # reads of the table only in the loops over the points
    ref = re.compile(r'\b%s\[' % tab)
    for k, line in enumerate(lines):
        if ref.search(line) and k not in (alloc[0], ) and not (
                first < k < region_end):
            raise ValueError('%s read outside the loops over the points' % tab)

    if not decl[0] < alloc[0] < nogil:
        raise ValueError('%s declared or allocated after the nogil block'
                         % tab)
    q = "'%s'" % tab
    # the store after the nogil block, the fill only for a new table, the
    # reads with the index of the point
    nogil_end = _block_end(lines, nogil)
    sp = ' '*_indent(lines[nogil])
    lines[nogil_end:nogil_end] = [
        '{0}if {1}_new:'.format(sp, tab),
        "{0}    store_table(shell, __name__, '{1}', {2}, {3}_key, {3})".format(
            sp, func, q, tab)]
    for k in range(first + 1, region_end):
        lines[k] = ref.sub('%s[%s, ' % (tab, pt), lines[k])
    for k in range(loop, end):
        if _code(lines[k]):
            lines[k] = '    ' + lines[k]
    lines[loop:loop] = [head['sp'] + 'if %s_new:' % tab]
    lines[nogil:nogil] = [
        sp + KEEP_MARK,
        '{0}{1}_key = ({2}{3})'.format(sp, tab, ', '.join(key),
                                       ',' if len(key) == 1 else ''),
        "{0}{1}, {1}_new = lookup_table(shell, __name__, '{2}', {3}, "
        "{1}_key, ({4}, {5}, {6}))".format(sp, tab, func, q, npts, nrows,
                                            size)]
    del lines[alloc[0]]
    a = alloc[0]
    if 0 < a < len(lines) and not lines[a - 1].strip() and not lines[a].strip():
        del lines[a]
    d = decl[0]
    head_decl, tabs = lines[d].split('] ', 1)
    tabs = [s.strip() for s in tabs.split('#')[0].split(',')]
    tabs.remove(tab)
    sp = ' '*_indent(lines[d])
    new = [sp + 'cdef double [:, :, ::1] ' + tab, sp + 'cdef int %s_new' % tab]
    if tabs:
        new.insert(0, head_decl + '] ' + ', '.join(tabs))
    lines[d:d + 1] = new


def _keep_function(text):
    """Keep across calls the tables of approximation functions of the
    kernel ``text``, a ``def`` function"""
    if 'lookup_table(' in text:
        return text
    sig = re.match(r'def \w+\((.*?)\)\s*:', text, flags=re.S).group(1)
    lines = text.split('\n')
    func = re.match(r'def (\w+)', text).group(1)
    skip = set()
    while True:
        found = None
        for n, line in enumerate(lines):
            head = FILL_FOR.match(line)
            if head is None or not _code(line) or n in skip:
                continue
            body = [lines[k] for k in range(n + 1, _block_end(lines, n))
                    if _code(lines[k])]
            rows = [FILL_ROW.match(b) for b in body]
            if (not body or any(r is None for r in rows)
                    or any(r['idx'] != head['idx'] for r in rows)
                    or not any(ANY_CALL.search(r['rhs']) for r in rows)):
                continue
            tab = rows[0]['tab']
            if any(r['tab'] != tab for r in rows):
                raise ValueError('loop filling several tables')
            if re.search(r'\bcdef double \[:, :, ::1\] %s\b' % tab,
                         '\n'.join(lines)):
                continue
            found = (n, tab)
            break
        if found is None:
            break
        n, tab = found
        fills = [k for k, line in enumerate(lines)
                 if re.match(r'^ +%s\[\d+, \w+\] = ' % tab, line)]
        if any(k > _block_end(lines, n) or k < n for k in fills):
            raise ValueError('%s filled in several places' % tab)
        try:
            _keep_table(lines, func, tab, n,
                        re.search(r'\bshell\b', sig) is not None)
        except _NotKept:
            skip.add(n)
        except ValueError as e:
            raise ValueError('%s: %s' % (func, e)) from None
    return '\n'.join(lines)


def keep_tables(source):
    r"""Keep the tables of approximation functions of the kernels of the
    module ``source`` across calls, on the shell

    Each table ``TAB`` of shape ``(rows, terms)`` filled by a loop ``for
    IDX in range(SIZE):`` of rows ``TAB[ROW, IDX] = EXPR`` in the loops
    over the integration points of a ``nogil`` block becomes a table of
    shape ``(points, rows, terms)``, with one entry per point of the loop
    over the integration points that its coordinate depends on, looked up
    with :func:`panels.basis_tables.lookup_table` before the ``nogil``
    block, filled only when it is new, stored with
    :func:`panels.basis_tables.store_table` after the block and read as
    ``TAB[PT, ROW, IDX]``. The key of the table gathers the names that
    ``EXPR`` depends on, through the assignments in the loops over the
    points, e.g. ``xis, xi1, xi2`` of the coordinate ``xi``, the boundary
    flags and the dimensions, together with ``SIZE`` and the number of
    points; a ``ValueError`` is raised when this dependence cannot be
    established, e.g. for a name assigned conditionally, or modified in the
    ``nogil`` block. The functions already rewritten are left unchanged,
    and so are the tables filled outside the loops over the integration
    points, e.g. at a single point given as argument.

    Parameters
    ----------
    source : str
        The text of a ``.pyx`` module, with tabulated kernels, or of a part
        of a module.

    Returns
    -------
    source : str
        The rewritten text, with the import of the functions of
        :mod:`panels.basis_tables` after the line ``from panels import
        ...``, when there is one and the import is missing.

    """
    parts = re.split(r'(?=^(?:def|cdef|cpdef) )', source, flags=re.M)
    new = ''.join(_keep_function(part) if part.startswith('def ')
                  else part for part in parts)
    # a part of a module, e.g. a block of kernels written by a generator into
    # an existing module, may have no imports, then the generator applies
    # keep_tables to the whole module
    mo = re.search(r'^from panels import .*$', new, flags=re.M)
    if 'lookup_table(' in new and IMPORT not in new and mo is not None:
        new = new[:mo.end()] + '\n' + IMPORT + new[mo.end():]
    return new


def tabulate_basis(source):
    r"""Tabulate the approximation functions of every kernel of the module
    ``source``, see the module docstring, and keep the tables across calls,
    see :func:`keep_tables`

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
    return keep_tables(''.join(_tabulate_function(part)
                               if part.startswith('def ') else part
                               for part in parts))


if __name__ == '__main__':
    for path in sys.argv[1:]:
        with open(path, encoding='utf-8') as f:
            text = f.read()
        new = tabulate_basis(text)
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(new)
        print(path, 'unchanged' if new == text else 'rewritten')
