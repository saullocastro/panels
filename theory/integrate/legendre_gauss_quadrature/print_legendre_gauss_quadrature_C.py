from itertools import izip

ppath = 'out_legendre_gauss_quadrature_points.txt'
wpath = 'out_legendre_gauss_quadrature_weights.txt'
out = {}
with open(ppath) as pfile, open(wpath) as wfile:
    myiter = izip(pfile, wfile)
    for pline, wline in myiter:
        if pline.startswith('"n'):
            n = int(pline.replace('"', '').split('=')[1])
            out[n] = []
            for i in range(n):
                pt, wg = myiter.next()
                out[n].append([pt.strip(), wg.strip()])

for v in out.values():
    v.sort(key=lambda x: float(x[0]))

with open('../../../compmech/lib/src/legendre_gauss_quadrature.c', 'wb') as f:
    f.write('// Legendre-Gauss Quadrature Points and Weights\n\n')
    f.write('#if defined(_WIN32) || defined(__WIN32__)\n')
    f.write('  #define EXPORTIT __declspec(dllexport)\n')
    f.write('#else\n')
    f.write('  #define EXPORTIT\n')
    f.write('#endif\n\n')
    f.write('EXPORTIT void leggauss_quad(int n, double *points, double *weights) {\n')
    f.write('    switch(n) {\n')
    for k, v in out.items():
        f.write('    case {0}:\n'.format(k))
        for i, (pt, wg) in enumerate(v):
            f.write('        points[{0}] = {1};\n'.format(i, pt))
        for i, (pt, wg) in enumerate(v):
            f.write('        weights[{0}] = {1};\n'.format(i, wg))
        f.write('        return;\n')
    f.write('    }\n')
    f.write('}\n')

with open('../../../compmech/include/legendre_gauss_quadrature.h', 'wb') as g:
    g.write('#if defined(_WIN32) || defined(__WIN32__)\n')
    g.write('  #define IMPORTIT __declspec(dllimport)\n')
    g.write('#else\n')
    g.write('  #define IMPORTIT\n')
    g.write('#endif\n\n')
    g.write('#ifndef LEGENDRE_GAUSS_QUADRATURE_H\n')
    g.write('#define LEGENDRE_GAUSS_QUADRATURE_H\n')
    g.write('IMPORTIT void leggauss_quad(int n, double *points, double *weights);\n')
    g.write('#endif /** LEGENDRE_GAUSS_QUADRATURE_H */')
