# from itertools import izip

ppath = 'out_legendre_gauss_quadrature_points.txt'
wpath = 'out_legendre_gauss_quadrature_weights.txt'
out = {}
with open(ppath) as pfile, open(wpath) as wfile:
    myiter = zip(pfile, wfile)
        # Zips the entire files together - size: (something x 2)
    for pline, wline in myiter:
        if pline.startswith('"n'):
            n = int(pline.replace('"', '').split('=')[1])
            # Removes " and splits about =, then takes the 2nd part i.e. the number
            out[n] = []
                # Creates all keys based on n
            for i in range(n):
                # Runs through n i.e. n points or weights
                pt, wg = next(myiter)
                    
                out[n].append([pt.strip(), wg.strip()])
                # .strip removes all spaces at the beginning and at the end
# %%
for v in out.values():
    v.sort(key=lambda x: float(x[0]))

# with open('../../../compmech/lib/src/legendre_gauss_quadrature.c', 'wb') as f:
with open('legendre_gauss_quadrature.c', 'w') as f:
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
    
    # %% 

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
