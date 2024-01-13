"""
Summary of all the variables used

@author: Nathan
"""

'''
u,v,w = c * f(xi) * g(eta)
c = amplitude or coeff
f,g are functions

i,j = row-wise positions within matrix 
    j = outer (larger increments)
    i = inner (smaller increments - handles u v w)
            j = 1                  j = 2          j = 3 ...... j = n (n terms since j controls y)
        -----------------------|------------|-----------|
        
        i=1 i=2 i=3 .... i = m (m terms since i controls x)
        ---|---|---|..         |i=1 i=2 ... m
      (u,v,w)
      
i controls f; j controls g 
    So this gives distribution of fu_i * gu_j where each term in the matrix is 1 term in the SF multiplication
        
k,l = col-wise positions within matrix
    l = outer (larger increments)
    k = inner (smaller increments - handles u v w)
    Similarly k controls f, l controls g
    So this gives distribution of fv_k * gv_l where each term in the matrix is 1 term in the SF multiplication

When both (fu_i * gu_j)*(fv_k * gv_l) multiplies (???) then 4 vars controls each term

m,n = no of terms in x and y
nx, ny = Number of integration points along `x` and `y`, respectively, for the Legendre-Gauss quadrature rule 
    applied in the numerical integration
row0 = starting row of this submatrix in the global matrix

p = prime
ABDnxny = constitutive relations for the laminate at each integration point.
    Must be a 4-D array of shape ``(nx, ny, 6, 6)`` when using CLT models.
NLgeom : bool, Flag to indicate if geometrically non-linearities should be considered.

'''