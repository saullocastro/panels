"""
Summary of all the variables used

@author: Nathan
"""

'''
u,v,w = c * f(xi) * g(eta)
c = amplitude or coeff
f,g are functions of x and y resp

i,j = row-wise positions for a single col within matrix 
    j = outer (larger increments)
    i = inner (smaller increments - handles u v w)
    
    This should be in the rows
            j = 0                  j = 1          j = 2 ...... j = n-1 (n terms since j controls y)
        -----------------------|------------|-----------|
        
        i=0 i=1 i=2 .... i = m-1 (m terms since i controls x)
        ---|---|---|..         |i=0 i=1 ... m-1
      (u,v,w)
      
i controls f(x); j controls g(y) 
    So this gives distribution of fu_i * gu_j where each term in the matrix is 1 term in the SF multiplication
        
k,l = col-wise positions for a single row within matrix
    l = outer (larger increments)
    k = inner (smaller increments - handles u v w)
    Similarly k controls f, l controls g
    So this gives distribution of fv_k * gv_l where each term in the matrix is 1 term in the SF multiplication

When both (fu_i * gu_j)*(fv_k * gv_l) multiplies (eq 31 from MD paper) then 4 vars controls each term

m,n = no of terms in x and y
nx, ny = Number of integration points along `x` and `y`, respectively, for the Legendre-Gauss quadrature rule 
    applied in the numerical integration
row0 = starting row of this submatrix in the global matrix

p = prime
ABDnxny = constitutive relations for the laminate at each integration point.
    Must be a 4-D array of shape ``(nx, ny, 6, 6)`` when using CLT models.
NLgeom : bool, Flag to indicate if geometrically non-linearities should be considered.


default boundary conditions:
    # Controls disp/rotation at boundaries i.e. flags
    # 0 = no disp or rotation
    # 1 = disp or rotation permitted
    
    # x1 and x2 are limits of x -- represent BCs with lines x = const
    # y1 and y2 ............. y -- .................. lines y = const
# - displacement at 4 edges is zero
# - free to rotate at 4 edges (simply supported by default)


'''