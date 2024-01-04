"""
Summary of all the variables used

@author: Nathan
"""

'''
u,v,w = c * f(xi) * g(eta)
c = amplitude or coeff
f,g are functions

i,j = For x dir i.e. u  +   row-wise positions within matrix 
    j = outer (larger increments)
    i = inner (smaller increments - handles u v w)
            j = 1          j = 2          j = 3 ...... j = n (n terms since j controls y)
        ---------------|------------|-----------|
        i=1 i=2 i=3
        ---|---|---|..
      (u,v,w)
      
i controls f; j controls g
        
k,l = For y dir i.e. v  +  col-wise positions within matrix
    l = outer (larger increments)
    k = inner (smaller increments - handles u v w)

m,n = no of terms in x and y

row0 = starting row of this submatrix in the global matrix

p = prime
'''