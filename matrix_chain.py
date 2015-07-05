import numpy as np
import sys



def matrix_chain_tree(shapes):

    n = len(shapes)
    m = np.zeros((n,n), dtype=np.int64) # Best cost for multiping shapes i to j (inclusive)
    s = np.zeros((n,n), dtype=np.int64) # Location of the best split for multiplying shapes i to j

    def matrix_chain_inner(i,j):
        if m[i,j] != 0:
            return m[i,j]
        if i == j:
            return 0

        m[i,j] = sys.maxint
        for k in range(i+1, j+1):
            q = matrix_chain_inner(i,k-1) + matrix_chain_inner(k,j) + shapes[i][0]*shapes[k][0]*shapes[j][1]
            if q < m[i,j]:
                m[i,j] = q
                s[i,j] = k

        return m[i,j]

    matrix_chain_inner(0, len(shapes)-1)

    def make_tree(i,j):
        split = s[i,j]
        if j == i:
            return i
        elif j-i == 1:
            return (i,j)
        return (make_tree(i,split-1), make_tree(split, j))

    return make_tree(0, len(shapes)-1)
