import numpy as np
import time

def g(n):
    for i in range(n):
        e = 50
        t = time.time()
        x = f1(140, e, 290)
        print "g iter %i Took %f seconds, sum is %f" %(i, time.time() - t, x.sum())
    print x.shape


def f1(a,b,c):
    rng = np.random.RandomState(55)
    s = rng.uniform(size=(a,b))
    t = rng.uniform(size=(b,c))
    res = 0.

    for i in range(a):
        for j in range(b):
            for k in range(c):
                res = res + s[i,j]*t[j,k]

    return np.array(res)

