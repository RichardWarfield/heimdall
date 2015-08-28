import numpy as np
import time

def g(n):
    for i in range(n):
        e = 2500
        t = time.time()
        x = f1(540, e, 290)
        print "g iter %i Took %f seconds, sum is %f" %(i, time.time() - t, x.sum())
    print x.shape


def f1(a,b,c):
    rng = np.random.RandomState(55)
    s = rng.uniform(size=(a,b))
    t = rng.uniform(size=(b,c))
    res = 0

    for i in range(a):
        for j in range(c):
            res = res + np.dot(s[i], t[:,j])

    return res

