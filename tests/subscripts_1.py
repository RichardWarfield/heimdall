
import numpy as np
import time

def g(n):
    for i in range(n):
        e = 2500
        t = time.time()
        x = f1(1400, e, 2900)
        print "g iter %i Took %f seconds, sum is %f" %(i, time.time() - t, x.sum())
    print x.shape


def f1(a,b,c):
    rng = np.random.RandomState(55)
    s = rng.uniform(size=(a,b))
    t = rng.uniform(size=(b,c))
    res = rng.uniform(size=(1,a,c))

    res[0,:,:] = np.dot(s, t)

    return res

